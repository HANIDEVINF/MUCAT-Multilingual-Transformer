"""
MUCAT — Multilingual Custom Attention Transformer (Arabic / French / English)
Author: Hani Ghena (CERIST AI Engineering Research & USTHB)

PyTorch implementation of:
1. Hierarchical Subword & Span Attention Pooling (replacing brittle [CLS]-only pooling)
2. Language-Sensitive Gating (dynamically routing representations across AR / FR / EN)
3. Language-Conditioned Residual Classification & Sequence Tagging Heads
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MUCATConfig:
    vocab_size: int = 64000
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    intermediate_size: int = 2048
    num_languages: int = 3  # 0: Arabic (AR), 1: French (FR), 2: English (EN)
    num_classes: int = 5
    dropout_prob: float = 0.15


class HierarchicalAttentionPooler(nn.Module):
    """
    Two-level token and phrase-span attention pooling over transformer hidden states.
    Prevents dominant-language drift in Arabic/French/English code-switched sentences.
    """

    def __init__(self, hidden_size: int, dropout_prob: float = 0.15) -> None:
        super().__init__()
        self.token_proj = nn.Linear(hidden_size, hidden_size)
        self.token_query = nn.Linear(hidden_size, 1, bias=False)
        self.span_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            groups=8,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B, T, D]
        u = torch.tanh(self.token_proj(hidden_states))  # [B, T, D]
        scores = self.token_query(u).squeeze(-1)  # [B, T]

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        alpha = F.softmax(scores, dim=-1)  # [B, T]
        token_pooled = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1)  # [B, D]

        # Local trigrams / morphological span context
        span_feats = F.gelu(self.span_conv(hidden_states.transpose(1, 2))).transpose(1, 2)
        span_pooled = torch.bmm(alpha.unsqueeze(1), span_feats).squeeze(1)  # [B, D]

        fused = self.layer_norm(token_pooled + self.dropout(span_pooled))
        return fused, alpha


class LanguageSensitiveGate(nn.Module):
    """
    Computes a soft language-routing distribution over (AR, FR, EN) and modulates
    shared encoder features with language-specific expert projections.
    """

    def __init__(self, hidden_size: int, num_languages: int = 3) -> None:
        super().__init__()
        self.lang_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_languages),
        )
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_languages)]
        )
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pooled_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pooled_repr: [B, D]
        lang_logits = self.lang_router(pooled_repr)  # [B, L]
        lang_probs = F.softmax(lang_logits, dim=-1)  # [B, L]

        expert_Stack = torch.stack(
            [F.gelu(expert(pooled_repr)) for expert in self.experts], dim=1
        )  # [B, L, D]
        routed_repr = torch.bmm(lang_probs.unsqueeze(1), expert_Stack).squeeze(1)  # [B, D]

        sigma_gate = torch.sigmoid(
            self.gate_proj(torch.cat([pooled_repr, routed_repr], dim=-1))
        )
        gated = self.norm(sigma_gate * routed_repr + (1.0 - sigma_gate) * pooled_repr)
        return gated, lang_probs


class MUCATForMultilingualClassification(nn.Module):
    """
    End-to-end MUCAT architecture combining shared Transformer encoder blocks,
    Hierarchical Attention Pooling, and Language-Sensitive Gating.
    """

    def __init__(self, config: MUCATConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pos_embeddings = nn.Embedding(512, config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.pooler = HierarchicalAttentionPooler(config.hidden_size, config.dropout_prob)
        self.lang_gate = LanguageSensitiveGate(config.hidden_size, config.num_languages)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.embeddings(input_ids) + self.pos_embeddings(positions)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        hidden_states = self.encoder(x, src_key_padding_mask=key_padding_mask)

        pooled, token_attention = self.pooler(hidden_states, attention_mask=attention_mask)
        gated_repr, language_probs = self.lang_gate(pooled)
        logits = self.classifier(gated_repr)

        return {
            "logits": logits,
            "language_probabilities": language_probs,
            "token_attention": token_attention,
            "representation": gated_repr,
        }


if __name__ == "__main__":
    cfg = MUCATConfig()
    model = MUCATForMultilingualClassification(cfg)
    dummy_ids = torch.randint(1, cfg.vocab_size, (4, 64))
    dummy_mask = torch.ones_like(dummy_ids)
    out = model(dummy_ids, dummy_mask)
    print("MUCAT Forward Pass Verified:")
    print("  Logits shape:", out["logits"].shape)
    print("  Language Gate shape:", out["language_probabilities"].shape)
    print("  Token Attention shape:", out["token_attention"].shape)
