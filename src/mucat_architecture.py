"""
MuCAT — Multilingual Uncertainty-Calibrated Attention Transformer
CERIST — Centre de Recherche sur l'Information Scientifique et Technique (UbiSys Team, DTISI)
Author: Hani Ghena | Supervisor: Nadir Bouchama

Implements the 4 architectural components stacked over mDeBERTa-v3-base:
  1. Hierarchical Attention Pooling (HAP) — learned gate mixing multi-head token attention with [CLS] residual
  2. Language-Aware Gating (LAG — FiLM) — affine scale/shift conditioned on writing-system family
  3. Evidential Dirichlet Head (EDL) — predicts Dirichlet concentration parameters alpha_i = e_i + 1
     yielding single-pass calibrated uncertainty u = K / S where S = sum(alpha_i)
  4. Auxiliary Script Head (AuxHead) — multi-task writing-system regularizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HierarchicalAttentionPooling(nn.Module):
    """
    1. Hierarchical Attention Pooling (HAP)
    Replaces raw [CLS] pooling by learning a gate that mixes a multi-head attention
    summary over all sequence tokens with the [CLS] residual representation.
    """
    def __init__(self, hidden_size: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_Readout = nn.Linear(hidden_size, 1)
        self.residual_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cls_token = hidden_states[:, 0, :]  # (B, H)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_out, _ = self.mha(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
        )
        scores = self.attn_Readout(torch.tanh(attn_out)).squeeze(-1)  # (B, L)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # (B, L)
        pooled_tokens = torch.bmm(weights.unsqueeze(1), attn_out).squeeze(1)  # (B, H)

        gate = torch.sigmoid(self.residual_gate(torch.cat([cls_token, pooled_tokens], dim=-1)))
        mixed = gate * pooled_tokens + (1.0 - gate) * cls_token
        return {
            "pooled": self.layer_norm(mixed),
            "token_weights": weights,
        }


class LanguageAwareFiLMGating(nn.Module):
    """
    2. Language-Aware Gating (LAG — FiLM)
    Applies an affine transformation (gamma * h + beta) conditioned on the detected
    writing-system / script family distribution (Perez et al., 2018).
    """
    def __init__(self, hidden_size: int = 768, num_scripts: int = 3):
        super().__init__()
        self.gamma_proj = nn.Linear(num_scripts, hidden_size)
        self.beta_proj = nn.Linear(num_scripts, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pooled: torch.Tensor, script_prior: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(script_prior)
        beta = self.beta_proj(script_prior)
        modulated = (1.0 + gamma) * pooled + beta
        return self.layer_norm(modulated)


class EvidentialDirichletHead(nn.Module):
    """
    3. Evidential Dirichlet Classification Head (Sensoy et al., 2018)
    Predicts non-negative evidence e_i >= 0 and Dirichlet concentration alpha_i = e_i + 1.
    Provides expected class probabilities p_i = alpha_i / S and native vacuity uncertainty u = K / S
    in a single forward pass without requiring Monte Carlo Dropout or post-hoc validation sets.
    """
    def __init__(self, hidden_size: int = 768, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = F.softplus(self.evidence_head(h))  # e_i >= 0
        alpha = evidence + 1.0                        # alpha_i = e_i + 1
        strength_S = torch.sum(alpha, dim=-1, keepdim=True)  # S = sum(alpha_i)
        probs = alpha / strength_S                    # p_i = alpha_i / S
        vacuity_u = self.num_classes / strength_S.squeeze(-1)  # u = K / S
        return {
            "evidence": evidence,
            "alpha": alpha,
            "strength_S": strength_S.squeeze(-1),
            "probs": probs,
            "uncertainty_u": vacuity_u,
        }


class MuCATClassifier(nn.Module):
    """
    Full MuCAT Architecture (mDeBERTa-v3-base + HAP + LAG FiLM + EDL + AuxHead)
    Target classes (K = 6): Modern Standard Arabic (ar), Algerian Arabic (arq),
    Kabyle (kab), Chaoui (shy), French (fr), English (en).
    """
    def __init__(self, hidden_size: int = 768, num_classes: int = 6, num_scripts: int = 3):
        super().__init__()
        self.hap = HierarchicalAttentionPooling(hidden_size=hidden_size)
        self.lag = LanguageAwareFiLMGating(hidden_size=hidden_size, num_scripts=num_scripts)
        self.edl = EvidentialDirichletHead(hidden_size=hidden_size, num_classes=num_classes)
        # 4. AuxiliaryHead: multi-task writing-system regularizer
        self.aux_head = nn.Linear(hidden_size, num_scripts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        script_prior: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hap_out = self.hap(hidden_states, attention_mask=attention_mask)
        modulated = self.lag(hap_out["pooled"], script_prior)
        edl_out = self.edl(modulated)
        aux_logits = self.aux_head(modulated)
        return {
            **edl_out,
            "aux_logits": aux_logits,
            "token_weights": hap_out["token_weights"],
        }


def get_llrd_parameter_groups(model_layers: nn.ModuleList, head_params, base_lr: float = 2e-5, llrd_factor: float = 0.95):
    """
    Layer-wise Learning Rate Decay (LLRD, Sun et al., 2019) with factor xi ~ 0.95.
    Applies highest learning rate to classification heads and decays multiplicatively
    from top encoder layers down to lower layers to prevent catastrophic forgetting.
    """
    param_groups = [{"params": list(head_params), "lr": base_lr * 2.0}]
    num_layers = len(model_layers)
    for idx, layer in enumerate(reversed(model_layers)):
        layer_lr = base_lr * (llrd_factor ** idx)
        param_groups.append({"params": list(layer.parameters()), "lr": layer_lr})
    return param_groups
