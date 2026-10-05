# MUCAT — Multilingual Custom Attention Transformer (Arabic / French / English)

[![Live Interactive Studio](https://img.shields.io/badge/Interactive_Demo-MUCAT_Attention_Studio-8b5cf6?style=for-the-badge)](https://portfolio-22yl3rwpj-hanighena4-gmailcoms-projects.vercel.app/lab/mucat-transformer)
[![Stack](https://img.shields.io/badge/Stack-PyTorch_%7C_Transformers_%7C_Multilingual_NLP-06b6d4?style=for-the-badge)](https://github.com/HANIDEVINF/MUCAT-Multilingual-Transformer)

## Architectural Overview
Standard multilingual encoders (`mBERT`, `XLM-R`) frequently suffer from representation degradation on **Arabic, French, and English code-switched text** common in North African / Maghreb clinical and enterprise environments due to reliance on single-token `[CLS]` pooling and dominant-language interference.

**MUCAT** (engineered during AI research at **CERIST**) introduces two dedicated modules atop a shared Transformer backbone:
1. **Hierarchical Subword & Span Attention Pooler (`HierarchicalAttentionPooler`)**: Combines learned token-level importance weights $\alpha_t$ with grouped $1\text{D}$ convolutional morphological span features to capture rich Arabic root-pattern morphology and French-Arabic phrase boundaries.
2. **Language-Sensitive Gating (`LanguageSensitiveGate`)**: Computes a soft routing distribution over $(\text{AR}, \text{FR}, \text{EN})$ experts and applies a learned sigmoid gate $g = \sigma(W_g [h_{\text{shared}} \parallel h_{\text{routed}}] + b_g)$ before the classification head.

## Repository Structure
- [`src/mucat_architecture.py`](src/mucat_architecture.py) — Complete PyTorch implementation (`MUCATConfig`, `HierarchicalAttentionPooler`, `LanguageSensitiveGate`, `MUCATForMultilingualClassification`).
- **Interactive Browser Playground**: Test live Arabic/French/English code-switched sentences and inspect token attention weights $\alpha_i$ and language gate activations in the [Live MUCAT Studio](https://portfolio-22yl3rwpj-hanighena4-gmailcoms-projects.vercel.app/lab/mucat-transformer).
