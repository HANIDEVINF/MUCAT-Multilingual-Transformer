# Literature Review — MuCAT (CERIST – UbiSys, DTISI)

**Title:** Multilingual classification with calibrated uncertainty — Transformer backbones, pooling, uncertainty quantification, and NLP for Algerian dialects and Berber languages  
**Institution:** CERIST, Centre de Recherche sur l'Information Scientifique et Technique — UbiSys Team, DTISI  
**Date:** August 12, 2026  

## Abstract
This document surveys the components required to design a language-conditioned, multilingual text classification system with native uncertainty quantification, applied to a low-resource setting: Modern Standard Arabic, Algerian Arabic, Kabyle, Chaoui, French, and English. Four axes are covered: (1) pretrained multilingual Transformer backbones, (2) sequence pooling strategies, (3) uncertainty quantification in deep learning, and (4) existing work on Algerian dialects and Berber languages. A fifth axis addresses training efficiency (layer freezing, layer-wise learning rate decay), directly relevant to the compute-time constraint of the internship.

## 1. Pretrained Multilingual Backbones
- **1.1 mBERT (Devlin et al., 2019):** Trained on 104 languages from Wikipedia without an explicit cross-lingual alignment objective; systematically outperformed by XLM-R on low-resource languages.
- **1.2 XLM-RoBERTa (Conneau et al., 2020):** Trained on 100 languages from 2.5 TB of filtered CommonCrawl data with a 250k-token SentencePiece tokenizer.
- **1.3 DeBERTa-v3 (He et al., 2023):** Introduces disentangled attention (separate content and relative position vectors) and ELECTRA-style replaced token detection pretraining (dense learning signal on 100% of tokens instead of 15% masked).
- **1.4 mmBERT (Marone et al., 2025):** Built on ModernBERT (alternating local/global attention, rotary embeddings, Flash Attention), trained on 3T+ tokens across 1,800+ languages with annealed language learning.
- **1.5 Practical Backbone Choice:** Because HuggingFace `transformers` v5 removed `TFAutoModel` and `mmBERT` has no native TensorFlow implementation, `XLM-RoBERTa-base` is selected as the defensible public BERT comparison baseline alongside `mDeBERTa-v3-base`.

## 2. Sequence Pooling Strategies
- **2.1 [CLS] Pooling:** Default reference since BERT; filters punctuation/padding better than naive mean-pooling on short sentences.
- **2.2 MaxPoolBERT (2025):** Demonstrates that adding a multi-head attention layer before classification significantly improves low-resource performance where raw `[CLS]` is insufficient.

## 3. Uncertainty Quantification in Deep Learning
- **3.1 The Calibration Problem (Guo et al., 2017):** Softmax confidence is poorly calibrated; temperature scaling requires a separate held-out validation set.
- **3.2 Evidential Deep Learning (Sensoy et al., 2018):** Predicts Dirichlet concentration parameters $\alpha_i$, total evidence strength $S = \sum_i \alpha_i$, and vacuity uncertainty $u = K/S$ in a single forward pass.
- **3.3 Honest Limitation:** Recent (2023) studies show Dirichlet evidential signals can partially correlate with dataset misclassification biases rather than pure epistemic uncertainty.
- **3.4 Alternative — MC-Dropout (Gal & Ghahramani, 2016):** Requires $10\text{–}50\times$ forward passes at inference time, making it unsuitable for real-time production.

## 4. NLP for Algerian Dialects and Berber Languages
- **4.1 DziriBERT & chDzDT (Abdaoui et al.):** Pretrained models for Algerian Arabic (Darija) leveraging Tatoeba data.
- **4.2 PADIC & AraDial (Meftouh et al.):** Multi-dialect Arabic corpora (Algiers, Annaba, Tunisian, Moroccan, Syrian, Palestinian, MSA).
- **4.3 GlotLID (Kargaran et al.):** Low-resource language identification system documenting expected confusions within the Berber family (Kabyle and Chaoui).
- **4.4 Central Takeaway:** No prior work combines a multilingual DeBERTa backbone, hierarchical attention pooling, explicit language/script conditioning, and a native calibrated uncertainty head for this 6-language setting.

## 5. Training Efficiency: Layer Freezing and LLRD
- **5.1 Selective Layer Freezing:** Freezing the first 9 of 12 encoder layers costs only 0.04 accuracy points compared to full fine-tuning.
- **5.2 Layer-wise Learning Rate Decay (LLRD, Sun et al., 2019):** Multiplicative decay factor $\xi \approx 0.95$ from top to bottom layers prevents catastrophic forgetting.
