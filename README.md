# MuCAT — Multilingual Uncertainty-Calibrated Attention Transformer

**Literature Review, Architecture, and Controlled Experimental Comparison for Low-Resource & Maghreb NLP**  
**Institution:** CERIST — Centre de Recherche sur l'Information Scientifique et Technique (UbiSys Team, DTISI)  
**Author:** Hani Ghena (Master in Artificial Intelligence, USTHB & CERIST AI Research Intern)  
**Supervisor:** Nadir Bouchama, UbiSys Team Leader  

[![Interactive Studio](https://img.shields.io/badge/Interactive_Demo-MuCAT_Attention_%26_Dirichlet_Studio-8b5cf6?style=for-the-badge)](https://ais-pre-4br5xw5mjzlfdkdvyz7xyq-909472834222.europe-west2.run.app/lab/mucat-transformer)
[![Stack](https://img.shields.io/badge/Backbone-mDeBERTa--v3_%7C_Evidential_Dirichlet_%7C_FiLM_Gating-06b6d4?style=for-the-badge)](https://github.com/HANIDEVINF/MUCAT-Multilingual-Transformer)

---

## Abstract

This repository presents **MuCAT** (*Multilingual Uncertainty-Calibrated Attention Transformer*), a language-conditioned multilingual text classification architecture with native uncertainty quantification, designed for a low-resource setting covering **6 languages**:
- **Modern Standard Arabic** (`ar`)
- **Algerian Arabic / Darija** (`arq`)
- **Kabyle** (`kab`)
- **Chaoui** (`shy`)
- **French** (`fr`)
- **English** (`en`)

The architecture stacks **four new components** on top of a standard pretrained `mDeBERTa-v3-base` backbone:
1. **Hierarchical Attention Pooling (HAP)**
2. **Language-Aware Gating (LAG — FiLM modulation)**
3. **Evidential Dirichlet Classification Head (EDL)** giving native calibrated uncertainty $u = K/S$ in a single forward pass
4. **Auxiliary Script-Discrimination Regularization Head (AuxHead)**

The contribution is evaluated through a **strictly controlled comparison at an identical training budget** (4 epochs frozen backbone + 6 epochs with top-8 layers unfrozen, identical class weighting, identical Layer-wise Learning Rate Decay $\xi \approx 0.95$) against two baselines:
- **`XLM-RoBERTa-base`** (representing the latest technically accessible public BERT model in TensorFlow/Keras pipelines)
- **Standard `mDeBERTa-v3-base` ablation** (same backbone as MuCAT, without the added layers), isolating the exact contribution of the proposed architecture from the backbone and compute budget.

---

## Architecture Pipeline

```text
Input text (ar / arq / kab / shy / fr / en)
                    │
                    ▼
   mDeBERTa-v3-base (standard, pretrained backbone)
                    │
                    ▼
   1 – Hierarchical Attention Pooling (HAP)
                    │
                    ▼
   2 – Language-Aware Gating (LAG — FiLM)
                    │
                    ▼
   3 – Evidential Dirichlet Head (EDL)  +  4 – AuxHead
                    │
                    ▼
   Prediction (class) + calibrated uncertainty u = K/S
```

### Detail of Added Layers and Technical Justification

| Added Layer | Replaces / Improves | Why This Choice Over Alternatives |
| :--- | :--- | :--- |
| **1 – Hierarchical Attention Pooling (HAP)** | Raw `[CLS]` pooling (used by XLM-R and standard mDeBERTa) | A learned gate mixes an attention-based summary over all tokens with the `[CLS]` residual, rather than forcing an *a priori* choice. Recent literature (*MaxPoolBERT*, 2025) demonstrates gains from an extra attention layer on low-resource sets — directly applicable to **Chaoui** and **Kabyle**. |
| **2 – Language-Aware Gating (LAG, FiLM)** | Total absence of language/script conditioning in baselines | An affine modulation ($\gamma \odot h + \beta$, scale/shift) conditioned on the detected writing-system family, rather than retraining per-language tokenizers or stacking separate per-language sub-networks that multiply parameters without guaranteed gain. |
| **3 – Evidential Dirichlet Head (EDL)** | Classic softmax + cross-entropy (both baselines) | Predicts Dirichlet concentration parameters $\alpha_i = e_i + 1$ with total evidence strength $S = \sum_i \alpha_i$ and native vacuity uncertainty $u = K/S$ in a **single forward pass**, avoiding post-hoc temperature scaling validation sets (*Guo et al., 2017*) or $10\text{–}50\times$ forward passes in MC-Dropout (*Gal & Ghahramani, 2016*). |
| **4 – AuxiliaryHead (AuxHead)** | Absence of multi-task regularization in baselines | Forces the model to retain discriminative writing-system information in its internal representation — a regularizing effect that limits overfitting on the main task at near-zero parameter cost. |

---

## Controlled Experimental Results (Tatoeba 6-Language Benchmark)

All three models are trained on the **same dataset** (Tatoeba: `ar`, `arq`, `kab`, `shy`, `fr`, `en`) with the **exact same training budget** (4 epochs frozen backbone + 6 epochs with the last 8 layers unfrozen, selective 9/12 layer freezing analysis, and Layer-wise Learning Rate Decay $\xi \approx 0.95$):

| Model | Validation Accuracy | Test Accuracy | Native Uncertainty |
| :--- | :---: | :---: | :--- |
| **XLM-RoBERTa-base (baseline)** | 97.75% | 3.88% | No (classic softmax) |
| **mDeBERTa-v3 standard (ablation)** | 98.15% | 23.74% | No (classic softmax) |
| **MuCAT (proposed: HAP + LAG + EDL + AuxHead)** | **98.26%** | **98.20%** | **Yes — native, calibrated (Dirichlet $u = K/S$)** |

---

## Repository Structure

- `src/mucat_architecture.py` — Reference implementation of the 4 MuCAT modules (**HierarchicalAttentionPooling**, **LanguageAwareFiLMGating**, **EvidentialDirichletHead**, **AuxiliaryScriptHead**) + Evidential Dirichlet loss and LLRD optimizer builder.
- `docs/MuCAT_Literature_Review_EN.md` — Full English Literature Review (*Multilingual classification with calibrated uncertainty — Transformer backbones, pooling, uncertainty quantification, and NLP for Algerian dialects and Berber languages*).
- `docs/MuCAT_Internship_Report_EN.md` — Full English CERIST UbiSys DTISI Internship Report.
- `docs/MuCAT_Rapport_de_Stage_FR.md` — Full French CERIST UbiSys DTISI Rapport de Stage (*État de l'art, architecture et comparaison expérimentale contrôlée*).

---

## References

1. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
2. Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)*. ACL.
3. He, P., Gao, J., Chen, W. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*. ICLR.
4. Marone, M., Weller, O., Fleshman, W., Yang, E., Lawrie, D., Van Durme, B. (2025). *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning*. arXiv:2509.06888.
5. Sensoy, M., Kaplan, L., Kandemir, M. (2018). *Evidential Deep Learning to Quantify Classification Uncertainty*. NeurIPS.
6. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.
7. Gal, Y., Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML.
8. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI.
9. Sun, C., Qiu, X., Xu, Y., Huang, X. (2019). *How to Fine-Tune BERT for Text Classification?*. CCL.
10. Abdaoui, A., et al. *DziriBERT: A Pre-trained Language Model for the Algerian Dialect*.
11. Meftouh, K., et al. *PADIC: A Parallel Arabic Dialect Corpus*.
12. Kargaran, A. H., et al. *GlotLID: Language Identification for Low-Resource Languages*. EMNLP.
