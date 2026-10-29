# MuCAT — Multilingual Uncertainty-Calibrated Attention Transformer

> **Reproducibility and sharing status (read first):** the public code currently provides the custom attention/gating/evidential-head modules over hidden states supplied by a caller. It does not include the complete pretrained-backbone loading, tokenizer/data preparation, training and end-to-end evaluation pipeline needed to reproduce the reported table from a fresh checkout. The 98.20% figure below is a project-reported result, not independently reproduced from this public repository. Confirm research-sharing permissions with CERIST before publishing additional code, data, weights or detailed methods.

**Literature Review, Architecture, and Controlled Experimental Comparison for Low-Resource & Maghreb NLP**  
**Institution:** CERIST — Centre de Recherche sur l'Information Scientifique et Technique (UbiSys Team, DTISI)  
**Author:** Hani Ghena (M.Sc. in Artificial Intelligence @ USTHB 2025–2027 · B.Sc. Computer Science @ USTHB 2022–2025 · CERIST AI Research Intern)  
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

## Project-Reported Experimental Results (Tatoeba 6-Language Benchmark; Reproduction Pending)

Project notes describe a controlled comparison on Tatoeba (`ar`, `arq`, `kab`, `shy`, `fr`, `en`) using a shared training budget. Because the full training and evaluation pipeline is not included in this public repository, the results below are reported project results and cannot yet be reproduced from a clean checkout:

| Model | Validation Accuracy | Test Accuracy (Aligned Labels) | Test Uncertainty |
| :--- | :---: | :---: | :--- |
| **XLM-RoBERTa-base (baseline)** | 97.75% | 81.40%* | No (classic softmax) |
| **mDeBERTa-v3 standard (ablation)** | 98.15% | 89.60%* | No (classic softmax) |
| **MuCAT (Proposed: HAP + LAG + EDL + AuxHead)** | **98.26%** | **98.20%** | **Yes — Dirichlet Epistemic Vacuity ( = K/S$)** |

> 🔬 **Evaluation Diagnostic & Label-Mapping Resolution Note ():**  
> An earlier version of this benchmark reported an artificial 3.88% test accuracy for XLM-RoBERTa and 23.74% for standard mDeBERTa due to an  dictionary key-permutation bug in the evaluation script where test splits were evaluated against an unsynchronized label index map. When re-evaluated with verified, aligned label mappings under the exact same test splits:
> - **XLM-RoBERTa-base baseline:** achieves **81.40%** test accuracy.
> - **Standard mDeBERTa-v3:** achieves **89.60%** test accuracy.
> - **MuCAT (Proposed):** the project README reports **98.20%** test accuracy (+8.60 percentage points relative to its reported standard mDeBERTa-v3 baseline). This result is not independently reproduced from the current public checkout; the causal contribution of each added module requires reproducible ablation runs.
>
> **Uncertainty quantification note:** the custom head returns Dirichlet concentration parameters and a vacuity-style score `u = K / S` in one forward pass. This mathematical output is not, by itself, proof of empirical calibration; evaluate calibration on held-out data (for example, ECE, Brier score and reliability diagrams).

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
