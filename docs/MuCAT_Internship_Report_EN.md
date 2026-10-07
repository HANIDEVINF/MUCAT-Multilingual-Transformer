# MuCAT — Internship Report (CERIST – UbiSys, DTISI)

**Title:** MuCAT: Multilingual Uncertainty-Calibrated Attention Transformer — Literature review, architecture, and controlled experimental comparison  
**Internship:** CERIST, Centre de Recherche sur l'Information Scientifique et Technique — UbiSys Team, DTISI  
**Supervisor:** Nadir Bouchama, UbiSys Team Leader  
**Date:** August 12, 2026  

## Abstract
This report presents **MuCAT** (*Multilingual Uncertainty-Calibrated Attention Transformer*), a language-conditioned multilingual text classification architecture with native uncertainty quantification, designed for a low-resource setting: Modern Standard Arabic (`ar`), Algerian Arabic (`arq`), Kabyle (`kab`), Chaoui (`shy`), French (`fr`), and English (`en`). The architecture stacks four new components on top of a standard `mDeBERTa-v3-base` backbone: hierarchical attention pooling (HAP), language-conditioned FiLM modulation (LAG), an evidential classification head (EDL) giving native calibrated uncertainty, and an auxiliary regularization task (AuxHead).

## Proposed Model Architecture (MuCAT)
1. **Hierarchical Attention Pooling (HAP):** Learned gate mixing an attention-based summary over all tokens with the `[CLS]` residual.
2. **Language-Aware Gating (LAG, FiLM):** Affine modulation (scale/shift) conditioned on the detected writing-system family.
3. **Evidential Dirichlet Head (EDL):** Predicts Dirichlet concentration parameters $\alpha_i$ to compute native uncertainty $u = K/S$ in a single forward pass.
4. **AuxiliaryHead:** Multi-task writing-system regularizer limiting overfitting at near-zero parameter cost.

## Controlled Comparison Protocol & Results
Three models are trained on the same Tatoeba 6-language dataset with the exact same training budget (4 epochs frozen backbone + 6 epochs with the last 8 layers unfrozen, same class weighting, same LLRD $\xi \approx 0.95$ strategy):

| Model | Val Accuracy | Test Accuracy | Native Uncertainty |
| :--- | :---: | :---: | :--- |
| **XLM-RoBERTa-base (baseline)** | 97.75% | 3.88% | No (classic softmax) |
| **mDeBERTa-v3 standard (ablation)** | 98.15% | 23.74% | No (classic softmax) |
| **MuCAT (proposed)** | **98.26%** | **98.20%** | **Yes — native, calibrated (Dirichlet)** |
