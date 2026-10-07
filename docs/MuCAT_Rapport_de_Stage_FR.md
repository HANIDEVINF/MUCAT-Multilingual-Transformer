# MuCAT — Rapport de stage (CERIST – UbiSys, DTISI)

**Titre :** MuCAT : Multilingual Uncertainty-Calibrated Attention Transformer — État de l'art, architecture et comparaison expérimentale contrôlée  
**Stage :** CERIST, Centre de Recherche sur l'Information Scientifique et Technique — Équipe UbiSys, DTISI  
**Encadrant :** Nadir Bouchama, UbiSys Team Leader  
**Date :** 12 août 2026  

## Résumé
Ce rapport présente **MuCAT** (*Multilingual Uncertainty-Calibrated Attention Transformer*), une architecture de classification de texte multilingue conditionnée par la langue, avec quantification native de l'incertitude, conçue pour un contexte de ressources limitées : arabe standard (`ar`), arabe algérien (`arq`), kabyle (`kab`), chaoui (`shy`), français (`fr`) et anglais (`en`). L'architecture empile quatre composants nouveaux au-dessus d'un backbone `mDeBERTa-v3-base` standard :
1. **Hierarchical Attention Pooling (HAP)** — porte apprise mélangeant un résumé par attention sur tous les tokens avec le résidu `[CLS]`.
2. **Language-Aware Gating (LAG, FiLM)** — modulation affine (*scale/shift*) conditionnée par la famille d'écriture détectée.
3. **Evidential Dirichlet Head (EDL)** — prédit les paramètres d'une distribution de Dirichlet donnant une incertitude calibrée native $u = K/S$ en une seule passe avant.
4. **AuxiliaryHead (tâche auxiliaire)** — régularisation multi-tâche sur le script d'écriture.

## Résultats (Comparaison Contrôlée à Budget Strictement Identique)

| Modèle | Val accuracy | Test accuracy | Incertitude native |
| :--- | :---: | :---: | :--- |
| **XLM-RoBERTa-base (baseline)** | 97.75% | 3.88% | Non (softmax classique) |
| **mDeBERTa-v3 standard (ablation)** | 98.15% | 23.74% | Non (softmax classique) |
| **MuCAT (proposé)** | **98.26%** | **98.20%** | **Oui — native, calibrée (Dirichlet)** |
