# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0 / MIT](https://img.shields.io/badge/License-Apache_2.0_%2F_MIT-green.svg)](https://opensource.org/licenses/)
[![Offline Certified](https://img.shields.io/badge/Inference-100%25_Offline-success.svg)](https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution)
[![Validation F0.5](https://img.shields.io/badge/Validation_Macro_F0.5-98.01%25-brightgreen.svg)](https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution)
[![Parameters](https://img.shields.io/badge/Parameters-%3C15k_nodes_(%E2%89%AA8B)-orange.svg)](https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution)

An enterprise-grade, 100% offline, competitive machine learning solution engineered for the **Amazon ML Challenge 2026: Business Entity Resolution**. 

This system resolves noisy, multi-source commercial entity fragments (**Source 2 & Source 3**) against deduplicated canonical reference business records (**Source 1**) across diverse geographic domains (United States, India, and France) using only business names, addresses, and country labels.

---

## 🏆 Benchmark & Competitive Performance

Optimized directly for the official competition metric — **Macro-averaged per-entity $F_{0.5}$** (precision weighted 2× over recall, singletons scored 1.0/0.0):

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

| Model / Milestone | Evaluation Split | Macro $F_{0.5}$ Score | Precision | Recall | Singleton Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trivial Baseline** (All Singletons) | Held-Out Validation | 0.05541 | 0.00% | 0.00% | 100.0% |
| **XGBoost Depth-wise Baseline** | 5-Fold CV OOF | 0.97262 | 97.41% | 96.68% | 96.39% |
| **LightGBM Leaf-wise Baseline** | 5-Fold CV OOF | 0.97254 | 97.35% | 96.72% | 96.39% |
| **CatBoost Oblivious Baseline** | 5-Fold CV OOF | 0.97220 | 97.28% | 96.65% | 96.39% |
| **Tri-Model Ensemble (Pre-Stage 7)** | Held-Out Validation | 0.97928 | 98.11% | 97.25% | 96.39% |
| **Tri-Model Ensemble + Stage 7 Post-Proc** | **Held-Out Validation** | **0.98006 (98.01%)** | **98.24%** | **97.10%** | **97.59%** |

- **5-Fold Cross-Validation (Mean $\pm$ Std):** **$0.97342 \pm 0.00409$**
- **Candidate Blocking Recall Ceiling:** **96.01%** (5,048 / 5,258 ground truth pairs captured with $>99.93\%$ space reduction)
- **Net Relative Improvement:** **+1,668.8%** over baseline

---

## 🏛️ The 6 Competitive Development Pillars

Our pipeline systematically operationalizes all 6 engineering pillars required to achieve and sustain top leaderboard ranking:

1. **Pillar 1 — Strict Official-Structure Data Splitting (`data.py`):**
   - Exact singleton ratio preservation (5.58% unmerged reference entities).
   - Stratified country distribution matching (US ~60%, India ~40%).
   - Zero entity leakage: grouped strictly at the Source 1 entity level.
   - Independent validation candidate pool with realistic negative distractors.

2. **Pillar 2 — 55-Dimensional Country-Agnostic Feature Engineering (`features.py`):**
   - Deterministic, zero-network signals spanning raw, legal-stripped root, and transliterated strings.
   - Token sort, token set, partial edit distances, Jaro-Winkler, first-token brand match, and character 2/3-grams.
   - Hierarchical postal code matching (prefix-2, prefix-3, full PIN) and logarithmic street number distance.
   - Non-linear interaction signals: harmonic mean $\frac{2 S_{\text{name}} S_{\text{addr}}}{S_{\text{name}} + S_{\text{addr}}}$, weakest-link minimum, and dual similarity indicators.
   - Ultra-fast C++ vector extraction (>11,800 pairs/sec via RapidFuzz).

3. **Pillar 3 — Calibrated Class Imbalance Weighting (`model.py`):**
   - Implements square-root ratio weighting:
     $$\text{scale\_pos\_weight} = \min\left(12.0, \max\left(2.0, 1.5 \times \sqrt{\frac{N_{\text{neg}}}{N_{\text{pos}}}}\right)\right) \approx 3.13$$
   - Prevents sigmoid probability saturation near 1.0, enabling continuous, highly discriminative probability ranking for optimal threshold optimization ($\tau^* = 0.830$).

4. **Pillar 4 — 5-Fold Grouped Cross-Validation (`model.py`):**
   - Employs `GroupKFold` partitioned strictly by `source1_entity_id`.
   - Prevents data leakage between candidate pairs of the same reference entity across training and validation folds.

5. **Pillar 5 — Automated Experiment Tracking (`tracker.py`):**
   - Logs hyperparameter configurations, feature counts, CV scores, validation metrics, and model paths into `experiments/experiment_tracker.csv` and `experiments/experiment_log.json`.
   - Inspectable anytime via the CLI: `python code/business_entity_resolution/src/pipeline.py --mode track`.

6. **Pillar 6 — Tri-Model GBDT Ensemble & Global Consistency Resolution (`model.py`, `consistency.py`):**
   - Blends three structurally diverse gradient boosting paradigms:
     - **XGBoost (40%):** Depth-wise histogram splitting (Apache-2.0).
     - **LightGBM (35%):** Leaf-wise gradient-based one-side sampling (MIT).
     - **CatBoost (25%):** Oblivious symmetric decision trees (Apache-2.0).
   - **Stage 7 Post-Processing:** Enforces physical domain constraint (1-to-at-most-1 fragment mapping) by resolving candidate assignment conflicts in favor of the highest confidence reference entity, eliminating false merges and boosting macro $F_{0.5}$.

---

## 📁 Repository Directory Structure

```text
Amazon-ML-Challenge-2026-Business-Entity-Resolution/
├── README.md                                 # Root project documentation & overview
├── Documentation_template.md                 # Official methodology template for final submission
├── PIPELINE_ARCHITECTURE_AND_DATASET_GUIDE.md# Complete architectural manual & dataset guide
├── business_entity_resolution_pipeline.ipynb # Interactive, GPU/local ready Jupyter Notebook
├── code/                                     # Official submission code directory
│   └── business_entity_resolution/
│       ├── README.md                         # Reproduction guide and CLI documentation
│       ├── requirements.txt                  # Pinned production dependencies
│       └── src/                              # Modular production source code
│           ├── __init__.py
│           ├── blocking.py                   # Stage 3: Multi-strategy inverted index blocking
│           ├── config.py                     # Centralized paths and hyperparameters
│           ├── consistency.py                # Stage 7: Global consistency conflict resolution
│           ├── data.py                       # Stage 1: Official-structure data loaders & splitters
│           ├── eda_report.py                 # Exploratory data analysis & noise profiling
│           ├── evaluate.py                   # Stage 6: Macro F0.5 metric & threshold sweep
│           ├── features.py                   # Stage 4: 55-dimensional deterministic feature suite
│           ├── model.py                      # Stage 5: Tri-model ensemble & GroupKFold CV
│           ├── normalize.py                  # Stage 2: Multilingual normalization & regex
│           ├── pipeline.py                   # Stage 8: Master end-to-end CLI orchestrator
│           └── tracker.py                    # Experiment tracking engine (CSV + JSON)
├── docs/                                     # Official competition materials
│   ├── amazon_ml_challenge_problem_statement.pdf
│   └── guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf
├── experiments/                              # Experiment tracking logs & audit trails
│   ├── experiment_log.json                   # Detailed JSON run configurations & metrics
│   └── experiment_tracker.csv                # Tabular experiment audit spreadsheet
├── models/                                   # Serialized model weights & bundle metadata
│   └── ensemble_matching_model/
│       ├── ensemble_matching_model_cat.cbm   # CatBoost model weights
│       ├── ensemble_matching_model_lgb.txt   # LightGBM model weights
│       ├── ensemble_matching_model_metadata.json # Feature manifest & hyperparameter specs
│       └── ensemble_matching_model_xgb.json  # XGBoost model weights
├── output/                                   # Generated competition submission outputs
│   ├── candidate_pairs.tsv                   # Stage 3 candidate blocking pairs
│   └── matching_results.tsv                  # Final resolved predictions (TSV)
└── student_resource/                         # Official evaluation tools & validation script
    ├── README.md                             # Competition guidelines
    └── utils/
        └── validate_submission.py            # Official submission validation script
```

---

## 💾 Dataset Download & Mirror

Download the complete Amazon ML Challenge 2026 dataset from the verified Google Drive mirror:
- **Google Drive Dataset Mirror:** [Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1L21j0i0xjc14bRVLgL0Be40Ijz1_MiQv?usp=sharing)
- **Folder ID:** `1L21j0i0xjc14bRVLgL0Be40Ijz1_MiQv`

Extract the files into `student_resource/dataset/` (or `dataset/`):
```text
student_resource/dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 🚀 Quick Start & Reproduction

### 1. Installation
Ensure Python 3.10+ is installed. Install all pinned dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. End-to-End Execution (CLI)
Run the entire pipeline from scratch (split generation, feature extraction, 5-fold CV, ensemble training, threshold optimization, and test inference):
```bash
python code/business_entity_resolution/src/pipeline.py --mode all
```

### 3. Dedicated Execution Modes
- **Validate Current Ensemble:**
  ```bash
  python code/business_entity_resolution/src/pipeline.py --mode validate
  ```
- **Train Ensemble on Official Split:**
  ```bash
  python code/business_entity_resolution/src/pipeline.py --mode train
  ```
- **Run Full Test Inference:**
  ```bash
  python code/business_entity_resolution/src/pipeline.py --mode inference
  ```
- **View Experiment Tracking Spreadsheet:**
  ```bash
  python code/business_entity_resolution/src/pipeline.py --mode track
  ```

### 4. Interactive Jupyter Notebook
An interactive walkthrough ready for local machines or Kaggle GPU instances (T4 / P100 / A100) is located at:
```bash
jupyter notebook business_entity_resolution_pipeline.ipynb
```

---

## 🔒 Hard Constraints & Competition Compliance

| Constraint | Requirement | Pipeline Implementation | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Model Size** | $\le 8\text{ Billion}$ parameters | Total ensemble nodes: ~10,440 ($< 0.000015\text{B}$) | **PASS (100% compliant)** |
| **Model Licensing** | Permissive (Apache-2.0, MIT, BSD) | XGBoost (Apache-2.0), LightGBM (MIT), CatBoost (Apache-2.0), RapidFuzz (MIT), Polars (MIT) | **PASS (100% compliant)** |
| **Network Access** | 100% Offline inference | Zero web APIs, zero external geocoders, all features computed purely in-memory | **PASS (100% compliant)** |
| **Country Generalization** | Open string (including unseen `France`) | All normalization, blocking, and feature extraction operate country-agnostically without hardcoded splits | **PASS (100% compliant)** |
| **Output Format** | Strict Tab-Separated Values (`.tsv`) | All read/write operations strictly enforce `sep="\t"` | **PASS (100% compliant)** |

---

## 📦 Final Submission Packaging

To generate and validate the official submission archive required by the competition organizers:

```bash
# 1. Verify output format with official validation utility
python student_resource/utils/validate_submission.py --output-dir output/ --test-dir student_resource/dataset/test/

# 2. Package required submission components
zip -r DataResolvers_submission.zip output/ code/business_entity_resolution/ Documentation_template.md
```

The resulting zip archive conforms exactly to the organizer's layout specification:
```text
DataResolvers_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 📄 License
This repository and all included source code are licensed under the [Apache 2.0 License](LICENSE) and [MIT License](LICENSE).
