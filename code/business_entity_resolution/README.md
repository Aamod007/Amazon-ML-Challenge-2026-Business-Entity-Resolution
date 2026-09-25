# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

A production-grade, reproducible pipeline that matches deduplicated reference business records (Source 1) against noisy entity fragments (Source 2 and Source 3) using name, address, and country fields, optimized for **macro-averaged per-entity $F_{0.5}$** (precision weighted 2× over recall, singletons scored 1.0/0.0).

---

## 1. Compliance & Constraints

- **100% Offline**: Zero external lookups, APIs, geocoding services, or internet access.
- **Model Licensing**: Built with **XGBoost (Apache-2.0)**, **LightGBM (MIT)**, and **CatBoost (Apache-2.0)**. Total parameters are $< 15,000$ tree decision nodes (well within the $\le 8\text{B}$ parameter limit).
- **Tab-Separated Formatting**: Strictly tab-separated (`sep="\t"`) across all input, intermediate, and submission files.
- **Open-String Country Handling**: No hardcoded `{US, India}` conditionals. Dynamically handles all countries in train and test (including France, unseen during training).
- **Strict Cardinality Post-Processing**: Evaluates and resolves 1-to-many conflicts where an S2/S3 candidate is claimed by multiple S1 entities by assigning it to the single highest-probability match.

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py           # Package indicator
│   ├── config.py             # Central configuration (hyperparameters, paths, regexes)
│   ├── data.py               # TSV loaders, ground truth parser, leakage-free official split
│   ├── normalize.py          # Stage 2: Unicode, legal suffix, and address parsing
│   ├── blocking.py           # Stage 3: TF-IDF weighted multi-strategy candidate ranking
│   ├── features.py           # Stage 4: 55-dimensional deterministic pairwise alignment signals
│   ├── model.py              # Stage 5: Tri-Model Ensemble (XGB+LGBM+CatBoost) with 5-Fold GroupKFold
│   ├── evaluate.py           # Stage 6: Macro F0.5 evaluation and threshold sweeping
│   ├── consistency.py        # Stage 7: Global consistency conflict resolution
│   ├── tracker.py            # Automated experiment tracking (CSV spreadsheet + JSON)
│   └── pipeline.py           # Stage 8: Master end-to-end CLI orchestrator
├── README.md                 # Reproduction instructions and architecture guide
└── requirements.txt          # Pinned dependency environment
```

---

## 3. Dataset Download & Setup

Download the complete Amazon ML Challenge 2026 dataset from the Google Drive mirror:
- **Dataset Link (Google Drive):** [Amazon ML Challenge 2026 Dataset](https://drive.google.com/drive/folders/1L21j0i0xjc14bRVLgL0Be40Ijz1_MiQv?usp=sharing)

Extract or place the dataset TSV files in `student_resource/dataset/` (or `dataset/`):
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

## 4. Environment Setup

Install pinned dependencies from `requirements.txt`:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 5. End-to-End Reproduction Instructions


To run the complete pipeline end-to-end (Train ensemble $\to$ Candidate generation $\to$ Feature engineering $\to$ Inference $\to$ Global consistency $\to$ Submission validation):

### Option A: Complete Run (All Stages)
```bash
python code/business_entity_resolution/src/pipeline.py --mode all --model ensemble
```

### Option B: Stage-by-Stage Execution

1. **Train 5-Fold Ensemble Model & Tune Threshold**:
   ```bash
   python code/business_entity_resolution/src/pipeline.py --mode train --model ensemble --n-train 20000 --n-val 4000 --n-folds 5
   ```

2. **View Experiment Tracking Scoreboard**:
   ```bash
   python code/business_entity_resolution/src/pipeline.py --mode track
   ```

3. **Generate Test Candidates & Final Matches**:
   ```bash
   python code/business_entity_resolution/src/pipeline.py --mode inference
   ```

4. **Validate Submission Files**:
   ```bash
   python student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir student_resource/dataset/test
   ```


---

## 5. Expected Submission Outputs

The pipeline produces two tab-separated files in `output/`:
- **`output/matching_results.tsv`**: Final matched entity IDs (one row per Source 1 test entity).
- **`output/candidate_pairs.tsv`**: Blocking candidate IDs evaluated by the matching model.
