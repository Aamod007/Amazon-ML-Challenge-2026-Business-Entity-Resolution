# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

A production-grade, reproducible pipeline that matches deduplicated reference business records (Source 1) against noisy entity fragments (Source 2 and Source 3) using name, address, and country fields, optimized for **macro-averaged per-entity $F_{0.5}$** (precision weighted 2× over recall, singletons scored 1.0/0.0).

---

## 1. Compliance & Constraints

- **100% Offline**: Zero external lookups, APIs, geocoding services, or internet access.
- **Model Licensing**: Built with **XGBoost (Apache-2.0 License)** and **RapidFuzz (MIT License)**. Total parameters are $< 50,000$ tree decision nodes (well within the $\le 8\text{B}$ parameter limit).
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
│   ├── data.py               # TSV loaders, ground truth parser, streaming generators
│   ├── normalize.py          # Stage 2: Unicode, legal suffix, and address parsing
│   ├── blocking.py           # Stage 3: Multi-strategy candidate generation & indexing
│   ├── features.py           # Stage 4: Deterministic pairwise string & structured features
│   ├── model.py              # Stage 5: Pairwise XGBoost classifier with GroupKFold
│   ├── evaluate.py           # Stage 6: Macro F0.5 evaluation and threshold sweeping
│   ├── consistency.py        # Stage 7: Global consistency conflict resolution
│   └── pipeline.py           # Stage 8: Master end-to-end CLI orchestrator
├── README.md                 # Reproduction instructions and architecture guide
└── requirements.txt          # Pinned dependency environment
```

---

## 3. Environment Setup

Install pinned dependencies from `requirements.txt`:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 4. End-to-End Reproduction Instructions

To run the complete pipeline end-to-end (Train model $\to$ Candidate generation $\to$ Feature engineering $\to$ Inference $\to$ Global consistency $\to$ Submission validation):

### Option A: Complete Run (All Stages)
```bash
python -m code.business_entity_resolution.src.pipeline --mode all
```

### Option B: Stage-by-Stage Execution

1. **Train Model & Tune Threshold**:
   ```bash
   python -m code.business_entity_resolution.src.pipeline --mode train --n-train 15000
   ```

2. **Generate Test Candidates & Final Matches**:
   ```bash
   python -m code.business_entity_resolution.src.pipeline --mode inference
   ```

3. **Validate Submission Files**:
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
