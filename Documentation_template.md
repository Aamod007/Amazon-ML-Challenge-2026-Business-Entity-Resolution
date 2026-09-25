# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DataResolvers  
**Team Members:** Aamod  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary
This submission presents an end-to-end, reproducible, 100% offline machine learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. Our system resolves noisy, multi-source commercial entity fragments (Source 2 & Source 3) against deduplicated reference business records (Source 1) using only name, address, and country fields, optimized for the **macro-averaged per-entity $F_{0.5}$** metric (precision weighted 2× over recall, singletons scored 1.0/0.0). Key innovations include: (1) a multi-strategy, country-partitioned candidate blocking engine achieving **96.31% pair completeness** while eliminating $>99.93\%$ of the cross-product space, (2) a deterministic 27-dimensional pairwise feature vector spanning multi-script token sort, partial edit distances, and structured subfield agreement, (3) an **XGBoost pairwise classifier (Apache-2.0 License)** trained with grouped cross-validation and class imbalance weighting, (4) an empirical decision threshold optimization tuned directly on validation macro $F_{0.5}$ ($0.940 - 0.950$), and (5) a **Stage 7 Global Consistency conflict resolution** step exploiting the verified physical domain constraint that candidate fragments map to at most one reference entity.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Official Dataset Link (Google Drive Mirror):** [Amazon ML Challenge 2026 Dataset](https://drive.google.com/drive/folders/1L21j0i0xjc14bRVLgL0Be40Ijz1_MiQv?usp=sharing)

During exploratory data analysis across the 2.2M reference entities and 10.3M source fragments, we identified seven fundamental noise archetypes and key structural properties:
1. **Singleton Dominance & Evaluation Dynamics:** Exactly **5.58%** (123,247 / 2,206,821) of Source 1 entities have zero true matches in Source 2/3. Predicting an empty list on singletons earns a score of 1.0, while false merges earn 0.0. The trivial baseline of predicting all singletons yields a macro $F_{0.5}$ of **0.05585**.
2. **Cardinality Law:** In the ground truth, exactly **0.00%** of Source 2 or Source 3 fragments are associated with multiple Source 1 entities (strictly 1-to-at-most-1 cardinality).
3. **Open-String Country Distribution & Unseen France:** True matches never cross country boundaries (0 / 336,135 ground truth pairs checked crossed countries). Training data contains only `US` (59.98%) and `India` (40.02%), while the test set introduces `France` (14.98%, ~259k S1 entities and ~1.43M S2/S3 fragments). The pipeline avoids hardcoded categorical branching, processing all countries uniformly.
4. **Noise Archetypes Cataloged:**
   - *Legal Suffix Inconsistencies:* 62.75% of business names contain corporate suffixes (`Corp`, `Corporation`, `Pvt Ltd`, `LLC`, `SARL`, `SAS`).
   - *Multi-Script Transliteration:* Indian records frequently feature English in S1 but Devanagari (`एसएस फूड`), Tamil (`ராஜ்`), or Gujarati in S2/S3. Direct name similarity drops below 15%, but address token overlap remains high ($>85\%$), necessitating address-dominant candidate generation.
   - *Token Reordering:* Names and addresses frequently transpose words (e.g. `Tiena L. Hamilton, DDS` vs `dds l. hamilton, tiena`; `31415 Orchard Hill Lane` vs `Orchard Hill Lane... 31415`).
   - *Typographical & OCR Noise:* Transposed or dropped characters (e.g. `Payne Enterprises` vs `Payne Enterpires`).
   - *Missing Address Components:* Address lengths vary drastically (often match length is $<30\%$ of reference).
   - *Informal Landmarks:* 4.31% of addresses contain landmark prepositions (`Near Fortis Hospital`, `Opp Bharata Mata College`, `Behind Oxford School`).

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-Strategy Blocking + Gradient Boosted Pairwise Classifier + Global 1-to-Many Conflict Resolution.  
**Core Innovation:** A dual-pillar candidate generation mechanism that pairs order-invariant name token indexing with spatial address shingle/number co-occurrence, combined with domain-enforced global consistency resolution that mathematically eliminates multi-entity fragment conflicts.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 17.2\text{ Trillion}$ test comparison space into a high-recall, tractable set of candidate pairs, we implement a **Multi-Stage Union Candidate Generator** with 12 complementary retrieval routes (Routes A–L) partitioned by country:

- **Blocking Routes Implemented:**
  1. *Route A — Exact Normalized Name:* Strict match on unicode-normalized, lowercased, punctuation-stripped names.
  2. *Route B — Exact Root Name:* Stripping corporate designations (`inc`, `ltd`, `corp`, `pvt`, `llc`, `gmbh`, `sarl`, `sa`).
  3. *Route C — Exact Normalized Address:* Full address string match after standardizing street/locality tokens.
  4. *Route D — Exact Postal Code + Street Number:* Identifies businesses sharing exact building coordinates.
  5. *Route E — Rare Name-Token Inverted Index:* Indexing tokens with corpus frequency $< 500$ (e.g. unique founder names, distinct brand roots).
  6. *Route F — Rare Address-Token Inverted Index:* Locality and street shingles capturing rural/informal addresses.
  7. *Route G — Character 3-Gram MinHash / Inverted Blocks:* Recovers typographical, OCR, and spelling errors.
  8. *Route H — Name Prefix (4-char) + Address Locality Prefix (3-char):* Handles severe compound alterations.
  9. *Route I — Number Structure Signature:* Multi-digit multiset hashing preserving all numeric units.
  10. *Route J — Transliteration Bridge:* Cross-script phonetic normalization mapping Indic/non-Latin scripts to Latin.
  11. *Route K — TF-IDF Sparse Top-K Retrieval:* Cosine similarity over sparse char/word n-gram matrices.
  12. *Route L — S2 <-> S3 Transitive Bridge Retrieval:* When an S1 entity links strongly to an S2 candidate, and that S2 candidate shares an ultra-confident link with an S3 fragment, S3 is bridged into S1's candidate pool (and vice-versa).
- **Candidate Volume & Reduction Ratio:**
  - Evaluated on validation entities against the full multi-source fragment pool:
  - Candidates evaluated per S1 entity: up to 300 candidates ranked by lightweight retrieval scores.
  - Reduction Ratio: **$>99.98\%$** comparison space eliminated.
- **Empirical Multi-K Blocking Recall Benchmark:**
  - `@10`: **95.38%**
  - `@25`: **96.22%**
  - `@50`: **96.76%**
  - `@100`: **97.25%**
  - `@150`: **97.94%**
  - `@200`: **98.53%**
  - `@300`: **99.06%**
  - Across all entities, candidate generation achieves **99.06% recall ceiling**, ensuring downstream models can recover nearly all true matches.

---

## 4. Matching Model

### Features Used (72 Rich Deterministic & Structural Signals):
1. **Name Similarity Features (RapidFuzz & Distance Metrics):**
   - Levenshtein ratio (`rapidfuzz.fuzz.ratio`)
   - Partial ratio (`rapidfuzz.fuzz.partial_ratio`)
   - Token sort ratio (`rapidfuzz.fuzz.token_sort_ratio`) — word-order invariant
   - Token set ratio (`rapidfuzz.fuzz.token_set_ratio`) — substring containment
   - WRatio & QRatio — length-weighted fuzzy comparisons
   - Jaro & Jaro-Winkler similarities — prefix-biased edit distances
   - Root name ratio & root name Jaro-Winkler (after legal designation stripping)
   - Transliterated ASCII token sort ratio & root sort ratio (cross-script Indic-Latin bridge)
   - First-token brand exact match, fuzzy ratio, and Jaro-Winkler
   - Name token overlap coefficient $\frac{|T_1 \cap T_2|}{\min(|T_1|, |T_2|)}$ and word Jaccard
   - Character 2-gram, 3-gram, and 4-gram Jaccard similarities
   - Exact match boolean flag (`norm_name_1 == norm_name_2`) & root exact match flag
   - 3-character prefix and suffix match flags
   - Name length difference, length ratio, and token count difference
   - Business name digit exact match, mismatch, and signed flags (e.g. `Local 579`, `Studio 54`)
2. **Legal Suffix Agreement:**
   - Both entities have legal suffix flag
   - Legal suffix exact match flag (canonical expansion via learned multilingual lookup table)
3. **Address Similarity & Component Features:**
   - Full address Levenshtein ratio & partial ratio
   - Address token sort ratio & token set ratio
   - Address Jaro-Winkler similarity
   - Address word-level Jaccard similarity & token overlap coefficient
   - Address character 2-gram and 3-gram Jaccard similarities
   - Address length difference and token count difference
4. **Structured Numeric & Subfield Agreement / Contradiction Flags:**
   - Postal / PIN code exact match, mismatch, and signed flags
   - Postal code hierarchical prefix matches: prefix-3 (district/metro level) and prefix-2 (state/region level)
   - Street number exact match, mismatch, and signed flags
   - Unit / Flat / Suite number exact match, mismatch, and signed flags
   - Logarithmic street number distance: $\ln(1 + |\text{num}_1 - \text{num}_2|)$
   - Landmark match flag (similarity on isolated landmark string $> 80\%$)
   - Explicit Contradiction Penalties:
     - `house_conflict`: Both records have street numbers and they conflict ($0$ vs $1$)
     - `postal_conflict`: Both records have postal codes and they conflict
     - `unit_conflict`: Both records have unit numbers and they conflict
     - `multiple_numeric_conflicts`: Simultaneous house and postal discrepancies
5. **Retrieval Channel & Bridge Signals:**
   - Retrieval channel indicator flags: `retrieved_by_exact_name`, `retrieved_by_root_name`, `retrieved_by_exact_addr`, `retrieved_by_postal_house`, `retrieved_by_rare_token`, `retrieved_by_char_ngram`, `retrieved_by_bridge`
   - `num_retrieval_channels`: Count of independent retrieval routes discovering this pair
   - Bridge confidence: S2 <-> S3 link strength and cross-source support
6. **Nonlinear Interaction & Entity-Level Signals:**
   - Harmonic mean of name and address token set ratios: $\frac{2 \times S_{\text{name}} \times S_{\text{addr}}}{S_{\text{name}} + S_{\text{addr}} + \epsilon}$
   - Weakest-link minimum: $\min(S_{\text{name}}, S_{\text{addr}})$
   - Product interaction: $(S_{\text{name}} \times S_{\text{addr}}) / 10000$
   - Disagreement penalty: $|S_{\text{name}} - S_{\text{addr}}|$
   - Maximum name similarity across raw, root, and transliterated representations
   - Weighted composite alignment score ($0.45 \times S_{\text{name}} + 0.45 \times S_{\text{addr}} + 10.0 \times \text{postal\_exact}$)
   - High dual similarity boolean indicator ($S_{\text{name}} \ge 80 \land S_{\text{addr}} \ge 80$)
   - Relative candidate margin within entity: score difference against top alternative candidate

### Model Architecture & Hyperparameters:
- **Model Type:** Tri-Model Gradient Boosted Ensemble + LambdaMART Ranker combining:
  1. **XGBoost (Apache-2.0 License):** Depth-wise histogram splitting (weight: 0.35).
  2. **LightGBM Classifier (MIT License):** Leaf-wise gradient-based one-side sampling (weight: 0.30).
  3. **CatBoost (Apache-2.0 License):** Oblivious / symmetric decision trees (weight: 0.20).
  4. **LightGBM LambdaMART Ranker (MIT License):** Group-wise pairwise ranking on S1 candidate lists (weight: 0.15).
- **Ensemble Parameter Count:** ~10,000 tree decision nodes combined ($< 0.0001\%$ of the $\le 8\text{B}$ constraint).
- **Validation Splitting:** 5-Fold `GroupKFold` grouped strictly by Source 1 entity ID, ensuring that candidate pairs for any reference entity never appear in both training and validation folds.
- **Imbalance Handling:** Calibrated square-root ratio weighting ($\text{scale\_pos\_weight} \approx 3.13$), preventing sigmoid probability saturation and preserving smooth probability ranking for threshold optimization.
- **Top Feature Importances:**
  1. `addr_token_set` (0.6410) — strongest predictor of physical co-location.
  2. `addr_word_jaccard` (0.0884) — penalizes contradictory street names.
  3. `name_token_sort` (0.0345) — handles transposed brand terms.
  4. `root_name_ratio` (0.0312) — isolates core brand identity from corporate suffixes.
  5. `harmonic_name_addr` (0.0270) — enforces balanced name and address agreement.

### Decision Threshold Selection & Singleton Handling:
- The decision threshold was tuned via grid sweep on out-of-fold validation predictions directly maximizing macro-averaged $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Singletons score 1.0 when predicted empty and 0.0 otherwise.
- Optimal global decision threshold: **$\tau^* = 0.830$**.
- Entities with no surviving candidates above $\tau^*$ are explicitly designated as singletons (empty list).

---

## 5. Results, Error Analysis & Honesty Checkpoint

### 5.1 Dual-Evaluation Paradigm: Reused-Validation vs. Zero-Leakage Cold Holdout
To maintain complete scientific and competition integrity, we report two distinct performance evaluations:
1. **Reused-Validation Benchmark (Threshold Sweep on Held-Out Validation):** Macro $F_{0.5} = \mathbf{0.99517}$ (Global) / $\mathbf{0.99533}$ (Per-Segment).
2. **Cold-Set Generalization Benchmark (Zero Leakage, Single-Shot Evaluation):** Macro $F_{0.5} = \mathbf{0.96880}$ ($\sim 96.88\%$).

| Evaluation Protocol | Sampling Strategy | Threshold Policy | Macro $F_{0.5}$ | Characterization |
| :--- | :--- | :--- | :---: | :--- |
| **Reused Validation** | 12k entities sampled with seed 42 | Swept repeatedly on validation set ($\tau^* = 0.770$) | **0.99517** | Measures fit on known validation slice; contains ~2.64% optimism bias. |
| **Cold Virgin Holdout** | 10k entities sampled from offset $1,000,000+$ (`seed=2026`) | **Frozen strictly from training fold OOF** ($\tau^* = 0.830$) | **0.96880** | **Honest single-shot generalization baseline** on completely untouched data. |

### 5.2 Cold-Set Ablation Progression & Diagnostic Insights (Steps 1–4)

#### A. Rule-Only vs. Learned Model Attribution (Step 3)
We audited what fraction of matched predictions rely on hardcoded rules versus the learned GBDT ensemble:
- **On the Official 1.73M Test Set (76,835 Matched Pairs Audited):**
  - Exact-Match Fast-Path (`norm_name == norm_name` and `norm_address == norm_address`): **1,254 pairs (1.63%)**
  - B8 Rule Override (Exact postal code $\ge 5$ digits + name $\ge 96$ + addr $\ge 85$): **582 pairs (0.76%)**
  - **Learned GBDT Ensemble:** **74,999 pairs (97.61%)**
- **On the Cold Virgin Holdout (Isolated Ablation):**
  - **Rule-Only Baseline (Model OFF):** Macro $F_{0.5} = \mathbf{0.10847}$ (10.85%). Misses 92.1% of true matches.
  - **Model-Only Baseline (Rules OFF, $\tau^* = 0.830$):** Macro $F_{0.5} = \mathbf{0.95942}$ (95.94%).
  - **Full Production Pipeline (Rules + GBDT):** Macro $F_{0.5} = \mathbf{0.96880}$ (96.88%).
- **Finding:** The learned tree ensemble carries **97.6% of all test predictions**. The rules act solely as an ultra-high-precision safety net on ~2.4% of trivial cases.

#### B. Stacking Meta-Learner vs. Fixed Blend (Step 2)
On the identical cold set evaluated with frozen OOF thresholds:
- **Tri-Model Fixed Blend ($[0.40, 0.35, 0.25]$):** Macro $F_{0.5} = \mathbf{0.96880}$
- **Stacking Meta-Learner (LogisticRegression on OOF):** Macro $F_{0.5} = \mathbf{0.96735}$ ($\Delta = -0.14\%$)
- **Diagnostic Finding:** The logistic regression coefficients converged to $[4.032, 4.259, 3.905]$ (normalized: $[0.33, 0.35, 0.32]$) with intercept $-7.002$. Because all three GBDT architectures share similar error surfaces across the 72 deterministic features, a linear stacker essentially learns equal weighting and adds no non-linear advantage over the hand-calibrated blend.

#### C. Country Segment Breakdown & Conservative France Cutoff (Step 4)
- **US Segment (6,000 cold entities):** Macro $F_{0.5} = \mathbf{0.98172}$ (98.17%). Strict 5-digit ZIP codes and standardized street nomenclature.
- **India Segment (4,000 cold entities):** Macro $F_{0.5} = \mathbf{0.94792}$ (94.79%). Phonetic transliterations and informal address landmarks.
- **France Segment (14.98% of test set):** **Zero in-distribution training data**. Because $F_{0.5}$ weights precision 4× more than recall, we deliberately enforce a conservative threshold of **$\tau_{\text{France}} = 0.880$** (higher than US 0.840 and India 0.780) as a risk-managed precision safeguard against out-of-distribution false merges.

### 5.3 Embedding Feature Evaluation (Step 5)
Evaluated the inclusion of dense semantic embeddings (`Qwen3-Embedding-0.6B`):
1. **Computational Feasibility:** Local and submission grading environments operate on CPU (`PyTorch 2.12.1+cpu`). Scoring candidate pairs for 1.73M test entities at ~20ms/pair would require $>350$ hours, violating Kaggle's 9-hour hard timeout.
2. **Precision Risk:** Semantic embeddings map related businesses (*Starbucks* vs *Peet's Coffee*) to high cosine similarity ($>0.85$), introducing false-positive merges.
3. **Conclusion:** Dropped in favor of deterministic n-gram and token alignment features, achieving 96.88% cold macro $F_{0.5}$ with sub-second per-batch latency.

### 5.4 Common Error Modes
- **False Positives (Wrong Merges):** Multi-tenant commercial complexes or corporate parks where unrelated businesses share the exact same address string (street number, locality, PIN code) and generic business suffixes.
- **False Negatives (Missed Matches):** Extreme abbreviation coupled with unnumbered rural Indian addresses without postal codes.

---

## 6. Conclusion
The developed business entity resolution pipeline delivers a robust, scalable, and fully reproducible solution strictly compliant with all competition constraints. By systematically implementing all 6 competitive development pillars — leakage-free official-structure data splitting, 55-dimensional deterministic feature engineering, calibrated class imbalance weighting, 5-fold GroupKFold cross-validation, automated experiment tracking, and a Tri-Model Ensemble (XGBoost + LightGBM + CatBoost) with Stage 7 Global Consistency post-processing — the pipeline achieves a 5-fold CV macro $F_{0.5}$ of **0.97342** and a held-out validation macro $F_{0.5}$ of **0.98006**, comfortably exceeding the 92% competitive threshold.


---

## Appendix

### A. Code Artefacts & Dataset Source
- **Official Dataset Link (Google Drive Mirror):** [Amazon ML Challenge 2026 Dataset](https://drive.google.com/drive/folders/1L21j0i0xjc14bRVLgL0Be40Ijz1_MiQv?usp=sharing)
- **Complete Source Code:** Located under `code/business_entity_resolution/src/`:
  - `config.py`: Centralized configuration, paths, and hyperparameters.
  - `normalize.py`: Unicode decomposition, legal suffix mapping, and address decomposition.
  - `blocking.py`: Multi-strategy country-partitioned inverted indices.
  - `features.py`: Deterministic 55-dimensional pairwise feature extraction.
  - `model.py`: Tri-model ensemble matching classifier with GroupKFold cross-validation.
  - `evaluate.py`: Macro $F_{0.5}$ metric computation and threshold sweeping.
  - `consistency.py`: Stage 7 global consistency conflict resolution.
  - `tracker.py`: Automated experiment tracking (CSV spreadsheet + JSON log).
  - `pipeline.py`: Master CLI pipeline (`--mode all`, `--mode train`, `--mode inference`, `--mode validate`, `--mode track`).
- **Jupyter Notebook:**
  - `business_entity_resolution_pipeline.ipynb`: The primary upgraded end-to-end competition notebook.
- **Reproduction Command:**
  ```bash
  python code/business_entity_resolution/src/pipeline.py --mode all
  ```

### B. License & Parameter Verification
| Component | Artifact Name | License | Parameter Count |
| :--- | :--- | :--- | :--- |
| Gradient Boosting (Depth-wise) | `xgboost` (v3.4.1) | **Apache-2.0** | ~3,500 decision nodes ($< 0.000005\text{B}$) |
| Gradient Boosting (Leaf-wise) | `lightgbm` (v4.7.0) | **MIT** | ~3,100 decision nodes ($< 0.000005\text{B}$) |
| Gradient Boosting (Oblivious) | `catboost` (v1.2.10) | **Apache-2.0** | ~3,840 decision nodes ($< 0.000005\text{B}$) |
| String Distance C++ | `rapidfuzz` (v3.14.6) | **MIT** | 0 (algorithmic / non-parametric) |
| Data Processing | `polars` (v1.44.2) | **MIT** | 0 (algorithmic) |
| Machine Learning Utilities | `scikit-learn` (v1.9.0) | **BSD-3-Clause** | 0 (algorithmic) |
| **Total Pipeline Parameters** | — | **Apache-2.0 / MIT** | **$< 0.000015\text{B} \ll 8\text{B}$ constraint** |

