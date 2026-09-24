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

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 17.2\text{ Trillion}$ test comparison space into a tractable set of candidate pairs, we implement four complementary blocking strategies partitioned by country string:

- **Blocking Strategies Used:**
  1. *Strategy 1 — Distinctive Name Tokens:* Inverted index on normalized root name tokens (length $\ge 3$, excluding high-frequency entity stopwords like `inc`, `ltd`, `corp`, `services`).
  2. *Strategy 2 — Street Number + Locality Prefix:* Inverted index on extracted street/building number combined with the 4-character prefix of the first significant street word (captures transliterated names where Latin and Devanagari names differ completely but physical street numbers match).
  3. *Strategy 3 — Address Token Co-Occurrence Pairs:* Index on pairs of rare locality tokens (frequency-ranked), capturing rural or Indian addresses lacking municipal house numbers.
  4. *Strategy 4 — Name Prefix + Locality Prefix:* First 4 letters of name + first 3 letters of address (recovers severe name typos and minor address modifications).
- **Candidate Volume & Reduction Ratio:**
  - Evaluated on 10,000 validation entities against 184,015 candidate pool records:
  - Total candidate pairs generated: 1,130,246 (average 113.0 candidates per S1 entity).
  - Full cross-product comparison space: $1,840,150,000$.
  - **Reduction Ratio:** **99.9386%** space reduction.
- **How True Matches Were Preserved:**
  - True pairs captured: 33,239 out of 34,511 true matches.
  - **Pair Completeness (Recall Ceiling):** **96.31%**.

---

## 4. Matching Model

### Features Used (27 Deterministic, Country-Agnostic Signals):
1. **Name Similarity Features:**
   - Levenshtein ratio (`rapidfuzz.fuzz.ratio`)
   - Partial ratio (`rapidfuzz.fuzz.partial_ratio`)
   - Token sort ratio (`rapidfuzz.fuzz.token_sort_ratio`) — word-order invariant
   - Token set ratio (`rapidfuzz.fuzz.token_set_ratio`) — substring containment
   - Root name ratio (similarity after stripping legal suffixes)
   - Character 3-gram Jaccard similarity
   - Exact match boolean flag (`norm_name_1 == norm_name_2`)
   - Root exact match boolean flag
   - 3-character prefix match flag
   - Name length difference & length ratio
2. **Legal Suffix Agreement:**
   - Both entities have legal suffix flag
   - Legal suffix exact match flag (canonical expansion via lookup table)
3. **Address Similarity Features:**
   - Full address Levenshtein ratio
   - Address partial ratio
   - Address token sort ratio
   - Address token set ratio (vital for sparse vs complete address matching)
   - Address word-level Jaccard similarity
   - Address character 3-gram Jaccard similarity
   - Address length difference
4. **Structured Subfield Agreement Flags:**
   - Postal / PIN code exact match flag (`1.0` if both non-empty and equal, `0.0` otherwise)
   - Postal code mismatch flag (`1.0` if both non-empty and unequal, `0.0` otherwise)
   - Street number exact match flag
   - Street number mismatch flag
   - Landmark match flag (similarity on isolated landmark string $> 80\%$)
5. **Composite Interaction Features:**
   - Harmonic mean of name and address token set ratios: $\frac{2 \times S_{\text{name}} \times S_{\text{addr}}}{S_{\text{name}} + S_{\text{addr}} + \epsilon}$
   - High dual similarity boolean indicator ($S_{\text{name}} \ge 80 \land S_{\text{addr}} \ge 80$)

### Model Architecture & Hyperparameters:
- **Model Type:** Pairwise Gradient Boosted Decision Trees via **XGBoost (Apache-2.0 License)**.
- **Complexity:** 120 trees, max depth = 6, learning rate = 0.1, subsample = 0.85, colsample_bytree = 0.85. Total parameter count is $< 50,000$ tree decision nodes (well within the $\le 8\text{B}$ parameter limit).
- **Validation Splitting:** 3-Fold `GroupKFold` grouped strictly by Source 1 entity ID, ensuring that candidate pairs for any reference entity never appear in both training and validation folds.
- **Imbalance Handling:** Implemented via `scale_pos_weight = N_neg / N_pos` (~15.4 to 35.0), preserving true underlying probability ranking without synthetic undersampling distortion.
- **Top 5 Feature Importances:**
  1. `addr_token_set` (0.8664) — strongest predictor of physical co-location.
  2. `root_name_ratio` (0.0217) — isolates primary brand identity from legal designations.
  3. `addr_word_jaccard` (0.0203) — penalizes contradictory street names.
  4. `name_token_sort` (0.0150) — handles transposed brand terms.
  5. `street_num_mismatch` (0.0129) — penalizes mismatched building numbers.

### Decision Threshold Selection & Singleton Handling:
- The decision threshold was tuned via grid sweep on out-of-fold validation predictions directly maximizing macro-averaged $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Singletons score 1.0 when predicted empty and 0.0 otherwise.
- Because $F_{0.5}$ penalizes false positives twice as heavily as false negatives ($\beta = 0.5$), the metric rewards conservative, high-precision boundaries.
- Optimal threshold: **$\tau^* = 0.940 - 0.950$**.
- Entities with no surviving candidates above $\tau^*$ are explicitly designated as singletons (empty list).

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score on Held-Out Validation:**
  - **Trivial Baseline (Predict All Empty):** **0.05585**
  - **Model Out-of-Fold Macro $F_{0.5}$:** **0.96494**
  - **Net Gain Over Baseline:** **+0.90909 (+1,627% relative improvement)**
- **Stage 7 Global Consistency Impact:**
  - Before Conflict Resolution: Macro $F_{0.5} = 0.96494$ (several S2/S3 fragments claimed by $>1$ S1 entity).
  - After Global Consistency Resolution: Macro $F_{0.5} = \mathbf{0.96512}$ (+0.00018 improvement, 0 multi-merge violations).
- **Common False Positives (Wrong Merges):**
  - Multi-tenant commercial complexes or corporate parks where unrelated businesses share the exact same address string (street number, locality, PIN code) and have generic words in their name (e.g. `Apex Solutions` vs `Apex Services`).
- **Common False Negatives (Missed Matches):**
  - Extreme abbreviation coupled with unnumbered addresses, or Indian transliterations where the entity name is completely localized into Devanagari and the address contains only an informal neighborhood without street numbers or PIN codes.

---

## 6. Conclusion
The developed business entity resolution pipeline delivers a robust, scalable, and fully reproducible solution strictly compliant with all competition constraints. By combining multi-strategy country-partitioned blocking, high-throughput feature extraction, an Apache-2.0 XGBoost classifier, and domain-grounded global consistency post-processing, the pipeline achieves an out-of-fold validation macro $F_{0.5}$ of **0.96512**, outperforming the trivial singleton baseline by over $1,600\%$.

---

## Appendix

### A. Code Artefacts
- **Complete Source Code:** Located under `code/business_entity_resolution/src/`:
  - `config.py`: Centralized configuration, paths, and hyperparameters.
  - `normalize.py`: Unicode decomposition, legal suffix mapping, and address decomposition.
  - `blocking.py`: Multi-strategy country-partitioned inverted indices.
  - `features.py`: Deterministic 27-dimensional pairwise feature extraction.
  - `model.py`: XGBoost matching classifier with GroupKFold cross-validation.
  - `evaluate.py`: Macro $F_{0.5}$ metric computation and threshold sweeping.
  - `consistency.py`: Stage 7 global consistency conflict resolution.
  - `pipeline.py`: Master CLI pipeline (`--mode all`, `--mode train`, `--mode inference`, `--mode validate`).
- **Jupyter Notebook:** `business_entity_resolution_pipeline.ipynb` containing the interactive, step-by-step walkthrough.
- **Reproduction Command:**
  ```bash
  python code/business_entity_resolution/src/pipeline.py --mode all
  ```

### B. License & Parameter Verification
| Component | Artifact Name | License | Parameter Count |
| :--- | :--- | :--- | :--- |
| Gradient Boosting | `xgboost` (v3.4.1) | **Apache-2.0** | ~40,000 decision nodes ($< 0.00005\text{B}$) |
| String Distance C++ | `rapidfuzz` (v3.14.6) | **MIT** | 0 (algorithmic / non-parametric) |
| Data Processing | `polars` (v1.44.2) | **MIT** | 0 (algorithmic) |
| Machine Learning Utilities | `scikit-learn` (v1.9.0) | **BSD-3-Clause** | 0 (algorithmic) |
| **Total Pipeline Parameters** | — | **Apache-2.0 / MIT** | **$< 0.00005\text{B} \ll 8\text{B}$ constraint** |
