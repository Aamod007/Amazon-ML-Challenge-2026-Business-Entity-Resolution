"""
Main End-to-End Pipeline Orchestrator
Amazon ML Challenge 2026 - Business Entity Resolution

Orchestrates:
  1. Data Ingestion & Grouped Validation Splitting
  2. Normalization & Coverage Audit
  3. Multi-Strategy Candidate Generation (Blocking) & Metric Audit
  4. Pairwise Feature Engineering
  5. XGBoost Model Training (GroupKFold, Class Imbalance Weighting)
  6. Decision Threshold Optimization for Macro F_0.5 & Singleton Handling
  7. Global Consistency Conflict Resolution (Stage 7)
  8. Output Generation (matching_results.tsv & candidate_pairs.tsv)
  9. Submission Validation (validate_submission.py)
"""

import os
import sys
import argparse
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd

# Ensure src directory is in sys.path for direct script execution
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from .config import CONFIG, PipelineConfig
    from .normalize import normalize_name, normalize_address, normalize_record
    from .blocking import CountryCandidateIndex, extract_name_tokens, extract_addr_tokens
    from .features import compute_pair_features, FEATURE_NAMES
    from .model import MatchingClassifier
    from .evaluate import evaluate_macro_f05, sweep_optimal_threshold
    from .consistency import resolve_global_consistency
    from .data import load_ground_truth, stream_tsv_records, load_records_by_country
except ImportError:
    from config import CONFIG, PipelineConfig
    from normalize import normalize_name, normalize_address, normalize_record
    from blocking import CountryCandidateIndex, extract_name_tokens, extract_addr_tokens
    from features import compute_pair_features, FEATURE_NAMES
    from model import MatchingClassifier
    from evaluate import evaluate_macro_f05, sweep_optimal_threshold
    from consistency import resolve_global_consistency
    from data import load_ground_truth, stream_tsv_records, load_records_by_country

def train_pipeline(
    train_dir: str,
    model_save_path: str,
    n_sample_s1: int = 15000,
    config: PipelineConfig = CONFIG
) -> Tuple[MatchingClassifier, float]:
    """
    Train pairwise matching model on training split and tune decision threshold.
    Preserves true singleton ratio and avoids validation leakage.
    """
    print("\n" + "="*70)
    print("STAGE 5: TRAINING PAIRWISE MATCHING MODEL")
    print("="*70)
    
    gt_file = os.path.join(train_dir, config.train_gt_file)
    s1_file = os.path.join(train_dir, config.train_s1_file)
    s2_file = os.path.join(train_dir, config.train_s2_file)
    s3_file = os.path.join(train_dir, config.train_s3_file)
    
    # 1. Load ground truth
    true_matches_all, all_s1_ids = load_ground_truth(gt_file)
    
    # Stratified sample by singleton status
    singletons = [s1 for s1 in all_s1_ids if len(true_matches_all[s1]) == 0]
    non_singletons = [s1 for s1 in all_s1_ids if len(true_matches_all[s1]) > 0]
    
    singleton_ratio = len(singletons) / len(all_s1_ids)
    n_singletons_sample = int(n_sample_s1 * singleton_ratio)
    n_non_singletons_sample = n_sample_s1 - n_singletons_sample
    
    rng = np.random.RandomState(config.random_state)
    sampled_s1 = set(rng.choice(singletons, size=n_singletons_sample, replace=False)).union(
        set(rng.choice(non_singletons, size=n_non_singletons_sample, replace=False))
    )
    
    print(f"Sampled {len(sampled_s1):,} S1 entities for model training (preserving {singleton_ratio*100:.2f}% singleton ratio).")
    
    # Target true match IDs
    target_match_ids = set()
    for s1 in sampled_s1:
        target_match_ids.update(true_matches_all[s1])
    print(f"True target S2/S3 entities to capture: {len(target_match_ids):,}")

    # 2. Load and normalize S1 records
    print("Loading and normalizing Source 1 records...")
    import polars as pl
    s1_records = {}
    df_s1 = pl.read_csv(s1_file, separator="\t")
    for row in df_s1.filter(pl.col("entity_id").is_in(list(sampled_s1))).iter_rows():
        s1_records[row[0]] = normalize_record(row)

    # 3. Load candidate pool (true matches + background records for realistic negative training)
    print("Loading candidate pool from Source 2 and Source 3...")
    pool_records = {}
    max_bg = 40000
    for src_file in [s2_file, s3_file]:
        df_src = pl.read_csv(src_file, separator="\t")
        targets_df = df_src.filter(pl.col("entity_id").is_in(list(target_match_ids)))
        bg_df = df_src.head(max_bg)
        comb_df = pl.concat([targets_df, bg_df]).unique(subset=["entity_id"])
        for row in comb_df.iter_rows():
            pool_records[row[0]] = normalize_record(row)

    print(f"Candidate pool loaded: {len(pool_records):,} records.")

    # 4. Group by country and build blocking indices
    print("Building country blocking indices...")
    countries = set(rec["country"] for rec in s1_records.values())
    country_indices = {}
    for c in countries:
        c_pool = {eid: rec for eid, rec in pool_records.items() if rec["country"] == c}
        c_idx = CountryCandidateIndex(c)
        c_idx.build(c_pool)
        country_indices[c] = c_idx

    # 5. Generate candidate pairs and extract features
    print("Generating candidate pairs and computing pairwise features...")
    pairs_X = []
    labels_y = []
    groups = []
    pair_meta = []
    
    for s1_id, rec1 in s1_records.items():
        c = rec1["country"]
        c_idx = country_indices.get(c)
        if c_idx is None:
            continue
        cands = c_idx.query(rec1, max_candidates=config.max_candidates_per_entity)
        true_for_s1 = true_matches_all[s1_id]
        
        for mid in cands:
            rec2 = pool_records.get(mid)
            if rec2 is None:
                continue
            feats = compute_pair_features(rec1, rec2)
            lbl = 1 if mid in true_for_s1 else 0
            
            pairs_X.append(feats)
            labels_y.append(lbl)
            groups.append(s1_id)
            pair_meta.append((s1_id, mid))

    X = np.array(pairs_X, dtype=np.float32)
    y = np.array(labels_y, dtype=np.int32)
    
    # 6. Train model with GroupKFold
    classifier = MatchingClassifier(config=config)
    train_res = classifier.train(X, y, groups=groups, n_splits=3)
    
    # 7. Tune Decision Threshold for Macro F_0.5
    print("\nSTAGE 6: TUNING DECISION THRESHOLD ON VALIDATION PREDICTIONS")
    oof_probs = train_res["oof_probs"]
    s1_cand_probs = defaultdict(list)
    for i, (s1_id, mid) in enumerate(pair_meta):
        s1_cand_probs[s1_id].append((mid, float(oof_probs[i])))

    best_thresh, best_score = sweep_optimal_threshold(
        s1_cand_probs, true_matches_all, list(sampled_s1)
    )
    print(f"Optimal Decision Threshold: {best_thresh:.3f} | Out-of-Fold Macro F_0.5: {best_score:.5f}")
    
    # Baseline comparison
    empty_preds = {s1: set() for s1 in sampled_s1}
    baseline_score = evaluate_macro_f05(true_matches_all, empty_preds, list(sampled_s1))
    print(f"Trivial 'Predict All Singletons' Baseline: {baseline_score:.5f}")
    print(f"Validation Gain Above Baseline: +{best_score - baseline_score:.5f} (+{(best_score - baseline_score)/baseline_score * 100:.1f}%)")

    # Save model artifact
    classifier.save(model_save_path)
    print(f"Saved trained model artifact to: {model_save_path}")
    
    return classifier, best_thresh

def run_test_inference(
    test_dir: str,
    output_dir: str,
    model: MatchingClassifier,
    threshold: float,
    config: PipelineConfig = CONFIG
):
    """
    Run candidate generation, feature engineering, and inference on the test set.
    Generates:
      - output/matching_results.tsv
      - output/candidate_pairs.tsv
    Guarantees:
      - Every S1 entity in test_source1.tsv appears in exact order
      - Empty matched_entity_ids for predicted singletons
      - Every matched ID is a strict subset of that entity's candidates
      - Zero self-matches, zero duplicate IDs
      - 100% open string country support (US, India, France, etc.)
    """
    print("\n" + "="*70)
    print("STAGE 8: GENERATING SUBMISSION OUTPUT FILES ON TEST SET")
    print("="*70)
    
    import gc
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, config.output_matching_file)
    candidate_out_path = os.path.join(output_dir, config.output_candidate_file)
    
    s1_test_file = os.path.join(test_dir, config.test_s1_file)
    s2_test_file = os.path.join(test_dir, config.test_s2_file)
    s3_test_file = os.path.join(test_dir, config.test_s3_file)
    
    # 1. Discover all countries dynamically from test_source1.tsv
    print("Scanning test set countries...")
    test_countries = set()
    s1_test_order = []
    s1_country_map = {}
    
    with open(s1_test_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_col = header.index("entity_id")
        c_col = header.index("country")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid = parts[id_col].strip()
                c = parts[c_col].strip()
                s1_test_order.append(eid)
                s1_country_map[eid] = c
                test_countries.add(c)

    print(f"Total test Source 1 entities: {len(s1_test_order):,}")
    print(f"Discovered test countries (open string): {test_countries}")
    
    # Pre-initialize result stores
    final_candidates = {s1: [] for s1 in s1_test_order}
    final_matches = {s1: [] for s1 in s1_test_order}
    
    # Process country by country to maintain low memory footprint (< 2 GB RAM)
    for c_idx, country in enumerate(sorted(test_countries), 1):
        print(f"\n--- Processing Country [{c_idx}/{len(test_countries)}]: {country} ---")
        t0 = time.time()
        
        # Load S1 records for this country
        print(f"Loading Source 1 records for {country}...")
        s1_country_recs = load_records_by_country(s1_test_file, target_country=country)
        print(f"Loaded {len(s1_country_recs):,} S1 records for {country}.")

        # Load S2 and S3 records for this country
        print(f"Loading Source 2 and Source 3 candidate pool for {country}...")
        pool_country_recs = load_records_by_country(s2_test_file, target_country=country)
        pool_country_recs.update(load_records_by_country(s3_test_file, target_country=country))
        print(f"Loaded {len(pool_country_recs):,} target records for {country}.")

        # Build candidate blocking index
        print(f"Building blocking index for {country}...")
        c_index = CountryCandidateIndex(country)
        c_index.build(pool_country_recs)

        # Generate candidates & score with classifier in batches
        print(f"Generating candidates and scoring for {country}...")
        s1_cand_probs = defaultdict(list)
        
        # Batch scoring
        batch_pairs = []
        batch_meta = []
        
        s1_items = list(s1_country_recs.items())
        total_s1 = len(s1_items)
        
        for idx, (s1_id, rec1) in enumerate(s1_items):
            cands = c_index.query(rec1, max_candidates=config.max_candidates_per_entity)
            cand_list = sorted(list(cands))
            final_candidates[s1_id] = cand_list
            
            for mid in cand_list:
                rec2 = pool_country_recs.get(mid)
                if rec2 is not None:
                    # Fast-path: exact matching on name and address directly assigns prob 1.0
                    if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                        s1_cand_probs[s1_id].append((mid, 1.0))
                    else:
                        feats = compute_pair_features(rec1, rec2)
                        batch_pairs.append(feats)
                        batch_meta.append((s1_id, mid))
                    
            if len(batch_pairs) >= config.batch_size or idx == total_s1 - 1:
                if batch_pairs:
                    X_batch = np.array(batch_pairs, dtype=np.float32)
                    probs = model.predict_proba(X_batch)
                    for (s1_ref, mid_ref), prob in zip(batch_meta, probs):
                        s1_cand_probs[s1_ref].append((mid_ref, float(prob)))
                    batch_pairs = []
                    batch_meta = []
                    
            if (idx + 1) % 25000 == 0 or idx == total_s1 - 1:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 0.1)
                rem_s = (total_s1 - (idx + 1)) / max(rate, 1)
                print(f"  Processed {idx + 1:,} / {total_s1:,} S1 entities ({rate:.1f} ent/s, ETA: {rem_s/60:.1f}m)...")

        # Apply Global Consistency (Stage 7) if enabled
        if config.use_global_consistency:
            print(f"Applying Stage 7 Global Consistency conflict resolution for {country}...")
            resolved_country_matches, n_conflicts = resolve_global_consistency(
                s1_cand_probs, threshold=threshold
            )
            print(f"Resolved {n_conflicts:,} candidate assignment conflicts in {country}.")
            for s1_id in s1_country_recs:
                # Retain sorted order matching candidates
                allowed_cands = set(final_candidates[s1_id])
                surviving = [m for m in final_candidates[s1_id] if m in resolved_country_matches.get(s1_id, set())]
                final_matches[s1_id] = surviving
        else:
            for s1_id in s1_country_recs:
                surviving = [mid for mid, prob in s1_cand_probs.get(s1_id, []) if prob >= threshold]
                final_matches[s1_id] = surviving

        print(f"Completed {country} processing in {time.time() - t0:.2f}s.")
        
        # Free country memory immediately
        del s1_country_recs
        del pool_country_recs
        del c_index
        del s1_cand_probs
        gc.collect()

    # 2. Write output files strictly tab-separated
    print(f"\nWriting {candidate_out_path}...")
    with open(candidate_out_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_test_order:
            cands = final_candidates.get(s1_id, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    print(f"Writing {matching_out_path}...")
    n_singletons = 0
    with open(matching_out_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_test_order:
            matches = final_matches.get(s1_id, [])
            if not matches:
                n_singletons += 1
            match_str = ",".join(matches)
            f_match.write(f"{s1_id}\t{match_str}\n")

    print(f"\nSuccessfully generated outputs:")
    print(f"  matching_results.tsv : {len(s1_test_order):,} rows ({n_singletons:,} singletons)")
    print(f"  candidate_pairs.tsv  : {len(s1_test_order):,} rows")

def validate_submission_files(matching_file: str, candidate_file: str, test_dir: str):
    """Run utils/validate_submission.py to ensure zero errors."""
    print("\n" + "="*70)
    print("RUNNING SUBMISSION VALIDATION")
    print("="*70)
    import subprocess
    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    return res.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Directory with train TSVs")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Directory with test TSVs")
    parser.add_argument("--output-dir", default="output", help="Directory for final output TSVs")
    parser.add_argument("--model-path", default="models/xgb_matching_model.json", help="Path to save/load model")
    parser.add_argument("--mode", choices=["all", "train", "inference", "validate"], default="all", help="Pipeline execution mode")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--n-train", type=int, default=15000, help="Number of S1 entities to train on")
    
    args = parser.parse_args()
    config = CONFIG
    
    classifier = MatchingClassifier(config=config)
    threshold = args.threshold or config.decision_threshold

    if args.mode in ["all", "train"]:
        classifier, tuned_thresh = train_pipeline(
            train_dir=args.train_dir,
            model_save_path=args.model_path,
            n_sample_s1=args.n_train,
            config=config
        )
        if args.threshold is None:
            threshold = tuned_thresh

    if args.mode in ["all", "inference"]:
        if not os.path.exists(args.model_path):
            print(f"Model file not found at {args.model_path}. Training first...")
            classifier, tuned_thresh = train_pipeline(
                train_dir=args.train_dir,
                model_save_path=args.model_path,
                n_sample_s1=args.n_train,
                config=config
            )
            threshold = tuned_thresh
        else:
            classifier.load(args.model_path)
            print(f"Loaded trained model from {args.model_path}")
            
        run_test_inference(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            model=classifier,
            threshold=threshold,
            config=config
        )
        # Mirror outputs to student_resource/output
        sr_output = os.path.join("student_resource", args.output_dir)
        if os.path.exists(sr_output) and sr_output != args.output_dir:
            import shutil
            for fname in [config.output_matching_file, config.output_candidate_file]:
                src_f = os.path.join(args.output_dir, fname)
                dst_f = os.path.join(sr_output, fname)
                if os.path.exists(src_f):
                    shutil.copy2(src_f, dst_f)

    if args.mode in ["all", "validate"]:
        matching_f = os.path.join(args.output_dir, config.output_matching_file)
        candidate_f = os.path.join(args.output_dir, config.output_candidate_file)
        valid = validate_submission_files(matching_f, candidate_f, args.test_dir)
        if not valid:
            print("VALIDATION FAILED!")
            sys.exit(1)
        else:
            print("VALIDATION PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
