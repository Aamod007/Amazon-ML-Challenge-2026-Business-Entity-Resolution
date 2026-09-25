"""
Main End-to-End Pipeline Orchestrator — Amazon ML Challenge 2026
Business Entity Resolution Solution (Rank-1 Max-Out Architecture)

Fully implements all 6 competitive development pillars:
  1. Proper Data Splitting: Stratified singleton & country train/val split respecting official test structure
  2. Feature Engineering First: 55-dimensional deterministic pairwise alignment feature vector
  3. Class Imbalance Handling: Calibrated square-root ratio scale weighting
  4. 5-Fold Cross-Validation: GroupKFold grouped strictly by Source 1 entity to eliminate leakage
  5. Experiment Tracking: Automated CSV spreadsheet (experiments/experiment_tracker.csv) and JSON logging
  6. Multi-Model Ensembling: Tri-model blend of XGBoost + LightGBM + CatBoost with Global Consistency
"""

import os
import sys
import argparse
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional
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
    from .evaluate import evaluate_macro_f05, sweep_optimal_threshold, compute_entity_f05
    from .consistency import resolve_global_consistency
    from .data import load_ground_truth, stream_tsv_records, load_records_by_country, create_official_split
    from .tracker import ExperimentTracker
except ImportError:
    from config import CONFIG, PipelineConfig
    from normalize import normalize_name, normalize_address, normalize_record
    from blocking import CountryCandidateIndex, extract_name_tokens, extract_addr_tokens
    from features import compute_pair_features, FEATURE_NAMES
    from model import MatchingClassifier
    from evaluate import evaluate_macro_f05, sweep_optimal_threshold, compute_entity_f05
    from consistency import resolve_global_consistency
    from data import load_ground_truth, stream_tsv_records, load_records_by_country, create_official_split
    from tracker import ExperimentTracker

def train_pipeline(
    train_dir: str,
    model_save_path: str,
    n_sample_s1: int = 20000,
    n_val_s1: int = 4000,
    architecture: str = "ensemble",
    n_folds: int = 5,
    imbalance_strategy: str = "sqrt_ratio",
    config: PipelineConfig = CONFIG
) -> Tuple[MatchingClassifier, float]:
    """
    Train pairwise matching model using 5-Fold GroupKFold CV, multi-model ensembling,
    feature engineering, and hold-out validation respecting the official test structure.
    """
    t_start = time.time()
    tracker = ExperimentTracker(log_dir=config.experiments_dir)

    # -------------------------------------------------------------------------
    # PILLAR 1: Proper Data Split Respecting Official Test Structure
    # -------------------------------------------------------------------------
    train_data, val_data = create_official_split(
        train_dir=train_dir,
        n_train_s1=n_sample_s1,
        n_val_s1=n_val_s1,
        max_bg_records=40000,
        random_state=config.random_state
    )

    s1_train_records = train_data["s1_records"]
    pool_train_records = train_data["pool_records"]
    true_matches_train = train_data["true_matches"]
    train_s1_ids = train_data["s1_ids"]

    # -------------------------------------------------------------------------
    # STAGE 3: Multi-Strategy Blocking for Training Candidates
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 3: MULTI-STRATEGY CANDIDATE GENERATION (TRAIN SET)")
    print("="*70)
    countries = set(rec["country"] for rec in s1_train_records.values())
    country_indices = {}
    for c in countries:
        c_pool = {eid: rec for eid, rec in pool_train_records.items() if rec["country"] == c}
        c_idx = CountryCandidateIndex(c)
        c_idx.build(c_pool)
        country_indices[c] = c_idx

    # -------------------------------------------------------------------------
    # PILLAR 2: Feature Engineering (55 Deterministic Signals)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"STAGE 4: EXTRACTING {len(FEATURE_NAMES)}-DIMENSIONAL FEATURE VECTORS")
    print("="*70)
    pairs_X = []
    labels_y = []
    groups = []
    pair_meta = []
    
    t_feat0 = time.time()
    for s1_id in train_s1_ids:
        rec1 = s1_train_records.get(s1_id)
        if rec1 is None:
            continue
        c = rec1["country"]
        c_idx = country_indices.get(c)
        if c_idx is None:
            continue
        cands = c_idx.query(rec1, max_candidates=config.max_candidates_per_entity)
        true_for_s1 = true_matches_train[s1_id]
        
        for mid in cands:
            rec2 = pool_train_records.get(mid)
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
    print(f"Extracted {len(pairs_X):,} candidate pairs in {time.time() - t_feat0:.2f}s ({len(pairs_X)/max(time.time() - t_feat0, 0.01):.0f} pairs/sec).")

    # -------------------------------------------------------------------------
    # PILLARS 3, 4, 6: Imbalance Handling, 5-Fold GroupKFold CV & Ensembling
    # -------------------------------------------------------------------------
    classifier = MatchingClassifier(architecture=architecture, config=config)
    cv_results = classifier.train_cv(
        X=X,
        y=y,
        groups=groups,
        pair_meta=pair_meta,
        true_matches_dict=true_matches_train,
        all_s1_ids=train_s1_ids,
        n_splits=n_folds,
        imbalance_strategy=imbalance_strategy
    )

    optimal_threshold = cv_results["optimal_threshold"]
    oof_score = cv_results["oof_macro_f05"]

    # -------------------------------------------------------------------------
    # EVALUATION ON HELD-OUT OFFICIAL STRUCTURE VALIDATION SET
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 6: EVALUATING FULL END-TO-END PIPELINE ON HELD-OUT VALIDATION SET")
    print("="*70)
    s1_val_records = val_data["s1_records"]
    pool_val_records = val_data["pool_records"]
    true_matches_val = val_data["true_matches"]
    val_s1_ids = val_data["s1_ids"]

    # Build validation blocking indices
    val_countries = set(rec["country"] for rec in s1_val_records.values())
    val_country_indices = {}
    for c in val_countries:
        c_pool = {eid: rec for eid, rec in pool_val_records.items() if rec["country"] == c}
        c_idx = CountryCandidateIndex(c)
        c_idx.build(c_pool)
        val_country_indices[c] = c_idx

    val_pairs_X = []
    val_pair_meta = []
    val_candidates = {}
    total_val_true_pairs = sum(len(true_matches_val[s1]) for s1 in val_s1_ids)
    captured_val_true_pairs = 0

    for s1_id in val_s1_ids:
        rec1 = s1_val_records.get(s1_id)
        if rec1 is None:
            continue
        c = rec1["country"]
        c_idx = val_country_indices.get(c)
        if c_idx is None:
            val_candidates[s1_id] = []
            continue
        cands = c_idx.query(rec1, max_candidates=config.max_candidates_per_entity)
        cand_list = sorted(list(cands))
        val_candidates[s1_id] = cand_list
        true_for_s1 = true_matches_val[s1_id]

        for mid in cand_list:
            if mid in true_for_s1:
                captured_val_true_pairs += 1
            rec2 = pool_val_records.get(mid)
            if rec2 is not None:
                feats = compute_pair_features(rec1, rec2)
                val_pairs_X.append(feats)
                val_pair_meta.append((s1_id, mid))

    val_recall = captured_val_true_pairs / max(total_val_true_pairs, 1)
    print(f"Validation Candidate Blocking Recall: {val_recall*100:.2f}% ({captured_val_true_pairs:,} / {total_val_true_pairs:,} true pairs captured).")

    # Score validation pairs with model / ensemble
    val_s1_cand_probs = defaultdict(list)
    if val_pairs_X:
        X_val = np.array(val_pairs_X, dtype=np.float32)
        val_probs = classifier.predict_proba(X_val)
        for (s1_id, mid), prob in zip(val_pair_meta, val_probs):
            val_s1_cand_probs[s1_id].append((mid, float(prob)))

    # Raw predictions before Global Consistency
    raw_val_preds = {}
    for s1_id in val_s1_ids:
        cands = val_s1_cand_probs.get(s1_id, [])
        raw_val_preds[s1_id] = set(mid for mid, prob in cands if prob >= optimal_threshold)

    score_before_gc = evaluate_macro_f05(true_matches_val, raw_val_preds, val_s1_ids)

    # Predictions after Stage 7 Global Consistency Resolution
    resolved_val_matches, n_val_conflicts = resolve_global_consistency(
        val_s1_cand_probs, threshold=optimal_threshold
    )
    score_after_gc = evaluate_macro_f05(true_matches_val, resolved_val_matches, val_s1_ids)

    # Evaluate singleton vs non-singleton breakdown
    val_singletons = [s for s in val_s1_ids if len(true_matches_val[s]) == 0]
    val_non_singletons = [s for s in val_s1_ids if len(true_matches_val[s]) > 0]
    singleton_score = evaluate_macro_f05(true_matches_val, resolved_val_matches, val_singletons)
    non_singleton_score = evaluate_macro_f05(true_matches_val, resolved_val_matches, val_non_singletons)

    print(f"\nHeld-Out Validation Results (Official Structure):")
    print(f"  Macro F0.5 (Before Global Consistency): {score_before_gc:.5f}")
    print(f"  Macro F0.5 (After Global Consistency) : {score_after_gc:.5f} (Lift: +{score_after_gc - score_before_gc:+.5f})")
    print(f"  Singleton Accuracy ({len(val_singletons):,} singletons)     : {singleton_score:.5f}")
    print(f"  Non-Singleton F0.5 ({len(val_non_singletons):,} entities)     : {non_singleton_score:.5f}")
    print(f"  Resolved 1-to-many candidate conflicts: {n_val_conflicts:,}")

    # Trivial baseline comparison
    empty_preds = {s1: set() for s1 in val_s1_ids}
    baseline_score = evaluate_macro_f05(true_matches_val, empty_preds, val_s1_ids)
    print(f"  Trivial Baseline ('Predict All Empty'): {baseline_score:.5f}")
    print(f"  Final Lift Above Baseline             : +{score_after_gc - baseline_score:.5f} (+{(score_after_gc - baseline_score)/baseline_score * 100:.1f}%)")

    # -------------------------------------------------------------------------
    # PILLAR 5: Track Every Experiment (Spreadsheet & JSON Log)
    # -------------------------------------------------------------------------
    duration = time.time() - t_start
    exp_id = tracker.log(
        model_architecture=architecture,
        n_train_s1=len(train_s1_ids),
        n_folds=n_folds,
        n_features=len(FEATURE_NAMES),
        cv_mean_macro_f05=cv_results["cv_mean"],
        cv_std_macro_f05=cv_results["cv_std"],
        optimal_threshold=optimal_threshold,
        fold_scores=cv_results["fold_scores"],
        singleton_f05=singleton_score,
        non_singleton_f05=non_singleton_score,
        imbalance_strategy=imbalance_strategy,
        scale_pos_weight=cv_results["scale_pos_weight"],
        ensemble_weights=classifier.ensemble_weights if architecture == "ensemble" else None,
        duration_seconds=duration,
        notes=f"Held-out Val F0.5: {score_after_gc:.5f} (Val S1={len(val_s1_ids):,}, Recall={val_recall*100:.1f}%)"
    )

    tracker.print_summary()

    # Save model artifact bundle
    classifier.save(model_save_path)
    print(f"Saved trained {architecture} model artifact to: {model_save_path}")

    return classifier, optimal_threshold

def run_test_inference(
    test_dir: str,
    output_dir: str,
    model: MatchingClassifier,
    threshold: float,
    config: PipelineConfig = CONFIG
):
    """
    Run memory-safe candidate generation, feature extraction, and model inference on the official test set.
    Generates:
      - output/matching_results.tsv
      - output/candidate_pairs.tsv
    Guarantees:
      - 100% test entities preserved in exact original order
      - Empty matched_entity_ids for predicted singletons
      - Every matched ID is a strict subset of that entity's candidates
      - Zero self-matches, zero duplicate IDs
      - 100% open string country support (US, India, France, etc.)
    """
    print("\n" + "="*70)
    print("STAGE 8: GENERATING OFFICIAL SUBMISSION OUTPUT FILES ON TEST SET")
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
                test_countries.add(c)

    print(f"Total test Source 1 entities: {len(s1_test_order):,}")
    print(f"Discovered test countries (open string): {test_countries}")
    
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
                    # Fast-path: exact matching on normalized name and address directly assigns prob 1.0
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

    # Write output files strictly tab-separated
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
    print(f"  matching_results.tsv : {len(s1_test_order):,} rows ({n_singletons:,} singletons / {len(s1_test_order)-n_singletons:,} non-singletons)")
    print(f"  candidate_pairs.tsv  : {len(s1_test_order):,} rows")

def validate_submission_files(matching_file: str, candidate_file: str, test_dir: str) -> bool:
    """Run utils/validate_submission.py to ensure zero errors."""
    print("\n" + "="*70)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
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
    parser.add_argument("--model-path", default="models/ensemble_matching_model", help="Path to save/load model")
    parser.add_argument("--model", choices=["ensemble", "xgboost", "lightgbm", "catboost"], default="ensemble", help="Model architecture")
    parser.add_argument("--mode", choices=["all", "train", "inference", "validate", "track"], default="all", help="Pipeline execution mode")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--n-train", type=int, default=20000, help="Number of S1 entities to train on")
    parser.add_argument("--n-val", type=int, default=4000, help="Number of S1 entities for validation")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--imbalance", choices=["sqrt_ratio", "full_ratio", "none"], default="sqrt_ratio", help="Imbalance weighting strategy")
    
    args = parser.parse_args()
    config = CONFIG
    
    if args.mode == "track":
        tracker = ExperimentTracker(log_dir=config.experiments_dir)
        tracker.print_summary()
        return

    classifier = MatchingClassifier(architecture=args.model, config=config)
    threshold = args.threshold or config.decision_threshold

    if args.mode in ["all", "train"]:
        classifier, tuned_thresh = train_pipeline(
            train_dir=args.train_dir,
            model_save_path=args.model_path,
            n_sample_s1=args.n_train,
            n_val_s1=args.n_val,
            architecture=args.model,
            n_folds=args.n_folds,
            imbalance_strategy=args.imbalance,
            config=config
        )
        if args.threshold is None:
            threshold = tuned_thresh

    if args.mode in ["all", "inference"]:
        if not os.path.exists(args.model_path) and not os.path.exists(args.model_path + "_metadata.json"):
            print(f"Model file not found at {args.model_path}. Training first...")
            classifier, tuned_thresh = train_pipeline(
                train_dir=args.train_dir,
                model_save_path=args.model_path,
                n_sample_s1=args.n_train,
                n_val_s1=args.n_val,
                architecture=args.model,
                n_folds=args.n_folds,
                imbalance_strategy=args.imbalance,
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
