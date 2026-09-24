import time
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from collections import defaultdict, Counter

from test_normalization import normalize_name, normalize_address

RE_WORD = re.compile(r"\w+")

def char_ngrams(text, n=3):
    return set(text[i:i+n] for i in range(len(text) - n + 1)) if len(text) >= n else set()

def jaccard_sim(set_a, set_b):
    if not set_a and not set_b: return 1.0
    if not set_a or not set_b: return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def compute_pair_features(rec1, rec2):
    n1 = rec1["norm_name"]
    n2 = rec2["norm_name"]
    rn1 = rec1["root_name"]
    rn2 = rec2["root_name"]
    
    a1 = rec1["norm_address"]
    a2 = rec2["norm_address"]
    
    nr = fuzz.ratio(n1, n2)
    npr = fuzz.partial_ratio(n1, n2)
    ntsort = fuzz.token_sort_ratio(n1, n2)
    ntset = fuzz.token_set_ratio(n1, n2)
    rnr = fuzz.ratio(rn1, rn2)
    
    n1_3g = char_ngrams(n1, 3)
    n2_3g = char_ngrams(n2, 3)
    n_3g_jac = jaccard_sim(n1_3g, n2_3g)
    
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    root_exact = 1.0 if (rn1 and rn1 == rn2) else 0.0
    prefix_3 = 1.0 if (len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3]) else 0.0
    
    len_diff_n = abs(len(n1) - len(n2))
    len_ratio_n = (min(len(n1), len(n2)) / max(len(n1), len(n2))) if (len(n1) > 0 and len(n2) > 0) else 0.0

    both_suffix = 1.0 if (rec1["had_legal_suffix"] and rec2["had_legal_suffix"]) else 0.0
    suffix_match = 1.0 if (both_suffix and rec1["legal_suffix"] == rec2["legal_suffix"]) else 0.0
    
    ar = fuzz.ratio(a1, a2)
    apr = fuzz.partial_ratio(a1, a2)
    atsort = fuzz.token_sort_ratio(a1, a2)
    atset = fuzz.token_set_ratio(a1, a2)
    
    a1_words = set(RE_WORD.findall(a1))
    a2_words = set(RE_WORD.findall(a2))
    a_word_jac = jaccard_sim(a1_words, a2_words)
    
    a1_3g = char_ngrams(a1, 3)
    a2_3g = char_ngrams(a2, 3)
    a_3g_jac = jaccard_sim(a1_3g, a2_3g)
    
    len_diff_a = abs(len(a1) - len(a2))

    pc1 = rec1["postal_code"]
    pc2 = rec2["postal_code"]
    if pc1 and pc2:
        pc_exact = 1.0 if pc1 == pc2 else 0.0
        pc_mismatch = 1.0 if pc1 != pc2 else 0.0
    else:
        pc_exact = 0.0
        pc_mismatch = 0.0
        
    sn1 = rec1["street_num"]
    sn2 = rec2["street_num"]
    if sn1 and sn2:
        sn_exact = 1.0 if sn1 == sn2 else 0.0
        sn_mismatch = 1.0 if sn1 != sn2 else 0.0
    else:
        sn_exact = 0.0
        sn_mismatch = 0.0

    lm1 = rec1["landmark"]
    lm2 = rec2["landmark"]
    lm_match = 1.0 if (lm1 and lm2 and fuzz.ratio(lm1, lm2) > 80) else 0.0

    harmonic = 2.0 * (ntset * atset) / (ntset + atset + 1e-5)
    high_both = 1.0 if (ntset >= 80 and atset >= 80) else 0.0

    return [
        nr, npr, ntsort, ntset, rnr, n_3g_jac, name_exact, root_exact, prefix_3, len_diff_n, len_ratio_n,
        both_suffix, suffix_match,
        ar, apr, atsort, atset, a_word_jac, a_3g_jac, len_diff_a,
        pc_exact, pc_mismatch, sn_exact, sn_mismatch, lm_match,
        harmonic, high_both
    ]

FEATURE_NAMES = [
    "name_ratio", "name_partial_ratio", "name_token_sort", "name_token_set", "root_name_ratio",
    "name_3g_jaccard", "name_exact", "root_exact", "name_prefix_3", "name_len_diff", "name_len_ratio",
    "both_have_legal_suffix", "legal_suffix_match",
    "addr_ratio", "addr_partial_ratio", "addr_token_sort", "addr_token_set", "addr_word_jaccard",
    "addr_3g_jaccard", "addr_len_diff",
    "postal_exact", "postal_mismatch", "street_num_exact", "street_num_mismatch", "landmark_match",
    "harmonic_name_addr", "high_both_sim"
]

def evaluate_macro_f05(y_true_dict, y_pred_dict, all_s1_ids):
    scores = []
    for s1 in all_s1_ids:
        t_set = y_true_dict.get(s1, set())
        p_set = y_pred_dict.get(s1, set())
        
        if not t_set:  # True singleton
            if not p_set:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:  # Has true matches
            if not p_set:
                scores.append(0.0)
            else:
                tp = len(t_set.intersection(p_set))
                if tp == 0:
                    scores.append(0.0)
                else:
                    prec = tp / len(p_set)
                    rec = tp / len(t_set)
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                    scores.append(f05)
    return float(np.mean(scores))

def apply_global_consistency(s1_cand_probs, threshold):
    # Pass 1: Filter by threshold and record all (s1, mid, prob)
    s2_best = {} # mid -> (best_s1, best_prob)
    for s1_id, cand_list in s1_cand_probs.items():
        for mid, prob in cand_list:
            if prob >= threshold:
                if mid not in s2_best or prob > s2_best[mid][1]:
                    s2_best[mid] = (s1_id, prob)
                    
    # Pass 2: Reconstruct predictions keeping only the winning S1 for each mid
    resolved_preds = defaultdict(set)
    for mid, (winning_s1, prob) in s2_best.items():
        resolved_preds[winning_s1].add(mid)
        
    return resolved_preds

def run_test():
    print("Loading data for end-to-end ML validation test...")
    gt_df = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=5000)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    val_s1_ids = list(gt_df["source1_entity_id"])
    
    true_matches = {}
    all_true_m = set()
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        mids = set(m.strip() for m in row["matched_entity_ids"].split(",") if m.strip())
        true_matches[s1] = mids
        all_true_m |= mids

    s1_dict = {}
    with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            p = line.strip().split("\t")
            if p[0] in true_matches:
                nm = normalize_name(p[1])
                ad = normalize_address(p[2])
                s1_dict[p[0]] = {**nm, **ad, "country": p[3], "raw_name": p[1], "raw_addr": p[2]}
                if len(s1_dict) == len(true_matches): break

    pool_dict = {}
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src}"
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4:
                    if p[0] in all_true_m or len(pool_dict) < 60000:
                        nm = normalize_name(p[1])
                        ad = normalize_address(p[2])
                        pool_dict[p[0]] = {**nm, **ad, "country": p[3], "raw_name": p[1], "raw_addr": p[2]}

    print(f"Loaded {len(s1_dict)} S1 entities and {len(pool_dict)} candidate pool entities.")

    from test_enhanced_blocking import get_name_tokens, get_address_tokens, ADDR_STOPWORDS
    idx_name_token = defaultdict(list)
    idx_addr_num_word = defaultdict(list)
    idx_addr_pair = defaultdict(list)
    idx_prefix = defaultdict(list)
    
    for mid, rec in pool_dict.items():
        c = rec["country"]
        n_tokens = get_name_tokens(rec["raw_name"])
        raw_a_tokens = [w.lower() for w in RE_WORD.findall(rec["raw_addr"]) if len(w) >= 2]
        a_tokens = [w for w in raw_a_tokens if w not in ADDR_STOPWORDS]
        for tok in n_tokens:
            if len(tok) >= 3: idx_name_token[(c, tok)].append(mid)
        street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
        clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        if clean_nums and words:
            s_num = clean_nums[0]
            for w in words[:3]:
                idx_addr_num_word[(c, s_num, w[:4])].append(mid)
        if len(words) >= 2:
            w1, w2 = sorted([words[0], words[1]])
            idx_addr_pair[(c, w1, w2)].append(mid)
        if n_tokens and words:
            idx_prefix[(c, n_tokens[0][:4], words[0][:3])].append(mid)

    pairs = []
    labels = []
    groups = []
    pair_ids = []
    
    for s1_id, rec1 in s1_dict.items():
        c = rec1["country"]
        n_tokens = get_name_tokens(rec1["raw_name"])
        raw_a_tokens = [w.lower() for w in RE_WORD.findall(rec1["raw_addr"]) if len(w) >= 2]
        a_tokens = [w for w in raw_a_tokens if w not in ADDR_STOPWORDS]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
        clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]

        cands = set()
        for tok in n_tokens:
            if len(tok) >= 3:
                matches = idx_name_token.get((c, tok), [])
                if len(matches) <= 250: cands.update(matches)
        if clean_nums and words:
            s_num = clean_nums[0]
            for w in words[:3]:
                cands.update(idx_addr_num_word.get((c, s_num, w[:4]), []))
        if len(words) >= 2:
            w1, w2 = sorted([words[0], words[1]])
            cands.update(idx_addr_pair.get((c, w1, w2), []))
        if n_tokens and words:
            cands.update(idx_prefix.get((c, n_tokens[0][:4], words[0][:3]), []))
            
        t_matches = true_matches[s1_id]
        for mid in cands:
            rec2 = pool_dict[mid]
            feat = compute_pair_features(rec1, rec2)
            lbl = 1 if mid in t_matches else 0
            pairs.append(feat)
            labels.append(lbl)
            groups.append(s1_id)
            pair_ids.append((s1_id, mid))

    X = np.array(pairs, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Candidate pairs matrix: {X.shape}, Positives: {sum(y):,} ({sum(y)/len(y)*100:.2f}%)")

    gkf = GroupKFold(n_splits=3)
    val_probs = np.zeros(len(y), dtype=np.float32)
    scale_pos_weight = (len(y) - sum(y)) / sum(y)

    models = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        print(f"Training fold {fold}...")
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            n_jobs=8,
            random_state=42,
            eval_metric="logloss"
        )
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        val_probs[val_idx] = probs
        models.append(clf)

    # Feature Importance Logging
    mean_importances = np.mean([m.feature_importances_ for m in models], axis=0)
    sorted_feat_idx = np.argsort(mean_importances)[::-1]
    print("\nTop 15 Most Important Features:")
    for rank, idx in enumerate(sorted_feat_idx[:15], 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:25s}: {mean_importances[idx]:.4f}")

    s1_cand_probs = defaultdict(list)
    for i, (s1_id, mid) in enumerate(pair_ids):
        s1_cand_probs[s1_id].append((mid, val_probs[i]))

    # Decision Threshold Sweeping
    best_thresh = 0.5
    best_raw_f05 = -1.0
    for thresh in np.linspace(0.80, 0.98, 19):
        raw_preds = {}
        for s1_id, cand_list in s1_cand_probs.items():
            raw_preds[s1_id] = set(mid for mid, p in cand_list if p >= thresh)
        score = evaluate_macro_f05(true_matches, raw_preds, val_s1_ids)
        if score > best_raw_f05:
            best_raw_f05 = score
            best_thresh = thresh

    print(f"\nOptimal Raw Threshold: {best_thresh:.3f} | Raw Macro F0.5: {best_raw_f05:.5f}")

    # STAGE 7: GLOBAL CONSISTENCY COMPARISON
    print("\n--- STAGE 7: GLOBAL CONSISTENCY EVALUATION ---")
    raw_preds_optimal = {}
    for s1_id, cand_list in s1_cand_probs.items():
        raw_preds_optimal[s1_id] = set(mid for mid, p in cand_list if p >= best_thresh)
    raw_score = evaluate_macro_f05(true_matches, raw_preds_optimal, val_s1_ids)

    # Count conflicts: S2/S3 entities claimed by > 1 S1 entity
    claimed_counts = Counter()
    for s1, mids in raw_preds_optimal.items():
        for mid in mids:
            claimed_counts[mid] += 1
    conflicts = sum(1 for mid, c in claimed_counts.items() if c > 1)
    print(f"Conflicting S2/S3 entities claimed by multiple S1 entities before resolution: {conflicts:,}")

    # Apply global 1-to-many resolution
    resolved_preds = apply_global_consistency(s1_cand_probs, best_thresh)
    resolved_score = evaluate_macro_f05(true_matches, resolved_preds, val_s1_ids)

    print(f"Before Global Consistency: Macro F0.5 = {raw_score:.5f}")
    print(f"After  Global Consistency: Macro F0.5 = {resolved_score:.5f}")
    diff = resolved_score - raw_score
    print(f"Net Improvement from Global Consistency: {diff:+.5f}")

    all_singletons = {s1: set() for s1 in val_s1_ids}
    baseline_score = evaluate_macro_f05(true_matches, all_singletons, val_s1_ids)
    print(f"\nTrivial Baseline (Predict Nothing): {baseline_score:.5f}")
    final_best = max(raw_score, resolved_score)
    print(f"Final Validation Score: {final_best:.5f} (+{final_best - baseline_score:.5f} above baseline)")

if __name__ == "__main__":
    run_test()
