import time
import re
from collections import defaultdict
import pandas as pd
import numpy as np

# Fast tokenization and cleaning
RE_WORD = re.compile(r"\w+")

STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "co", "company", "llc", "llp", "the", "and", "of",
    "services", "solutions", "enterprises", "group", "holdings", "management"
}

def get_name_tokens(norm_name):
    words = [w.lower() for w in RE_WORD.findall(norm_name) if len(w) >= 2]
    return [w for w in words if w not in STOPWORDS]

def get_address_tokens(norm_address):
    return [w.lower() for w in RE_WORD.findall(norm_address) if len(w) >= 2]

def test_blocking_on_val():
    print("Preparing validation slice for blocking evaluation...")
    
    # 1. Load 10,000 S1 entities from train
    gt_df = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=10000)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    
    val_s1_ids = set(gt_df["source1_entity_id"])
    
    # Build ground truth dict: s1_id -> set of true matched s2/s3 ids
    true_matches = {}
    all_true_m_ids = set()
    total_true_pairs = 0
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        mids = set(m.strip() for m in row["matched_entity_ids"].split(",") if m.strip())
        true_matches[s1] = mids
        all_true_m_ids |= mids
        total_true_pairs += len(mids)
        
    print(f"Validation S1 entities: {len(val_s1_ids):,}")
    print(f"Total true match pairs: {total_true_pairs:,}")
    print(f"Unique true S2/S3 target entities: {len(all_true_m_ids):,}")
    
    # Load S1 records
    val_s1_records = {}
    with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in val_s1_ids:
                val_s1_records[parts[0]] = {
                    "name": parts[1], "address": parts[2], "country": parts[3]
                }
                if len(val_s1_records) == len(val_s1_ids): break
                
    # Load matching S2/S3 records PLUS background records (total ~100k records for realistic test)
    candidate_pool = {}
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src}"
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    # Include if it is a true match OR up to 50k background rows
                    if eid in all_true_m_ids or len(candidate_pool) < 150000:
                        candidate_pool[eid] = {
                            "name": parts[1], "address": parts[2], "country": parts[3]
                        }
        print(f"Loaded pool size: {len(candidate_pool):,} records")

    print(f"Total candidate pool records: {len(candidate_pool):,}")
    
    # Verify all true targets are in pool
    found_targets = sum(1 for m in all_true_m_ids if m in candidate_pool)
    print(f"True targets present in pool: {found_targets} / {len(all_true_m_ids)}")

    # -------------------------------------------------------------------------
    # BUILD BLOCKING INDICES OVER CANDIDATE POOL (BY COUNTRY)
    # -------------------------------------------------------------------------
    start_time = time.time()
    
    # Strategies:
    # 1. Exact Name Token (rare/distinctive tokens)
    # 2. Address Street Number + First Word of Street/City
    # 3. Name 3-gram or First 4 chars prefix
    
    index_name_token = defaultdict(list)     # (country, token) -> [m_id]
    index_addr_num_word = defaultdict(list)  # (country, num, first_word) -> [m_id]
    index_name_prefix = defaultdict(list)    # (country, prefix4) -> [m_id]
    
    for mid, rec in candidate_pool.items():
        c = rec["country"]
        n_tokens = get_name_tokens(rec["name"])
        a_tokens = get_address_tokens(rec["address"])
        
        # Index Strategy 1: Name tokens
        for tok in n_tokens:
            if len(tok) >= 3:
                index_name_token[(c, tok)].append(mid)
                
        # Index Strategy 2: Street number + first street word
        street_nums = [tok for tok in a_tokens if tok.isdigit()]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        if street_nums and words:
            s_num = street_nums[0]
            s_word = words[0][:4]
            index_addr_num_word[(c, s_num, s_word)].append(mid)
            if len(words) > 1:
                index_addr_num_word[(c, s_num, words[1][:4])].append(mid)
                
        # Index Strategy 3: Name prefix (first 4 letters) + first address word prefix
        if n_tokens and words:
            p_name = n_tokens[0][:4]
            p_addr = words[0][:3]
            index_name_prefix[(c, p_name, p_addr)].append(mid)

    index_time = time.time() - start_time
    print(f"Indices built in {index_time:.2f}s.")

    # -------------------------------------------------------------------------
    # QUERY FOR S1 ENTITIES
    # -------------------------------------------------------------------------
    start_query = time.time()
    candidates_per_s1 = {}
    total_candidates_generated = 0
    true_pairs_found = 0
    
    for s1_id, rec in val_s1_records.items():
        c = rec["country"]
        n_tokens = get_name_tokens(rec["name"])
        a_tokens = get_address_tokens(rec["address"])
        
        cand_set = set()
        
        # Strategy 1: Name tokens (cap per token list to prevent explosive blocks)
        for tok in n_tokens:
            if len(tok) >= 3:
                matches = index_name_token.get((c, tok), [])
                if len(matches) <= 200:  # don't explode on overly frequent tokens
                    cand_set.update(matches)
                    
        # Strategy 2: Street number + street word
        street_nums = [tok for tok in a_tokens if tok.isdigit()]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        if street_nums and words:
            s_num = street_nums[0]
            for w in words[:3]:
                cand_set.update(index_addr_num_word.get((c, s_num, w[:4]), []))
                
        # Strategy 3: Prefix name + prefix addr
        if n_tokens and words:
            cand_set.update(index_name_prefix.get((c, n_tokens[0][:4], words[0][:3]), []))
            
        candidates_per_s1[s1_id] = cand_set
        total_candidates_generated += len(cand_set)
        
        # Check completeness
        true_for_s1 = true_matches.get(s1_id, set())
        found_for_s1 = len(cand_set.intersection(true_for_s1))
        true_pairs_found += found_for_s1

    query_time = time.time() - start_query
    print(f"Query completed in {query_time:.2f}s.")

    # -------------------------------------------------------------------------
    # EVALUATION METRICS
    # -------------------------------------------------------------------------
    pair_completeness = true_pairs_found / total_true_pairs if total_true_pairs > 0 else 0
    full_cross_product = len(val_s1_records) * len(candidate_pool)
    reduction_ratio = 1.0 - (total_candidates_generated / full_cross_product)
    avg_cands_per_s1 = total_candidates_generated / len(val_s1_records)

    print("\n" + "="*60)
    print("BLOCKING EVALUATION RESULTS")
    print("="*60)
    print(f"Total True Pairs in Validation Set: {total_true_pairs:,}")
    print(f"True Pairs Captured by Blocking   : {true_pairs_found:,}")
    print(f"Pair Completeness (Recall Ceiling): {pair_completeness * 100:.2f}%")
    print(f"Total Candidate Pairs Generated   : {total_candidates_generated:,}")
    print(f"Average Candidates per S1 Entity  : {avg_cands_per_s1:.1f}")
    print(f"Full Cross Product Size           : {full_cross_product:,}")
    print(f"Reduction Ratio                   : {reduction_ratio * 100:.6f}%")

if __name__ == "__main__":
    test_blocking_on_val()
