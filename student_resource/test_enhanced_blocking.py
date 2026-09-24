import time
import re
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

RE_WORD = re.compile(r"\w+")

NAME_STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "co", "company", "llc", "llp", "the", "and", "of",
    "services", "solutions", "enterprises", "group", "holdings", "management"
}

ADDR_STOPWORDS = {
    "road", "rd", "street", "st", "lane", "ln", "avenue", "ave", "floor",
    "near", "opp", "opposite", "behind", "flat", "plot", "no", "co", "c", "o",
    "dr", "drive", "way", "blvd", "boulevard", "h", "house", "shop", "block",
    "bldg", "building", "apt", "apartment", "unit", "suite", "north", "south",
    "east", "west", "new", "delhi", "nagar", "colony", "city"
}

def get_name_tokens(norm_name):
    words = [w.lower() for w in RE_WORD.findall(norm_name) if len(w) >= 2]
    return [w for w in words if w not in NAME_STOPWORDS]

def get_address_tokens(norm_address):
    words = [w.lower() for w in RE_WORD.findall(norm_address) if len(w) >= 2]
    return [w for w in words if w not in ADDR_STOPWORDS]

def test_enhanced_blocking():
    print("Testing enhanced multi-strategy blocking...")
    gt_df = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=10000)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    
    val_s1_ids = set(gt_df["source1_entity_id"])
    true_matches = {}
    all_true_m_ids = set()
    total_true_pairs = 0
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        mids = set(m.strip() for m in row["matched_entity_ids"].split(",") if m.strip())
        true_matches[s1] = mids
        all_true_m_ids |= mids
        total_true_pairs += len(mids)
        
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
                
    candidate_pool = {}
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src}"
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    if eid in all_true_m_ids or len(candidate_pool) < 150000:
                        candidate_pool[eid] = {
                            "name": parts[1], "address": parts[2], "country": parts[3]
                        }
                        
    print(f"Pool size: {len(candidate_pool):,} records. True targets: {len(all_true_m_ids):,}")

    start_time = time.time()
    
    # Indices:
    # 1. Name token index: (country, token) -> [m_id]
    # 2. Address Street Num + Token: (country, num, token[:4]) -> [m_id]
    # 3. Address Distinctive Token-Pair: (country, tok_a, tok_b) -> [m_id]
    # 4. Name prefix + Addr prefix: (country, name_prefix, addr_prefix) -> [m_id]
    
    idx_name_token = defaultdict(list)
    idx_addr_num_word = defaultdict(list)
    idx_addr_pair = defaultdict(list)
    idx_prefix = defaultdict(list)
    
    # Calculate corpus token frequencies to find rare tokens
    addr_token_counts = Counter()
    for rec in candidate_pool.values():
        a_tokens = get_address_tokens(rec["address"])
        for t in set(a_tokens):
            if len(t) >= 4 and not t.isdigit():
                addr_token_counts[t] += 1
                
    for mid, rec in candidate_pool.items():
        c = rec["country"]
        n_tokens = get_name_tokens(rec["name"])
        raw_a_tokens = [w.lower() for w in RE_WORD.findall(rec["address"]) if len(w) >= 2]
        a_tokens = [w for w in raw_a_tokens if w not in ADDR_STOPWORDS]
        
        # 1. Name tokens
        for tok in n_tokens:
            if len(tok) >= 3:
                idx_name_token[(c, tok)].append(mid)
                
        # 2. Street number + address word
        street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
        clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        if clean_nums and words:
            s_num = clean_nums[0]
            for w in words[:3]:
                idx_addr_num_word[(c, s_num, w[:4])].append(mid)
                
        # 3. Address token pairs (sort by rarity)
        rare_words = sorted([w for w in words if len(w) >= 4], key=lambda w: addr_token_counts[w])
        if len(rare_words) >= 2:
            w1, w2 = sorted([rare_words[0], rare_words[1]])
            idx_addr_pair[(c, w1, w2)].append(mid)
            if len(rare_words) >= 3:
                w1, w3 = sorted([rare_words[0], rare_words[2]])
                idx_addr_pair[(c, w1, w3)].append(mid)
                
        # 4. Name prefix + addr prefix
        if n_tokens and words:
            idx_prefix[(c, n_tokens[0][:4], words[0][:3])].append(mid)

    index_time = time.time() - start_time
    print(f"Indices built in {index_time:.2f}s.")

    # Query for S1 entities
    start_query = time.time()
    total_cands = 0
    true_pairs_found = 0
    
    for s1_id, rec in val_s1_records.items():
        c = rec["country"]
        n_tokens = get_name_tokens(rec["name"])
        raw_a_tokens = [w.lower() for w in RE_WORD.findall(rec["address"]) if len(w) >= 2]
        a_tokens = [w for w in raw_a_tokens if w not in ADDR_STOPWORDS]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
        clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]

        cand_set = set()
        
        # 1. Name tokens (cap list size to 300)
        for tok in n_tokens:
            if len(tok) >= 3:
                matches = idx_name_token.get((c, tok), [])
                if len(matches) <= 300:
                    cand_set.update(matches)
                    
        # 2. Street number + words
        if clean_nums and words:
            s_num = clean_nums[0]
            for w in words[:3]:
                cand_set.update(idx_addr_num_word.get((c, s_num, w[:4]), []))
                
        # 3. Address pairs
        rare_words = sorted([w for w in words if len(w) >= 4], key=lambda w: addr_token_counts[w])
        if len(rare_words) >= 2:
            w1, w2 = sorted([rare_words[0], rare_words[1]])
            cand_set.update(idx_addr_pair.get((c, w1, w2), []))
            if len(rare_words) >= 3:
                w1, w3 = sorted([rare_words[0], rare_words[2]])
                cand_set.update(idx_addr_pair.get((c, w1, w3), []))
                
        # 4. Prefix
        if n_tokens and words:
            cand_set.update(idx_prefix.get((c, n_tokens[0][:4], words[0][:3]), []))
            
        total_cands += len(cand_set)
        true_for_s1 = true_matches.get(s1_id, set())
        true_pairs_found += len(cand_set.intersection(true_for_s1))

    query_time = time.time() - start_query
    print(f"Query completed in {query_time:.2f}s.")
    
    pair_completeness = true_pairs_found / total_true_pairs if total_true_pairs > 0 else 0
    full_cross = len(val_s1_records) * len(candidate_pool)
    reduction = 1.0 - (total_cands / full_cross)
    
    print("\n" + "="*60)
    print("ENHANCED BLOCKING EVALUATION RESULTS")
    print("="*60)
    print(f"Total True Pairs in Validation Set: {total_true_pairs:,}")
    print(f"True Pairs Captured by Blocking   : {true_pairs_found:,}")
    print(f"Pair Completeness (Recall Ceiling): {pair_completeness * 100:.2f}%")
    print(f"Total Candidate Pairs Generated   : {total_cands:,}")
    print(f"Average Candidates per S1 Entity  : {total_cands / len(val_s1_records):.1f}")
    print(f"Reduction Ratio                   : {reduction * 100:.6f}%")

if __name__ == "__main__":
    test_enhanced_blocking()
