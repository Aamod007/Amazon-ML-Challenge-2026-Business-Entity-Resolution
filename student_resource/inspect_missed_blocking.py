import pandas as pd
from evaluate_blocking import get_name_tokens, get_address_tokens, STOPWORDS, RE_WORD
from collections import defaultdict

def inspect_missed_blocking():
    # Load same sample
    gt_df = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=5000)
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    val_s1_ids = set(gt_df["source1_entity_id"])
    
    true_matches = {}
    all_true_m = set()
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        mids = set(m.strip() for m in row["matched_entity_ids"].split(",") if m.strip())
        true_matches[s1] = mids
        all_true_m |= mids

    val_s1 = {}
    with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            p = line.strip().split("\t")
            if p[0] in val_s1_ids:
                val_s1[p[0]] = (p[1], p[2], p[3])
                if len(val_s1) == len(val_s1_ids): break

    m_records = {}
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src}"
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                p = line.strip().split("\t")
                if p[0] in all_true_m:
                    m_records[p[0]] = (p[1], p[2], p[3])
                    if len(m_records) == len(all_true_m): break

    # Build indices as in evaluate_blocking
    index_name_token = defaultdict(list)
    index_addr_num_word = defaultdict(list)
    index_name_prefix = defaultdict(list)
    
    for mid, rec in m_records.items():
        c = rec[2]
        n_tokens = get_name_tokens(rec[0])
        a_tokens = get_address_tokens(rec[1])
        for tok in n_tokens:
            if len(tok) >= 3: index_name_token[(c, tok)].append(mid)
        street_nums = [tok for tok in a_tokens if tok.isdigit()]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        if street_nums and words:
            s_num = street_nums[0]
            index_addr_num_word[(c, s_num, words[0][:4])].append(mid)
            if len(words) > 1:
                index_addr_num_word[(c, s_num, words[1][:4])].append(mid)
        if n_tokens and words:
            index_name_prefix[(c, n_tokens[0][:4], words[0][:3])].append(mid)

    missed = []
    for s1_id, rec in val_s1.items():
        c = rec[2]
        n_tokens = get_name_tokens(rec[0])
        a_tokens = get_address_tokens(rec[1])
        cands = set()
        for tok in n_tokens:
            if len(tok) >= 3:
                matches = index_name_token.get((c, tok), [])
                if len(matches) <= 200:
                    cands.update(matches)
        street_nums = [tok for tok in a_tokens if tok.isdigit()]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        if street_nums and words:
            s_num = street_nums[0]
            for w in words[:3]:
                cands.update(index_addr_num_word.get((c, s_num, w[:4]), []))
        if n_tokens and words:
            cands.update(index_name_prefix.get((c, n_tokens[0][:4], words[0][:3]), []))

        for true_m in true_matches.get(s1_id, set()):
            if true_m in m_records and true_m not in cands:
                missed.append((s1_id, rec, true_m, m_records[true_m]))

    print(f"Total missed true pairs in sample: {len(missed)}")
    print("\nSample 10 missed pairs:")
    for i, (s1_id, s1_r, mid, m_r) in enumerate(missed[:10], 1):
        print(f"\n--- Missed #{i} ({s1_r[2]}) ---")
        print(f"S1 Name:    {s1_r[0]}")
        print(f"Match Name: {m_r[0]}")
        print(f"S1 Addr:    {s1_r[1]}")
        print(f"Match Addr: {m_r[1]}")

if __name__ == "__main__":
    inspect_missed_blocking()
