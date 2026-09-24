import pandas as pd
from rapidfuzz import fuzz
import re

def investigate_low_name_sim():
    print("Investigating how true matches with low name similarity match...")
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=10000)
    gt = gt[gt["matched_entity_ids"].notna() & (gt["matched_entity_ids"] != "")]
    
    needed_s1 = set(gt["source1_entity_id"])
    needed_m = set()
    for mids in gt["matched_entity_ids"]:
        for m in mids.split(","):
            m = m.strip()
            if m: needed_m.add(m)
            
    s1_rows = {}
    with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in needed_s1:
                s1_rows[parts[0]] = (parts[1], parts[2], parts[3])
                if len(s1_rows) == len(needed_s1): break
                
    m_rows = {}
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src}"
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[0] in needed_m:
                    m_rows[parts[0]] = (parts[1], parts[2], parts[3])
                    if len(m_rows) == len(needed_m): break

    low_name_matches = []
    high_name_matches = []
    
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_rows: continue
        s1_n, s1_a, s1_c = s1_rows[s1_id]
        for mid in row["matched_entity_ids"].split(","):
            mid = mid.strip()
            if mid not in m_rows: continue
            m_n, m_a, m_c = m_rows[mid]
            
            nr = fuzz.ratio(s1_n.lower(), m_n.lower())
            tsr = fuzz.token_sort_ratio(s1_n.lower(), m_n.lower())
            best_name = max(nr, tsr)
            
            ar = fuzz.ratio(s1_a.lower(), m_a.lower())
            ats = fuzz.token_set_ratio(s1_a.lower(), m_a.lower())
            best_addr = max(ar, ats)
            
            if best_name < 40:
                low_name_matches.append({
                    "s1_n": s1_n, "m_n": m_n,
                    "s1_a": s1_a, "m_a": m_a,
                    "best_name": best_name,
                    "best_addr": best_addr,
                    "country": s1_c
                })
            else:
                high_name_matches.append(best_name)
                
    print(f"Total matching pairs examined: {len(low_name_matches) + len(high_name_matches)}")
    print(f"Pairs with best_name < 40: {len(low_name_matches)} ({len(low_name_matches)/(len(low_name_matches)+len(high_name_matches))*100:.2f}%)")
    print(f"Pairs with best_name >= 40: {len(high_name_matches)} ({len(high_name_matches)/(len(low_name_matches)+len(high_name_matches))*100:.2f}%)")
    
    print("\nSample low-name matches and their addresses:")
    for ex in low_name_matches[:10]:
        print(f"Country: {ex['country']}")
        print(f"  Names: {ex['s1_n']} <===> {ex['m_n']} (score: {ex['best_name']})")
        print(f"  Addrs: {ex['s1_a']} <===> {ex['m_a']} (score: {ex['best_addr']})")

if __name__ == "__main__":
    investigate_low_name_sim()
