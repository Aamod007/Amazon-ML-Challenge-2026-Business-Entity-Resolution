import pandas as pd
import re
from rapidfuzz import fuzz

def analyze_noise():
    print("Loading samples to catalog noise patterns...")
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=20000)
    gt = gt[gt["matched_entity_ids"].notna() & (gt["matched_entity_ids"] != "")]
    
    # Get all target IDs we need
    needed_s1 = set(gt["source1_entity_id"])
    needed_m = set()
    for mids in gt["matched_entity_ids"]:
        for m in mids.split(","):
            m = m.strip()
            if m:
                needed_m.add(m)
                
    print(f"Targeting {len(needed_s1)} S1 entities and {len(needed_m)} matched S2/S3 entities...")
    
    # Load S1 matching rows
    s1_rows = {}
    with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        c_idx = header.index("country")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[id_idx] in needed_s1:
                s1_rows[parts[id_idx]] = (parts[name_idx], parts[addr_idx], parts[c_idx])
                if len(s1_rows) == len(needed_s1):
                    break
                    
    print(f"Loaded {len(s1_rows)} S1 records.")
    
    # Load S2 and S3 matching rows
    m_rows = {}
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src}"
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            addr_idx = header.index("business_address")
            c_idx = header.index("country")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[id_idx] in needed_m:
                    m_rows[parts[id_idx]] = (parts[name_idx], parts[addr_idx], parts[c_idx])
                    if len(m_rows) == len(needed_m):
                        break
        print(f"Loaded {len(m_rows)} matched records so far...")

    # Now correlate pairs
    pairs = []
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_rows:
            continue
        s1_name, s1_addr, s1_country = s1_rows[s1_id]
        for mid in row["matched_entity_ids"].split(","):
            mid = mid.strip()
            if mid in m_rows:
                m_name, m_addr, m_country = m_rows[mid]
                pairs.append({
                    "s1_id": s1_id,
                    "m_id": mid,
                    "s1_name": s1_name,
                    "m_name": m_name,
                    "s1_addr": s1_addr,
                    "m_addr": m_addr,
                    "country": s1_country,
                    "name_ratio": fuzz.ratio(s1_name.lower(), m_name.lower()),
                    "name_token_sort": fuzz.token_sort_ratio(s1_name.lower(), m_name.lower()),
                    "addr_ratio": fuzz.ratio(s1_addr.lower(), m_addr.lower()),
                    "addr_token_set": fuzz.token_set_ratio(s1_addr.lower(), m_addr.lower()),
                })

    print(f"\nTotal extracted matching pairs: {len(pairs)}")
    
    # Categorize noise types
    noise_categories = {
        "abbreviations": [],
        "transliteration_or_script": [],
        "token_reordering": [],
        "typos": [],
        "missing_address_components": [],
        "landmark_references": [],
        "punctuation_inconsistencies": []
    }
    
    landmark_keywords = ["near", "opp", "opposite", "behind", "beside", "adjacent", "next to", "front of", "near to"]
    legal_suffixes = ["corp", "corporation", "ltd", "limited", "pvt", "private", "inc", "incorporated", "co", "company", "llc", "llp"]

    for p in pairs:
        s1_n = p["s1_name"].lower()
        m_n = p["m_name"].lower()
        s1_a = p["s1_addr"].lower()
        m_a = p["m_addr"].lower()
        
        # 1. Landmarks
        if any(re.search(r"\b" + kw + r"\b", m_a) for kw in landmark_keywords) and not any(re.search(r"\b" + kw + r"\b", s1_a) for kw in landmark_keywords):
            if len(noise_categories["landmark_references"]) < 5:
                noise_categories["landmark_references"].append(p)
                
        # 2. Transliteration / non-ASCII
        if any(ord(c) > 127 for c in s1_n + m_n + s1_a + m_a):
            if len(noise_categories["transliteration_or_script"]) < 5:
                noise_categories["transliteration_or_script"].append(p)
                
        # 3. Token reordering: token_sort_ratio high (>85) but fuzz.ratio lower (<65)
        if p["name_token_sort"] > 85 and p["name_ratio"] < 65:
            if len(noise_categories["token_reordering"]) < 5:
                noise_categories["token_reordering"].append(p)
                
        # 4. Typos: high ratio (75 - 95), length similar, but not identical
        if 75 <= p["name_ratio"] < 95 and abs(len(s1_n) - len(m_n)) <= 3:
            if len(noise_categories["typos"]) < 5:
                noise_categories["typos"].append(p)
                
        # 5. Abbreviations: legal suffix difference or & vs and
        if any(s in s1_n for s in legal_suffixes) != any(s in m_n for s in legal_suffixes) or ("&" in s1_n and "and" in m_n) or ("and" in s1_n and "&" in m_n):
            if len(noise_categories["abbreviations"]) < 5:
                noise_categories["abbreviations"].append(p)
                
        # 6. Punctuation inconsistencies: removing punctuation makes them identical or near-identical
        clean_s1 = re.sub(r"[^\w\s]", "", s1_n)
        clean_m = re.sub(r"[^\w\s]", "", m_n)
        if s1_n != m_n and clean_s1 == clean_m:
            if len(noise_categories["punctuation_inconsistencies"]) < 5:
                noise_categories["punctuation_inconsistencies"].append(p)
                
        # 7. Missing address components: length of one address is < 50% of the other, but token set ratio > 80
        if (len(s1_a) < 0.6 * len(m_a) or len(m_a) < 0.6 * len(s1_a)) and p["addr_token_set"] > 85:
            if len(noise_categories["missing_address_components"]) < 5:
                noise_categories["missing_address_components"].append(p)

    print("\n" + "="*80)
    print("NOISE CATALOG WITH CONCRETE EXAMPLES FROM TRAINING DATA")
    print("="*80)
    for cat, examples in noise_categories.items():
        print(f"\n--- Category: {cat.upper()} ({len(examples)} examples found) ---")
        for i, ex in enumerate(examples, 1):
            print(f"[{i}] Country: {ex['country']}")
            print(f"    S1 ID: {ex['s1_id']} | M ID: {ex['m_id']}")
            print(f"    S1 Name:    {ex['s1_name']}")
            print(f"    Match Name: {ex['m_name']}")
            print(f"    S1 Addr:    {ex['s1_addr']}")
            print(f"    Match Addr: {ex['m_addr']}")
            print(f"    Sim Scores: Name Ratio={ex['name_ratio']}, TokenSort={ex['name_token_sort']}, AddrTokenSet={ex['addr_token_set']}")

if __name__ == "__main__":
    analyze_noise()
