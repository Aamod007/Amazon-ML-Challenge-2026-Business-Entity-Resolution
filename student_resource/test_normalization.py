import re
import unicodedata
import pandas as pd
from collections import Counter

# Compile Regex Patterns
RE_COMBINING = re.compile(r"[\u0300-\u036f]")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")

# Landmark pattern
RE_LANDMARK = re.compile(
    r"\b(?:near|opp\.?|opposite|behind|b/h|beside|adjacent(?:\s+to)?|next\s+to|in\s+front\s+of|close\s+to)\s+([^,;]+)",
    re.IGNORECASE
)

# Postal / PIN code pattern: 6 digits (India) or 5 digits (US/France)
RE_POSTAL = re.compile(r"\b([1-9]\d{5}|\d{5}(?:-\d{4})?)\b")

# Street number pattern
RE_STREET_NUM = re.compile(r"\b(?:(?:h\.?no\.?|house\s+no\.?|plot\s+no\.?|flat\s+no\.?|shop\s+no\.?|unit\s+no\.?|no\.?|#)\s*)?(\d+[-/]?\w*)\b", re.IGNORECASE)

# Legal suffix table (canonical target)
LEGAL_SUFFIX_MAP = {
    # English / International
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "pvt": "private", "private": "private",
    "co": "company", "company": "company",
    "llc": "llc", "l.l.c.": "llc",
    "llp": "llp", "l.l.p.": "llp",
    "pllc": "pllc", "p.l.l.c.": "pllc",
    "pc": "pc", "p.c.": "pc",
    "plc": "plc", "p.l.c.": "plc",
    # French
    "sarl": "sarl", "s.a.r.l.": "sarl",
    "sas": "sas", "s.a.s.": "sas",
    "sasu": "sasu", "s.a.s.u.": "sasu",
    "sa": "sa", "s.a.": "sa",
    "sci": "sci", "s.c.i.": "sci",
    "eurl": "eurl", "e.u.r.l.": "eurl",
    "snc": "snc", "s.n.c.": "snc",
    "ste": "societe", "societe": "societe", "société": "societe",
    # Hindi / Devanagari
    "प्राइवेट लिमिटेड": "private limited",
    "प्रा. लि.": "private limited",
    "प्रा लि": "private limited",
    "लिमिटेड": "limited",
    "लि.": "limited",
    "एलएलपी": "llp",
    "कंपनी": "company",
    "कम्पनी": "company",
}

RE_LEGAL_SUFFIX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

def normalize_text_unicode(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFKD", text)
    text = RE_COMBINING.sub("", text)
    return text.lower().strip()

def normalize_name(raw_name: str):
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {"norm_name": "", "root_name": "", "had_legal_suffix": False, "legal_suffix": ""}
    
    # 1. Unicode & lowercase
    t = normalize_text_unicode(raw_name)
    
    # 2. Check legal suffix
    match = RE_LEGAL_SUFFIX.search(t)
    had_legal_suffix = bool(match)
    legal_suffix = match.group(0).lower() if match else ""
    
    # Replace & with and
    t = re.sub(r"\s*&\s*", " and ", t)
    
    # Remove punctuation
    t_clean = RE_PUNCT.sub(" ", t)
    t_clean = RE_SPACES.sub(" ", t_clean).strip()
    
    # Root name without legal suffix
    root_name = RE_LEGAL_SUFFIX.sub("", t_clean)
    root_name = RE_SPACES.sub(" ", root_name).strip()
    
    return {
        "norm_name": t_clean,
        "root_name": root_name,
        "had_legal_suffix": had_legal_suffix,
        "legal_suffix": LEGAL_SUFFIX_MAP.get(legal_suffix, legal_suffix)
    }

def normalize_address(raw_address: str):
    if not isinstance(raw_address, str) or not raw_address.strip():
        return {
            "norm_address": "", "landmark": "", "postal_code": "",
            "street_num": "", "address_residual": ""
        }
        
    t = normalize_text_unicode(raw_address)
    
    # 1. Landmark extraction
    landmark_match = RE_LANDMARK.search(t)
    landmark = landmark_match.group(1).strip() if landmark_match else ""
    
    # Remove landmark from address residual
    t_no_landmark = RE_LANDMARK.sub(" ", t) if landmark_match else t
    
    # 2. Postal code extraction
    postal_match = RE_POSTAL.search(t_no_landmark)
    postal_code = postal_match.group(1).strip() if postal_match else ""
    
    # 3. Street number extraction
    street_num_match = RE_STREET_NUM.search(t_no_landmark)
    street_num = street_num_match.group(1).strip() if street_num_match else ""
    
    # 4. Clean residual address
    clean_addr = RE_PUNCT.sub(" ", t_no_landmark)
    clean_addr = RE_SPACES.sub(" ", clean_addr).strip()
    
    return {
        "norm_address": RE_SPACES.sub(" ", RE_PUNCT.sub(" ", t)).strip(),
        "landmark": landmark,
        "postal_code": postal_code,
        "street_num": street_num,
        "address_residual": clean_addr
    }

def test_normalization_coverage():
    print("Testing normalization coverage across train and test samples...")
    # Load 10k train_source1, 10k train_source2, 10k test_source1 (includes France)
    s1_train = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", nrows=10000)
    s2_train = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t", nrows=10000)
    s1_test = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", nrows=10000)
    
    sample_df = pd.concat([s1_train, s2_train, s1_test], ignore_index=True)
    n = len(sample_df)
    print(f"Total sampled records for coverage audit: {n:,}")
    
    stats = Counter()
    for _, row in sample_df.iterrows():
        name_res = normalize_name(str(row["business_name"]))
        addr_res = normalize_address(str(row["business_address"]))
        
        if name_res["had_legal_suffix"]:
            stats["name_has_legal_suffix"] += 1
        if name_res["norm_name"] != str(row["business_name"]).strip().lower():
            stats["name_transformed"] += 1
        if addr_res["landmark"]:
            stats["addr_has_landmark"] += 1
        if addr_res["postal_code"]:
            stats["addr_has_postal_code"] += 1
        if addr_res["street_num"]:
            stats["addr_has_street_num"] += 1
            
    print("\nNormalization Rule Coverage Rates:")
    for rule, count in stats.items():
        print(f"  - {rule:25s}: {count:,} / {n:,} ({count / n * 100:.2f}%)")

if __name__ == "__main__":
    test_normalization_coverage()
