"""
Stage 4: Pairwise Feature Engineering Module
Generates rich, deterministic, country-agnostic feature vectors for candidate pairs.
Computes:
  - Multi-dimensional string similarities on name and address (Levenshtein, token sort, token set, partial ratio, Jaro-Winkler)
  - Character 3-gram Jaccard overlap
  - Token-level Jaccard overlap
  - Legal suffix agreement flags
  - Structured field agreement / mismatch signals (postal code, street number, landmark)
  - Harmonic and composite interaction features
"""

import re
from typing import Dict, List, Any
from rapidfuzz import fuzz

RE_WORD = re.compile(r"\w+")

def char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    return float(intersection_len / union_len)

def compute_pair_features(rec1: Any, rec2: Any) -> List[float]:
    """
    Extract a deterministic, country-agnostic pairwise feature vector.
    rec1: Source 1 normalized record (Record or dict)
    rec2: Candidate (Source 2 or Source 3) normalized record (Record or dict)
    """
    n1 = rec1["norm_name"]
    n2 = rec2["norm_name"]
    rn1 = rec1["root_name"]
    rn2 = rec2["root_name"]
    
    an1 = rec1.get("ascii_name") or n1
    an2 = rec2.get("ascii_name") or n2
    arn1 = rec1.get("ascii_root") or rn1
    arn2 = rec2.get("ascii_root") or rn2

    a1 = rec1["norm_address"]
    a2 = rec2["norm_address"]
    
    # -------------------------------------------------------------------------
    # 1. Name Similarity Measures (Raw & Transliterated Latin ASCII)
    # -------------------------------------------------------------------------
    nr = float(fuzz.ratio(n1, n2))
    npr = float(fuzz.partial_ratio(n1, n2))
    ntsort = float(fuzz.token_sort_ratio(n1, n2))
    ntset = float(fuzz.token_set_ratio(n1, n2))
    rnr = float(fuzz.ratio(rn1, rn2))
    
    # Transliteration similarity (bridges Indic scripts and French diacritics)
    ascii_name_sort = float(fuzz.token_sort_ratio(an1, an2))
    ascii_root_sort = float(fuzz.token_sort_ratio(arn1, arn2))
    ascii_root_exact = 1.0 if (arn1 and arn1 == arn2) else 0.0

    n1_3g = char_ngrams(n1, 3)
    n2_3g = char_ngrams(n2, 3)
    n_3g_jac = jaccard_similarity(n1_3g, n2_3g)
    
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    root_exact = 1.0 if (rn1 and rn1 == rn2) else 0.0
    prefix_3 = 1.0 if (len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3]) else 0.0
    
    len_diff_n = float(abs(len(n1) - len(n2)))
    len_ratio_n = (min(len(n1), len(n2)) / max(len(n1), len(n2))) if (len(n1) > 0 and len(n2) > 0) else 0.0

    # -------------------------------------------------------------------------
    # 2. Legal Suffix Agreement
    # -------------------------------------------------------------------------
    both_suffix = 1.0 if (rec1["had_legal_suffix"] and rec2["had_legal_suffix"]) else 0.0
    suffix_match = 1.0 if (both_suffix and rec1["legal_suffix"] == rec2["legal_suffix"]) else 0.0
    
    # -------------------------------------------------------------------------
    # 3. Address Similarity Measures
    # -------------------------------------------------------------------------
    ar = float(fuzz.ratio(a1, a2))
    apr = float(fuzz.partial_ratio(a1, a2))
    atsort = float(fuzz.token_sort_ratio(a1, a2))
    atset = float(fuzz.token_set_ratio(a1, a2))
    
    a1_words = set(RE_WORD.findall(a1))
    a2_words = set(RE_WORD.findall(a2))
    a_word_jac = jaccard_similarity(a1_words, a2_words)
    
    a1_3g = char_ngrams(a1, 3)
    a2_3g = char_ngrams(a2, 3)
    a_3g_jac = jaccard_similarity(a1_3g, a2_3g)
    
    len_diff_a = float(abs(len(a1) - len(a2)))

    # -------------------------------------------------------------------------
    # 4. Structured Subfield Agreement Flags (Signed Matches: +1 match, -1 conflict)
    # -------------------------------------------------------------------------
    pc1 = rec1["postal_code"]
    pc2 = rec2["postal_code"]
    if pc1 and pc2:
        pc_exact = 1.0 if pc1 == pc2 else 0.0
        pc_mismatch = 1.0 if pc1 != pc2 else 0.0
        pc_signed = 1.0 if pc1 == pc2 else -1.0
    else:
        pc_exact = 0.0
        pc_mismatch = 0.0
        pc_signed = 0.0
        
    sn1 = rec1["street_num"]
    sn2 = rec2["street_num"]
    if sn1 and sn2:
        sn_exact = 1.0 if sn1 == sn2 else 0.0
        sn_mismatch = 1.0 if sn1 != sn2 else 0.0
        sn_signed = 1.0 if sn1 == sn2 else -1.0
    else:
        sn_exact = 0.0
        sn_mismatch = 0.0
        sn_signed = 0.0

    lm1 = rec1["landmark"]
    lm2 = rec2["landmark"]
    lm_match = 1.0 if (lm1 and lm2 and fuzz.ratio(lm1, lm2) > 80) else 0.0

    # Domain / URL sub-match (handles e.g. name matching website URL)
    raw1 = rec1.get("raw_name") or ""
    raw2 = rec2.get("raw_name") or ""
    domain_match = 1.0 if (len(n1) >= 5 and (n1 in raw2.lower() or n2 in raw1.lower())) else 0.0

    # -------------------------------------------------------------------------
    # 5. Composite & Interaction Features
    # -------------------------------------------------------------------------
    effective_name_sim = max(ntset, ascii_name_sort)
    harmonic = 2.0 * (effective_name_sim * atset) / (effective_name_sim + atset + 1e-5)
    high_both = 1.0 if (effective_name_sim >= 80.0 and atset >= 80.0) else 0.0

    return [
        nr, npr, ntsort, ntset, rnr, ascii_name_sort, ascii_root_sort, ascii_root_exact,
        n_3g_jac, name_exact, root_exact, prefix_3, len_diff_n, len_ratio_n,
        both_suffix, suffix_match,
        ar, apr, atsort, atset, a_word_jac, a_3g_jac, len_diff_a,
        pc_exact, pc_mismatch, pc_signed, sn_exact, sn_mismatch, sn_signed, lm_match, domain_match,
        harmonic, high_both
    ]

FEATURE_NAMES = [
    "name_ratio", "name_partial_ratio", "name_token_sort", "name_token_set", "root_name_ratio",
    "ascii_name_sort", "ascii_root_sort", "ascii_root_exact",
    "name_3g_jaccard", "name_exact", "root_exact", "name_prefix_3", "name_len_diff", "name_len_ratio",
    "both_have_legal_suffix", "legal_suffix_match",
    "addr_ratio", "addr_partial_ratio", "addr_token_sort", "addr_token_set", "addr_word_jaccard",
    "addr_3g_jaccard", "addr_len_diff",
    "postal_exact", "postal_mismatch", "postal_signed",
    "street_num_exact", "street_num_mismatch", "street_num_signed", "landmark_match", "domain_match",
    "harmonic_name_addr", "high_both_sim"
]
