"""
Stage 4: Advanced Pairwise Feature Engineering Module
Amazon ML Challenge 2026 - Business Entity Resolution

Generates a 55-dimensional deterministic, country-agnostic feature vector:
  1. Multi-dimensional string similarities on name and address (Levenshtein, token sort, token set, partial, Jaro-Winkler)
  2. Brand / first token alignment signals (first token exact, ratio, Jaro-Winkler)
  3. Token overlap and inclusion coefficients (subset matching)
  4. Character 2-gram and 3-gram Jaccard overlaps (fine-grained typo & abbreviation tolerance)
  5. Numeric & digit alignment in business name (e.g. 7-Eleven, Studio 54)
  6. Legal suffix agreement flags
  7. Hierarchical postal code matching (exact, prefix-3 metro, prefix-2 region)
  8. Street number agreement and proximity signals
  9. Non-linear composite interactions (harmonic mean, weakest-link minimum, product, disagreement penalty)
"""

import re
import math
from typing import Dict, List, Any, Set
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

RE_WORD = re.compile(r"\w+")
RE_DIGITS = re.compile(r"\d+")

def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    return float(intersection_len / union_len)

def overlap_coefficient(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return float(len(set_a.intersection(set_b)) / min(len(set_a), len(set_b)))

def extract_first_token(text: str) -> str:
    """Extract first alphanumeric word token."""
    if not text:
        return ""
    words = RE_WORD.findall(text)
    return words[0].lower() if words else ""

def extract_digits(text: str) -> str:
    """Extract concatenated digits from text."""
    if not text:
        return ""
    digits = RE_DIGITS.findall(text)
    return "".join(digits)

def compute_pair_features(rec1: Any, rec2: Any) -> List[float]:
    """
    Extract a 55-dimensional deterministic, country-agnostic pairwise feature vector.
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
    njw = float(JaroWinkler.similarity(n1, n2) * 100.0)

    rnr = float(fuzz.ratio(rn1, rn2))
    rnjw = float(JaroWinkler.similarity(rn1, rn2) * 100.0)
    
    # Transliteration similarity (bridges Indic scripts and French diacritics)
    ascii_name_sort = float(fuzz.token_sort_ratio(an1, an2))
    ascii_root_sort = float(fuzz.token_sort_ratio(arn1, arn2))
    ascii_root_exact = 1.0 if (arn1 and arn1 == arn2) else 0.0

    # Brand / First Token Signals
    ft1 = extract_first_token(rn1 or n1)
    ft2 = extract_first_token(rn2 or n2)
    if ft1 and ft2:
        ft_exact = 1.0 if ft1 == ft2 else 0.0
        ft_ratio = float(fuzz.ratio(ft1, ft2))
        ft_jw = float(JaroWinkler.similarity(ft1, ft2) * 100.0)
    else:
        ft_exact = 0.0
        ft_ratio = 0.0
        ft_jw = 0.0

    # Token sets
    n1_words = set(w.lower() for w in RE_WORD.findall(n1))
    n2_words = set(w.lower() for w in RE_WORD.findall(n2))
    name_token_overlap = overlap_coefficient(n1_words, n2_words)
    name_token_jac = jaccard_similarity(n1_words, n2_words)

    # Character n-grams
    n1_2g = char_ngrams(n1, 2)
    n2_2g = char_ngrams(n2, 2)
    n_2g_jac = jaccard_similarity(n1_2g, n2_2g)

    n1_3g = char_ngrams(n1, 3)
    n2_3g = char_ngrams(n2, 3)
    n_3g_jac = jaccard_similarity(n1_3g, n2_3g)
    
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    root_exact = 1.0 if (rn1 and rn1 == rn2) else 0.0
    prefix_3 = 1.0 if (len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3]) else 0.0
    
    len_diff_n = float(abs(len(n1) - len(n2)))
    len_ratio_n = (min(len(n1), len(n2)) / max(len(n1), len(n2))) if (len(n1) > 0 and len(n2) > 0) else 0.0

    # Numeric & digit alignment in business names
    d1 = extract_digits(n1)
    d2 = extract_digits(n2)
    if d1 and d2:
        name_digits_exact = 1.0 if d1 == d2 else 0.0
        name_digits_mismatch = 1.0 if d1 != d2 else 0.0
        name_digits_signed = 1.0 if d1 == d2 else -1.0
    else:
        name_digits_exact = 0.0
        name_digits_mismatch = 0.0
        name_digits_signed = 0.0

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
    ajw = float(JaroWinkler.similarity(a1, a2) * 100.0)
    
    a1_words = set(w.lower() for w in RE_WORD.findall(a1))
    a2_words = set(w.lower() for w in RE_WORD.findall(a2))
    a_word_jac = jaccard_similarity(a1_words, a2_words)
    addr_token_overlap = overlap_coefficient(a1_words, a2_words)
    
    a1_2g = char_ngrams(a1, 2)
    a2_2g = char_ngrams(a2, 2)
    a_2g_jac = jaccard_similarity(a1_2g, a2_2g)

    a1_3g = char_ngrams(a1, 3)
    a2_3g = char_ngrams(a2, 3)
    a_3g_jac = jaccard_similarity(a1_3g, a2_3g)
    
    len_diff_a = float(abs(len(a1) - len(a2)))

    # -------------------------------------------------------------------------
    # 4. Structured Subfield Agreement Flags
    # -------------------------------------------------------------------------
    pc1 = str(rec1["postal_code"] or "").strip()
    pc2 = str(rec2["postal_code"] or "").strip()
    if pc1 and pc2:
        pc_exact = 1.0 if pc1 == pc2 else 0.0
        pc_mismatch = 1.0 if pc1 != pc2 else 0.0
        pc_signed = 1.0 if pc1 == pc2 else -1.0
        pc_prefix_3 = 1.0 if (len(pc1) >= 3 and len(pc2) >= 3 and pc1[:3] == pc2[:3]) else 0.0
        pc_prefix_2 = 1.0 if (len(pc1) >= 2 and len(pc2) >= 2 and pc1[:2] == pc2[:2]) else 0.0
    else:
        pc_exact = 0.0
        pc_mismatch = 0.0
        pc_signed = 0.0
        pc_prefix_3 = 0.0
        pc_prefix_2 = 0.0
        
    sn1 = str(rec1["street_num"] or "").strip()
    sn2 = str(rec2["street_num"] or "").strip()
    if sn1 and sn2:
        sn_exact = 1.0 if sn1 == sn2 else 0.0
        sn_mismatch = 1.0 if sn1 != sn2 else 0.0
        sn_signed = 1.0 if sn1 == sn2 else -1.0
        if sn1.isdigit() and sn2.isdigit():
            sn_diff_log = math.log1p(abs(int(sn1) - int(sn2)))
        else:
            sn_diff_log = 0.0 if sn1 == sn2 else 2.0
    else:
        sn_exact = 0.0
        sn_mismatch = 0.0
        sn_signed = 0.0
        sn_diff_log = 0.0

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
    name_addr_min = min(effective_name_sim, atset)
    name_addr_prod = (effective_name_sim * atset) / 10000.0
    name_addr_abs_diff = abs(effective_name_sim - atset)
    max_name_sim = max(nr, ntsort, ntset, ascii_name_sort)
    composite_match = (0.45 * ntset) + (0.45 * atset) + (10.0 * pc_exact)
    high_both = 1.0 if (effective_name_sim >= 80.0 and atset >= 80.0) else 0.0

    return [
        nr, npr, ntsort, ntset, njw, rnr, rnjw,
        ascii_name_sort, ascii_root_sort, ascii_root_exact,
        ft_exact, ft_ratio, ft_jw,
        name_token_overlap, name_token_jac,
        n_2g_jac, n_3g_jac,
        name_exact, root_exact, prefix_3, len_diff_n, len_ratio_n,
        name_digits_exact, name_digits_mismatch, name_digits_signed,
        both_suffix, suffix_match,
        ar, apr, atsort, atset, ajw, a_word_jac, addr_token_overlap,
        a_2g_jac, a_3g_jac, len_diff_a,
        pc_exact, pc_mismatch, pc_signed, pc_prefix_3, pc_prefix_2,
        sn_exact, sn_mismatch, sn_signed, sn_diff_log,
        lm_match, domain_match,
        harmonic, name_addr_min, name_addr_prod, name_addr_abs_diff,
        max_name_sim, composite_match, high_both
    ]

FEATURE_NAMES = [
    "name_ratio", "name_partial_ratio", "name_token_sort", "name_token_set", "name_jaro_winkler",
    "root_name_ratio", "root_name_jaro_winkler",
    "ascii_name_sort", "ascii_root_sort", "ascii_root_exact",
    "first_token_exact", "first_token_ratio", "first_token_jw",
    "name_token_overlap", "name_token_jaccard",
    "name_2g_jaccard", "name_3g_jaccard",
    "name_exact", "root_exact", "name_prefix_3", "name_len_diff", "name_len_ratio",
    "name_digits_exact", "name_digits_mismatch", "name_digits_signed",
    "both_have_legal_suffix", "legal_suffix_match",
    "addr_ratio", "addr_partial_ratio", "addr_token_sort", "addr_token_set", "addr_jaro_winkler",
    "addr_word_jaccard", "addr_token_overlap",
    "addr_2g_jaccard", "addr_3g_jaccard", "addr_len_diff",
    "postal_exact", "postal_mismatch", "postal_signed", "postal_prefix_3", "postal_prefix_2",
    "street_num_exact", "street_num_mismatch", "street_num_signed", "street_num_diff_log",
    "landmark_match", "domain_match",
    "harmonic_name_addr", "name_addr_min", "name_addr_prod", "name_addr_abs_diff",
    "max_name_sim", "composite_match_score", "high_both_sim"
]
