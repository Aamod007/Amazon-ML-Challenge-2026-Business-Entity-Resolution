"""
Stage 2: Normalization Module
Handles robust Unicode normalization, punctuation stripping, diacritic removal,
bidirectional legal-suffix canonicalization, and structured subfield extraction
(postal/PIN code, street number, landmarks, residual free-text address).
Preserves raw originals while producing normalized artifacts.
"""

import math
import re
import unicodedata
from typing import Dict, Any, List
try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG

# Pre-compiled regular expressions for high throughput
RE_COMBINING = re.compile(r"[\u0300-\u036f]")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_AMP = re.compile(r"\s*&\s*")

# Landmark pattern: captures phrases following landmark prepositions
RE_LANDMARK = re.compile(
    r"\b(?:near|opp\.?|opposite|behind|b/h|beside|adjacent(?:\s+to)?|next\s+to|in\s+front\s+of|close\s+to)\s+([^,;]+)",
    re.IGNORECASE
)

# Postal / PIN code pattern: 6 digits (India) or 5 digits (US/France)
RE_POSTAL = re.compile(r"\b([1-9]\d{5}|\d{5}(?:-\d{4})?)\b")

# Street number pattern (numbers preceded optionally by unit / building prefixes)
RE_STREET_NUM = re.compile(
    r"\b(?:(?:h\.?no\.?|house\s+no\.?|plot\s+no\.?|flat\s+no\.?|shop\s+no\.?|unit\s+no\.?|no\.?|#)\s*)?(\d+[-/]?\w*)\b",
    re.IGNORECASE
)

# Legal suffix regex built dynamically from configuration
_sorted_suffixes = sorted(CONFIG.legal_suffix_map.keys(), key=len, reverse=True)
RE_LEGAL_SUFFIX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _sorted_suffixes) + r")\b",
    re.IGNORECASE
)

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters, strip diacritics, and convert to lowercase."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = RE_COMBINING.sub("", text)
    return text.lower().strip()

def normalize_name(raw_name: str) -> Dict[str, Any]:
    """
    Produce normalized name representations while preserving raw original.
    Extracts legal suffix, sets 'had_legal_suffix' flag, expands/standardizes legal suffix,
    and strips punctuation.
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "norm_name": "",
            "root_name": "",
            "had_legal_suffix": False,
            "legal_suffix": ""
        }
    
    # 1. Unicode decomposition & lowercase
    t = normalize_unicode(raw_name)
    
    # 2. Check and extract legal suffix
    match = RE_LEGAL_SUFFIX.search(t)
    had_legal_suffix = bool(match)
    raw_suffix = match.group(0).lower() if match else ""
    canonical_suffix = CONFIG.legal_suffix_map.get(raw_suffix, raw_suffix)
    
    # 3. Standardize '&' to 'and'
    t = RE_AMP.sub(" and ", t)
    
    # 4. Punctuation stripping
    t_clean = RE_PUNCT.sub(" ", t)
    t_clean = RE_SPACES.sub(" ", t_clean).strip()
    
    # 5. Extract root name without legal suffix
    root_name = RE_LEGAL_SUFFIX.sub("", t_clean)
    root_name = RE_SPACES.sub(" ", root_name).strip()
    
    return {
        "norm_name": t_clean,
        "root_name": root_name,
        "had_legal_suffix": had_legal_suffix,
        "legal_suffix": canonical_suffix
    }

def normalize_address(raw_address: str) -> Dict[str, Any]:
    """
    Produce normalized address representations while preserving raw original.
    Extracts structured subfields:
      - landmark ("Near X") separated out as weaker/noisier signal
      - postal / PIN code
      - street / plot / building number
      - residual normalized free-text address
    """
    if not isinstance(raw_address, str) or not raw_address.strip():
        return {
            "norm_address": "",
            "landmark": "",
            "postal_code": "",
            "street_num": "",
            "address_residual": ""
        }
        
    t = normalize_unicode(raw_address)
    
    # 1. Landmark extraction
    landmark_match = RE_LANDMARK.search(t)
    landmark = landmark_match.group(1).strip() if landmark_match else ""
    
    # Remove landmark from address stream to leave clean residual
    t_no_landmark = RE_LANDMARK.sub(" ", t) if landmark_match else t
    
    # 2. Postal code extraction
    postal_match = RE_POSTAL.search(t_no_landmark)
    postal_code = postal_match.group(1).strip() if postal_match else ""
    
    # 3. Street / building number extraction
    street_num_match = RE_STREET_NUM.search(t_no_landmark)
    street_num = street_num_match.group(1).strip() if street_num_match else ""
    
    # 4. Clean full normalized address and residual
    clean_norm_addr = RE_SPACES.sub(" ", RE_PUNCT.sub(" ", t)).strip()
    clean_residual = RE_SPACES.sub(" ", RE_PUNCT.sub(" ", t_no_landmark)).strip()
    
    return {
        "norm_address": clean_norm_addr,
        "landmark": landmark,
        "postal_code": postal_code,
        "street_num": street_num,
        "address_residual": clean_residual
    }

def normalize_record(record_tuple: tuple) -> Dict[str, Any]:
    """
    Normalize an entity record tuple (entity_id, business_name, business_address, country).
    Handles None, NaN, and non-string inputs safely to prevent NoneType errors in downstream stages.
    """
    eid, raw_name, raw_addr, country = record_tuple[0], record_tuple[1], record_tuple[2], record_tuple[3]
    eid_str = str(eid).strip() if eid is not None else ""
    raw_name_str = str(raw_name).strip() if (raw_name is not None and not (isinstance(raw_name, float) and math.isnan(raw_name))) else ""
    raw_addr_str = str(raw_addr).strip() if (raw_addr is not None and not (isinstance(raw_addr, float) and math.isnan(raw_addr))) else ""
    country_str = str(country).strip() if (country is not None and not (isinstance(country, float) and math.isnan(country))) else ""

    name_info = normalize_name(raw_name_str)
    addr_info = normalize_address(raw_addr_str)
    return {
        "entity_id": eid_str,
        "raw_name": raw_name_str,
        "raw_addr": raw_addr_str,
        "country": country_str,
        **name_info,
        **addr_info
    }
