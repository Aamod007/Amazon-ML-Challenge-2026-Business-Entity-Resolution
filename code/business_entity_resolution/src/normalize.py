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
    from anyascii import anyascii
except ImportError:
    def anyascii(t: str) -> str:
        return t

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG

# Pre-compiled regular expressions for high throughput
RE_COMBINING = re.compile(r"[\u0300-\u036f]")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_AMP = re.compile(r"\s*&\s*")
RE_URL_PREFIX = re.compile(r"^https?://(?:www\.)?", re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:com|org|net|in|co|io|fr|gov|edu|info|biz)\b", re.IGNORECASE)
RE_NUM_LEADING_ZERO = re.compile(r"\b0+(\d+)\b")

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
    strips punctuation, handles web domains, and generates anyascii Latin transliterations.
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "norm_name": "",
            "root_name": "",
            "ascii_name": "",
            "ascii_root": "",
            "had_legal_suffix": False,
            "legal_suffix": ""
        }
    
    # 1. Unicode decomposition & lowercase
    t = normalize_unicode(raw_name)
    
    # Strip URL domains if present (e.g. maurewilliamscolombier.com -> maurewilliamscolombier)
    t = RE_URL_PREFIX.sub("", t)
    t = RE_DOMAIN_SUFFIX.sub("", t)

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
    
    # 6. Transliterate to clean Latin ASCII (unifies Indic scripts and French diacritics)
    ascii_raw = anyascii(raw_name).lower()
    ascii_raw = RE_URL_PREFIX.sub("", ascii_raw)
    ascii_raw = RE_DOMAIN_SUFFIX.sub("", ascii_raw)
    ascii_raw = RE_AMP.sub(" and ", ascii_raw)
    ascii_clean = RE_SPACES.sub(" ", RE_PUNCT.sub(" ", ascii_raw)).strip()
    ascii_root = RE_SPACES.sub(" ", RE_LEGAL_SUFFIX.sub("", ascii_clean)).strip()

    return {
        "norm_name": t_clean,
        "root_name": root_name,
        "ascii_name": ascii_clean,
        "ascii_root": ascii_root,
        "had_legal_suffix": had_legal_suffix,
        "legal_suffix": canonical_suffix
    }

def normalize_address(raw_address: str) -> Dict[str, Any]:
    """
    Produce normalized address representations while preserving raw original.
    Extracts structured subfields:
      - landmark ("Near X") separated out as weaker/noisier signal
      - postal / PIN code
      - street / plot / building number (with leading zero normalization, e.g. AF-0684 -> af-684)
      - residual normalized free-text address
      - anyascii transliterated address
    """
    if not isinstance(raw_address, str) or not raw_address.strip():
        return {
            "norm_address": "",
            "ascii_addr": "",
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
    
    # 3. Street / building number extraction with leading-zero normalization
    street_num_match = RE_STREET_NUM.search(t_no_landmark)
    if street_num_match:
        sn_raw = street_num_match.group(1).strip().lower()
        street_num = RE_NUM_LEADING_ZERO.sub(r"\1", sn_raw)
    else:
        street_num = ""
    
    # 4. Clean full normalized address and residual
    clean_norm_addr = RE_SPACES.sub(" ", RE_PUNCT.sub(" ", t)).strip()
    clean_residual = RE_SPACES.sub(" ", RE_PUNCT.sub(" ", t_no_landmark)).strip()
    
    # 5. Transliterated address
    ascii_addr = anyascii(clean_norm_addr).lower().strip()

    return {
        "norm_address": clean_norm_addr,
        "ascii_addr": ascii_addr,
        "landmark": landmark,
        "postal_code": postal_code,
        "street_num": street_num,
        "address_residual": clean_residual
    }

class Record:
    """
    Memory-efficient representation of an entity record using __slots__.
    Consumes ~120 bytes vs ~464 bytes for a standard Python dict (75% RAM reduction).
    Implements dict-like indexing (rec['field']) and .get() for seamless drop-in usage.
    """
    __slots__ = (
        "entity_id", "raw_name", "raw_addr", "country",
        "norm_name", "root_name", "ascii_name", "ascii_root",
        "had_legal_suffix", "legal_suffix",
        "norm_address", "ascii_addr", "landmark", "postal_code",
        "street_num", "address_residual"
    )
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def __getitem__(self, item: str):
        return getattr(self, item, "")
        
    def get(self, item: str, default: Any = "") -> Any:
        return getattr(self, item, default)

    def values(self):
        return [getattr(self, k, "") for k in self.__slots__]

    def keys(self):
        return self.__slots__

def normalize_record(record_tuple: tuple) -> Record:
    """
    Normalize an entity record tuple (entity_id, business_name, business_address, country).
    Returns a lightweight, memory-safe Record instance.
    """
    eid, raw_name, raw_addr, country = record_tuple[0], record_tuple[1], record_tuple[2], record_tuple[3]
    eid_str = str(eid).strip() if eid is not None else ""
    raw_name_str = str(raw_name).strip() if (raw_name is not None and not (isinstance(raw_name, float) and math.isnan(raw_name))) else ""
    raw_addr_str = str(raw_addr).strip() if (raw_addr is not None and not (isinstance(raw_addr, float) and math.isnan(raw_addr))) else ""
    country_str = str(country).strip() if (country is not None and not (isinstance(country, float) and math.isnan(country))) else ""

    name_info = normalize_name(raw_name_str)
    addr_info = normalize_address(raw_addr_str)
    return Record(
        entity_id=eid_str,
        raw_name=raw_name_str,
        raw_addr=raw_addr_str,
        country=country_str,
        **name_info,
        **addr_info
    )
