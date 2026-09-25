"""
Stage 3: Blocking / Candidate Generation Module
Implements multiple independent blocking strategies targeting each noise pattern:
  1. Distinctive name token inverted index (reordering, missing tokens, abbreviations)
  2. Street number + street word prefix (handles cross-script transliteration & DBA names)
  3. Address token co-occurrence pairs (handles unstructured / missing number addresses)
  4. Phonetic / Prefix name + locality prefix (handles typos, character transpositions)
Enforces open-string country partitioning with zero cross-country candidate generation.
"""

import re
from collections import defaultdict, Counter
try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG

RE_WORD = re.compile(r"\w+")

def extract_name_tokens(norm_name: str) -> List[str]:
    """Extract significant name tokens, omitting common generic entity stopwords."""
    if not norm_name or not isinstance(norm_name, str):
        return []
    words = [w.lower() for w in RE_WORD.findall(norm_name) if len(w) >= 2]
    return [w for w in words if w not in CONFIG.name_stopwords]

def extract_addr_tokens(norm_address: str) -> List[str]:
    """Extract significant address tokens, omitting common street type stopwords."""
    if not norm_address or not isinstance(norm_address, str):
        return []
    words = [w.lower() for w in RE_WORD.findall(norm_address) if len(w) >= 2]
    return [w for w in words if w not in CONFIG.addr_stopwords]

class CountryCandidateIndex:
    """
    Candidate blocking index constructed over Source 2 and Source 3 records for a single country.
    Supports multi-script ASCII transliteration blocking and bounded inverted index pruning.
    """
    def __init__(self, country: str, max_freq: int = None):
        self.country = country
        self.max_freq = max_freq or CONFIG.max_block_token_frequency
        self.idx_name_token = defaultdict(list)
        self.idx_ascii_token = defaultdict(list)
        self.idx_addr_num_word = defaultdict(list)
        self.idx_addr_pair = defaultdict(list)
        self.idx_prefix = defaultdict(list)
        self.idx_postal_num = defaultdict(list)
        self.idx_name_digits = defaultdict(list)
        self.addr_token_counts = Counter()

    def build(self, records: Dict[str, Any]):
        """Build multi-strategy indices over target candidate records."""
        # Pass 1: compute address token frequency to identify rare distinctive tokens
        for rec in records.values():
            a_tokens = extract_addr_tokens(rec.get("norm_address") or "")
            for t in set(a_tokens):
                if len(t) >= 4 and not t.isdigit():
                    self.addr_token_counts[t] += 1

        # Pass 2: populate inverted indices
        for mid, rec in records.items():
            n_tokens = extract_name_tokens(rec.get("raw_name") or "")
            ascii_tokens = extract_name_tokens(rec.get("ascii_name") or "")
            raw_addr = rec.get("raw_addr") or ""
            raw_a_tokens = [w.lower() for w in RE_WORD.findall(raw_addr) if len(w) >= 2]
            a_tokens = [w for w in raw_a_tokens if w not in CONFIG.addr_stopwords]
            words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
            
            # Strategy 1: Raw Name tokens
            for tok in n_tokens:
                if len(tok) >= CONFIG.min_token_len:
                    self.idx_name_token[tok].append(mid)

            # Strategy 1b: Transliterated ASCII Name tokens (cross-script bridge)
            for tok in ascii_tokens:
                if len(tok) >= CONFIG.min_token_len and tok not in n_tokens:
                    self.idx_ascii_token[tok].append(mid)

            # Strategy 1c: Distinctive Name Digits (e.g. Local 579, Studio 54)
            name_d = "".join(re.findall(r"\d+", rec.get("raw_name") or ""))
            if len(name_d) >= 2:
                self.idx_name_digits[name_d].append(mid)

            # Strategy 2: Street number + first street word prefix
            s_num = str(rec.get("street_num") or "").strip()
            if not s_num:
                street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
                clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]
                if clean_nums:
                    s_num = clean_nums[0]
            if s_num and words:
                for w in words[:2]:
                    self.idx_addr_num_word[(s_num, w[:4])].append(mid)

            # Strategy 2b: Postal code + street number (exact location anchor)
            p_code = str(rec.get("postal_code") or "").strip()
            if p_code and s_num:
                self.idx_postal_num[(p_code, s_num)].append(mid)

            # Strategy 3: Address distinctive token-pairs
            rare_words = sorted([w for w in words if len(w) >= 4], key=lambda w: self.addr_token_counts[w])
            if len(rare_words) >= 2:
                w1, w2 = sorted([rare_words[0], rare_words[1]])
                self.idx_addr_pair[(w1, w2)].append(mid)
                if len(rare_words) >= 3:
                    w1, w3 = sorted([rare_words[0], rare_words[2]])
                    self.idx_addr_pair[(w1, w3)].append(mid)

            # Strategy 4: Name prefix + address prefix
            if (n_tokens or ascii_tokens) and words:
                first_name_tok = (ascii_tokens or n_tokens)[0]
                self.idx_prefix[(first_name_tok[:4], words[0][:3])].append(mid)

    def query(self, rec: Any, max_candidates: int = None) -> Set[str]:
        """
        Query candidate indices with multi-strategy TF-IDF weighted ranking.
        Prioritizes candidates with high multi-channel agreement and rare distinctive tokens.
        """
        max_cands = max_candidates or CONFIG.max_candidates_per_entity
        n_tokens = extract_name_tokens(rec.get("raw_name") or "")
        ascii_tokens = extract_name_tokens(rec.get("ascii_name") or "")
        raw_addr = rec.get("raw_addr") or ""
        raw_a_tokens = [w.lower() for w in RE_WORD.findall(raw_addr) if len(w) >= 2]
        a_tokens = [w for w in raw_a_tokens if w not in CONFIG.addr_stopwords]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        
        s_num = str(rec.get("street_num") or "").strip()
        if not s_num:
            street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
            clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]
            if clean_nums:
                s_num = clean_nums[0]
                
        p_code = str(rec.get("postal_code") or "").strip()
        cand_scores = Counter()

        # Strategy 1: Raw Name tokens (inverse-frequency weighted)
        for tok in n_tokens:
            if len(tok) >= CONFIG.min_token_len:
                matches = self.idx_name_token.get(tok, [])
                if len(matches) <= self.max_freq:
                    w = 12.0 / (len(matches) + 1.0)
                    for mid in matches:
                        cand_scores[mid] += w

        # Strategy 1b: Transliterated ASCII tokens
        for tok in ascii_tokens:
            if len(tok) >= CONFIG.min_token_len:
                matches = self.idx_ascii_token.get(tok, [])
                if len(matches) <= self.max_freq:
                    w = 10.0 / (len(matches) + 1.0)
                    for mid in matches:
                        cand_scores[mid] += w

        # Strategy 1c: Distinctive Name Digits (e.g. Local 579)
        name_d = "".join(re.findall(r"\d+", rec.get("raw_name") or ""))
        if len(name_d) >= 2:
            matches = self.idx_name_digits.get(name_d, [])
            if len(matches) <= self.max_freq:
                w = 15.0 / (len(matches) + 1.0)
                for mid in matches:
                    cand_scores[mid] += w

        # Strategy 2: Street number + word prefix
        if s_num and words:
            for w in words[:2]:
                matches = self.idx_addr_num_word.get((s_num, w[:4]), [])
                if len(matches) <= self.max_freq:
                    w = 15.0 / (len(matches) + 1.0)
                    for mid in matches:
                        cand_scores[mid] += w

        # Strategy 2b: Postal code + Street number (highest location precision)
        if p_code and s_num:
            matches = self.idx_postal_num.get((p_code, s_num), [])
            if len(matches) <= self.max_freq:
                w = 20.0 / (len(matches) + 1.0)
                for mid in matches:
                    cand_scores[mid] += w

        # Strategy 3: Address distinctive pairs
        rare_words = sorted([w for w in words if len(w) >= 4], key=lambda w: self.addr_token_counts[w])
        if len(rare_words) >= 2:
            w1, w2 = sorted([rare_words[0], rare_words[1]])
            matches = self.idx_addr_pair.get((w1, w2), [])
            if len(matches) <= self.max_freq:
                w = 12.0 / (len(matches) + 1.0)
                for mid in matches:
                    cand_scores[mid] += w
            if len(rare_words) >= 3:
                w1, w3 = sorted([rare_words[0], rare_words[2]])
                matches = self.idx_addr_pair.get((w1, w3), [])
                if len(matches) <= self.max_freq:
                    w = 10.0 / (len(matches) + 1.0)
                    for mid in matches:
                        cand_scores[mid] += w

        # Strategy 4: Name prefix + locality prefix
        if (n_tokens or ascii_tokens) and words:
            first_name_tok = (ascii_tokens or n_tokens)[0]
            matches = self.idx_prefix.get((first_name_tok[:4], words[0][:3]), [])
            if len(matches) <= self.max_freq:
                w = 8.0 / (len(matches) + 1.0)
                for mid in matches:
                    cand_scores[mid] += w

        # Return top candidates ranked by composite strategy score
        if len(cand_scores) <= max_cands:
            return set(cand_scores.keys())
        return set(mid for mid, _ in cand_scores.most_common(max_cands))

