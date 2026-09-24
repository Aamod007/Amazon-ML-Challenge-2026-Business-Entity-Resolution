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
    """
    def __init__(self, country: str):
        self.country = country
        self.idx_name_token = defaultdict(list)
        self.idx_addr_num_word = defaultdict(list)
        self.idx_addr_pair = defaultdict(list)
        self.idx_prefix = defaultdict(list)
        self.addr_token_counts = Counter()

    def build(self, records: Dict[str, Dict[str, Any]]):
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
            raw_addr = rec.get("raw_addr") or ""
            raw_a_tokens = [w.lower() for w in RE_WORD.findall(raw_addr) if len(w) >= 2]
            a_tokens = [w for w in raw_a_tokens if w not in CONFIG.addr_stopwords]
            words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
            
            # Strategy 1: Name tokens
            for tok in n_tokens:
                if len(tok) >= CONFIG.min_token_len:
                    self.idx_name_token[tok].append(mid)

            # Strategy 2: Street number + first street word prefix
            street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
            clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]
            if clean_nums and words:
                s_num = clean_nums[0]
                for w in words[:3]:
                    self.idx_addr_num_word[(s_num, w[:4])].append(mid)

            # Strategy 3: Address distinctive token-pairs
            rare_words = sorted([w for w in words if len(w) >= 4], key=lambda w: self.addr_token_counts[w])
            if len(rare_words) >= 2:
                w1, w2 = sorted([rare_words[0], rare_words[1]])
                self.idx_addr_pair[(w1, w2)].append(mid)
                if len(rare_words) >= 3:
                    w1, w3 = sorted([rare_words[0], rare_words[2]])
                    self.idx_addr_pair[(w1, w3)].append(mid)

            # Strategy 4: Name prefix + address prefix
            if n_tokens and words:
                self.idx_prefix[(n_tokens[0][:4], words[0][:3])].append(mid)

    def query(self, rec: Dict[str, Any], max_candidates: int = 50) -> Set[str]:
        """
        Query candidate indices for a Source 1 entity and union candidate IDs across strategies.
        """
        n_tokens = extract_name_tokens(rec.get("raw_name") or "")
        raw_addr = rec.get("raw_addr") or ""
        raw_a_tokens = [w.lower() for w in RE_WORD.findall(raw_addr) if len(w) >= 2]
        a_tokens = [w for w in raw_a_tokens if w not in CONFIG.addr_stopwords]
        words = [tok for tok in a_tokens if not tok.isdigit() and len(tok) >= 3]
        street_nums = [tok for tok in raw_a_tokens if tok.isdigit() or (tok[:-1].isdigit() and tok[-1].isalpha())]
        clean_nums = [re.sub(r"[^\d]", "", tok) for tok in street_nums if re.sub(r"[^\d]", "", tok)]

        candidates = set()

        # Strategy 1: Name tokens (skip explosive high-frequency blocks)
        for tok in n_tokens:
            if len(tok) >= CONFIG.min_token_len:
                matches = self.idx_name_token.get(tok, [])
                if len(matches) <= CONFIG.max_block_token_frequency:
                    candidates.update(matches)

        # Strategy 2: Street number + word prefix
        if clean_nums and words:
            s_num = clean_nums[0]
            for w in words[:3]:
                candidates.update(self.idx_addr_num_word.get((s_num, w[:4]), []))

        # Strategy 3: Address distinctive pairs
        rare_words = sorted([w for w in words if len(w) >= 4], key=lambda w: self.addr_token_counts[w])
        if len(rare_words) >= 2:
            w1, w2 = sorted([rare_words[0], rare_words[1]])
            candidates.update(self.idx_addr_pair.get((w1, w2), []))
            if len(rare_words) >= 3:
                w1, w3 = sorted([rare_words[0], rare_words[2]])
                candidates.update(self.idx_addr_pair.get((w1, w3), []))

        # Strategy 4: Name prefix + locality prefix
        if n_tokens and words:
            candidates.update(self.idx_prefix.get((n_tokens[0][:4], words[0][:3]), []))

        # Cap candidates per entity to avoid runaway combinatorial evaluation
        if len(candidates) > max_candidates:
            # Keep a prioritized subset
            return set(list(candidates)[:max_candidates])
        return candidates
