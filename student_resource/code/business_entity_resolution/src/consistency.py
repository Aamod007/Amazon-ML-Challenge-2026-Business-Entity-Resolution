"""
Stage 7: Global Consistency Post-Processing Module
Resolves multi-match assignment conflicts:
Because the ground-truth entity resolution domain exhibits strict 1-to-many cardinality
(an S2 or S3 fragment belongs to at most one real-world business entity), any S2/S3 record
predicted for multiple S1 entities is resolved to its single highest-probability S1 match.
"""

from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

def resolve_global_consistency(
    s1_cand_probs: Dict[str, List[Tuple[str, float]]],
    threshold: float
) -> Tuple[Dict[str, Set[str]], int]:
    """
    Apply global consistency conflict resolution:
    Each candidate S2/S3 entity is assigned strictly to the single Source 1 entity
    that produced the maximum predicted probability >= threshold.
    Returns:
      - resolved_predictions: Dict[s1_id -> set of matched_entity_ids]
      - conflict_count: number of S2/S3 entities that had multi-S1 conflicts before resolution
    """
    # Step 1: Track best S1 match per candidate S2/S3 entity
    s2_best_s1 = {}  # mid -> (best_s1, max_prob)
    conflict_tracker = Counter()
    
    for s1_id, cand_list in s1_cand_probs.items():
        for mid, prob in cand_list:
            if prob >= threshold:
                conflict_tracker[mid] += 1
                if mid not in s2_best_s1 or prob > s2_best_s1[mid][1]:
                    s2_best_s1[mid] = (s1_id, prob)
                    
    conflict_count = sum(1 for mid, cnt in conflict_tracker.items() if cnt > 1)
    
    # Step 2: Assemble resolved predictions grouped by S1
    resolved_predictions = defaultdict(set)
    for mid, (best_s1, prob) in s2_best_s1.items():
        resolved_predictions[best_s1].add(mid)
        
    return resolved_predictions, conflict_count
