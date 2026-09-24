"""
Evaluation and Metrics Module
Implements the exact competition metric: macro-averaged per-entity F_0.5 score
including singleton handling (1.0 for correctly predicted empty list, 0.0 otherwise).
Also provides threshold sweeping utilities.
"""

from typing import Dict, Set, List, Tuple
import numpy as np

def compute_entity_f05(y_true: Set[str], y_pred: Set[str]) -> float:
    """
    Compute F_0.5 score for an individual Source 1 entity.
    Singletons:
      - If true is empty and pred is empty -> 1.0
      - If true is empty and pred non-empty -> 0.0
    Non-singletons:
      - If true is non-empty and pred is empty -> 0.0
      - Otherwise:
        P = TP / |pred|
        R = TP / |true|
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if not y_true:  # Ground truth singleton
        return 1.0 if not y_pred else 0.0
    
    if not y_pred:  # Missed match on non-singleton
        return 0.0
    
    tp = len(y_true.intersection(y_pred))
    if tp == 0:
        return 0.0
    
    prec = tp / len(y_pred)
    rec = tp / len(y_true)
    return float((1.25 * prec * rec) / (0.25 * prec + rec))

def evaluate_macro_f05(
    y_true_dict: Dict[str, Set[str]],
    y_pred_dict: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> float:
    """
    Compute macro-averaged F_0.5 score across all Source 1 entities in the evaluation set.
    """
    scores = [
        compute_entity_f05(y_true_dict.get(s1, set()), y_pred_dict.get(s1, set()))
        for s1 in all_s1_ids
    ]
    return float(np.mean(scores))

def sweep_optimal_threshold(
    s1_cand_probs: Dict[str, List[Tuple[str, float]]],
    y_true_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    threshold_range: np.ndarray = np.linspace(0.80, 0.98, 19)
) -> Tuple[float, float]:
    """
    Sweep decision thresholds on validation predictions to maximize macro F_0.5.
    Returns (best_threshold, best_macro_f05).
    """
    best_thresh = 0.90
    best_score = -1.0
    
    for thresh in threshold_range:
        preds = {}
        for s1_id in all_s1_ids:
            cands = s1_cand_probs.get(s1_id, [])
            preds[s1_id] = set(mid for mid, prob in cands if prob >= thresh)
            
        score = evaluate_macro_f05(y_true_dict, preds, all_s1_ids)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            
    return best_thresh, best_score
