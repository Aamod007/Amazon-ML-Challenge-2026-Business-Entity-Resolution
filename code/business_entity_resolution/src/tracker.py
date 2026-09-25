"""
Stage 5 Tracking: Experiment Tracker Module
Tracks every training experiment and validation benchmark in an append-only CSV spreadsheet
and JSON log, adhering strictly to competitive ML best practices (Point 5 on slide).
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

class ExperimentTracker:
    """
    Automated experiment tracking system that logs all training runs, cross-validation metrics,
    hyperparameters, and ensemble configurations to:
      1. experiments/experiment_tracker.csv (human-readable spreadsheet compatible with Excel/Sheets)
      2. experiments/experiment_log.json (deep serialization for full reproducibility)
    """
    CSV_COLUMNS = [
        "experiment_id",
        "timestamp",
        "model_architecture",
        "n_train_s1",
        "n_folds",
        "n_features",
        "cv_mean_macro_f05",
        "cv_std_macro_f05",
        "optimal_threshold",
        "singleton_f05",
        "non_singleton_f05",
        "fold_scores",
        "imbalance_strategy",
        "scale_pos_weight",
        "ensemble_weights",
        "duration_seconds",
        "notes"
    ]

    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "experiment_tracker.csv")
        self.json_path = os.path.join(self.log_dir, "experiment_log.json")
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)

    def _get_next_experiment_id(self) -> str:
        """Generate incremental experiment ID: EXP-001, EXP-002, etc."""
        if not os.path.exists(self.csv_path):
            return "EXP-001"
        try:
            df = pd.read_csv(self.csv_path)
            if len(df) == 0:
                return "EXP-001"
            last_id = str(df["experiment_id"].iloc[-1])
            if last_id.startswith("EXP-"):
                num = int(last_id.split("-")[1])
                return f"EXP-{num + 1:03d}"
            return f"EXP-{len(df) + 1:03d}"
        except Exception:
            return f"EXP-{int(time.time())}"

    def log(
        self,
        model_architecture: str,
        n_train_s1: int,
        n_folds: int,
        n_features: int,
        cv_mean_macro_f05: float,
        cv_std_macro_f05: float,
        optimal_threshold: float,
        fold_scores: List[float],
        singleton_f05: Optional[float] = None,
        non_singleton_f05: Optional[float] = None,
        imbalance_strategy: str = "sqrt_ratio",
        scale_pos_weight: float = 1.0,
        ensemble_weights: Optional[Dict[str, float]] = None,
        duration_seconds: float = 0.0,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_importances: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> str:
        """Log a complete experiment run to both CSV spreadsheet and JSON log."""
        exp_id = self._get_next_experiment_id()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fold_scores_str = ",".join(f"{s:.5f}" for s in fold_scores)
        ens_str = json.dumps(ensemble_weights) if ensemble_weights else "N/A"

        row = [
            exp_id,
            timestamp,
            model_architecture,
            n_train_s1,
            n_folds,
            n_features,
            f"{cv_mean_macro_f05:.5f}",
            f"{cv_std_macro_f05:.5f}",
            f"{optimal_threshold:.3f}",
            f"{singleton_f05:.5f}" if singleton_f05 is not None else "N/A",
            f"{non_singleton_f05:.5f}" if non_singleton_f05 is not None else "N/A",
            fold_scores_str,
            imbalance_strategy,
            f"{scale_pos_weight:.2f}",
            ens_str,
            f"{duration_seconds:.1f}",
            notes
        ]

        # 1. Append to CSV spreadsheet
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 2. Append to JSON log
        json_entry = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "model_architecture": model_architecture,
            "n_train_s1": n_train_s1,
            "n_folds": n_folds,
            "n_features": n_features,
            "cv_mean_macro_f05": cv_mean_macro_f05,
            "cv_std_macro_f05": cv_std_macro_f05,
            "optimal_threshold": optimal_threshold,
            "fold_scores": fold_scores,
            "singleton_f05": singleton_f05,
            "non_singleton_f05": non_singleton_f05,
            "imbalance_strategy": imbalance_strategy,
            "scale_pos_weight": scale_pos_weight,
            "ensemble_weights": ensemble_weights,
            "duration_seconds": duration_seconds,
            "hyperparameters": hyperparameters or {},
            "feature_importances": feature_importances or {},
            "notes": notes
        }

        all_logs = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    all_logs = json.load(f)
            except Exception:
                all_logs = []

        all_logs.append(json_entry)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, indent=2)

        print(f"\n[ExperimentTracker] Logged run {exp_id} -> {self.csv_path}")
        return exp_id

    def print_summary(self):
        """Print formatted summary table of all recorded experiments."""
        if not os.path.exists(self.csv_path):
            print("No experiments logged yet.")
            return

        df = pd.read_csv(self.csv_path)
        if len(df) == 0:
            print("No experiments logged yet.")
            return

        print("\n" + "=" * 110)
        print("EXPERIMENT TRACKING SCOREBOARD (Spreadsheet: experiments/experiment_tracker.csv)")
        print("=" * 110)
        display_cols = [
            "experiment_id", "timestamp", "model_architecture", "n_features",
            "cv_mean_macro_f05", "cv_std_macro_f05", "optimal_threshold", "notes"
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
        print("=" * 110 + "\n")
