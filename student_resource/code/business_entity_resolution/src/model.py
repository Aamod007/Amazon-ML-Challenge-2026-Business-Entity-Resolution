"""
Stage 5: Matching Model Module
Trains and executes the pairwise matching classifier using XGBoost (Apache 2.0 License).
Uses grouped cross-validation keyed by Source 1 entity, class imbalance weighting,
feature importance tracking, and model serialization.
"""

import os
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
try:
    from .config import CONFIG
    from .features import FEATURE_NAMES
except ImportError:
    from config import CONFIG
    from features import FEATURE_NAMES

class MatchingClassifier:
    """
    Pairwise business entity matching classifier wrapper.
    License: Apache 2.0 (XGBoost)
    Total parameters: < 50,000 trees / nodes (well within the <= 8B parameter constraint).
    """
    def __init__(self, config=CONFIG):
        self.config = config
        self.model = None
        self.feature_names = FEATURE_NAMES
        self.feature_importances_ = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: List[str],
        n_splits: int = 3
    ) -> Dict[str, Any]:
        """
        Train using GroupKFold keyed by Source 1 entity to eliminate fold data leakage.
        """
        pos_count = int(np.sum(y))
        neg_count = len(y) - pos_count
        scale_weight = float(neg_count / max(pos_count, 1))
        
        print(f"Dataset Shape: {X.shape}, Positives: {pos_count:,} ({pos_count/len(y)*100:.2f}%)")
        print(f"Calculated scale_pos_weight: {scale_weight:.2f}")

        gkf = GroupKFold(n_splits=n_splits)
        fold_models = []
        oof_probs = np.zeros(len(y), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            clf = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                scale_pos_weight=scale_weight,
                random_state=self.config.random_state + fold,
                n_jobs=self.config.n_jobs,
                tree_method=self.config.tree_method,
                device=self.config.device,
                eval_metric="logloss"
            )
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            oof_probs[val_idx] = probs
            fold_models.append(clf)
            print(f"Fold {fold} training completed.")

        # Train final production model on full training candidate dataset
        print("Training final production model on full training pairs...")
        final_model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            scale_pos_weight=scale_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            tree_method=self.config.tree_method,
            device=self.config.device,
            eval_metric="logloss"
        )
        final_model.fit(X, y)
        self.model = final_model
        self.feature_importances_ = final_model.feature_importances_

        return {
            "oof_probs": oof_probs,
            "feature_importances": dict(zip(self.feature_names, self.feature_importances_.tolist()))
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities for candidate feature matrix."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def save(self, model_path: str):
        """Save trained model and metadata."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        meta_path = model_path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "license": "Apache-2.0",
                "framework": "XGBoost",
                "model_type": "GradientBoostedTrees",
                "features": self.feature_names,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "feature_importances": dict(zip(self.feature_names, self.feature_importances_.tolist())) if self.feature_importances_ is not None else {}
            }, f, indent=2)

    def load(self, model_path: str):
        """Load trained model."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
