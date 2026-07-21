import numpy as np
import xgboost as xgb
from hybrid_1_1.pana_features import build_pana_features

class PanaTypeClassifier:
    """Predicts the probability of a Pana being SP (0), DP (1), or TP (2)."""
    
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=3,
            n_jobs=-1,
            random_state=42
        )
        self.is_trained = False
        
    def fit(self, pana_series):
        """Train the model on the historical sequence of Panas."""
        try:
            X, y, _ = build_pana_features(pana_series)
            if len(X) < 10:
                return  # Not enough data
                
            self.model.fit(X, y)
            self.is_trained = True
        except Exception:
            pass
            
    def predict_proba(self, pana_series) -> np.ndarray:
        """
        Returns probabilities for SP (idx 0), DP (idx 1), TP (idx 2)
        for the *next* draw.
        """
        if not self.is_trained:
            # Baseline historical probabilities (approximate)
            return np.array([0.72, 0.27, 0.01])
            
        try:
            _, _, last_row = build_pana_features(pana_series)
            probs = self.model.predict_proba(last_row)[0]
            
            # Ensure it returns exactly 3 classes even if TP was missing in training
            if len(probs) < 3:
                full_probs = np.zeros(3)
                # Map available probabilities to their correct classes
                # (XGBoost classes_ attribute tells us which indices match which classes)
                for i, c in enumerate(self.model.classes_):
                    if c < 3:
                        full_probs[c] = probs[i]
                return full_probs
                
            return probs
        except Exception:
            return np.array([0.72, 0.27, 0.01])
