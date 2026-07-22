import numpy as np
import xgboost as xgb
from panas.features import build_pana_features, build_deep_pana_features
from panas.universe import get_panas_for_digit

class PanaTypeClassifier:
    """Legacy model for backward compatibility with hybrid_1_1."""
    
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
        try:
            X, y, _ = build_pana_features(pana_series)
            if len(X) < 10: return
            self.model.fit(X, y)
            self.is_trained = True
        except Exception:
            pass
            
    def predict_proba(self, pana_series) -> np.ndarray:
        if not self.is_trained: return np.array([0.72, 0.27, 0.01])
        try:
            _, _, last_row = build_pana_features(pana_series)
            probs = self.model.predict_proba(last_row)[0]
            if len(probs) < 3:
                full_probs = np.zeros(3)
                for i, c in enumerate(self.model.classes_):
                    if c < 3: full_probs[c] = probs[i]
                return full_probs
            return probs
        except Exception:
            return np.array([0.72, 0.27, 0.01])


class HybridPanaPredictor:
    """
    Advanced model for daily Pana predictions.
    Utilizes a cascading architecture (Digit + Type) integrated with 
    Astrological fallback/human fantasy metrics for the 220 matrix.
    """
    def __init__(self):
        self.digit_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            objective='multi:softprob', num_class=10, n_jobs=-1, random_state=42
        )
        self.type_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            objective='multi:softprob', num_class=3, n_jobs=-1, random_state=42
        )
        self.is_trained = False
        
    def save_models(self, dir_path="trained_models"):
        """Saves the trained XGBoost models to disk."""
        import os
        if not self.is_trained:
            print("Warning: Models are not trained yet. Nothing to save.")
            return
            
        os.makedirs(dir_path, exist_ok=True)
        digit_path = os.path.join(dir_path, "pana_digit_model.json")
        type_path = os.path.join(dir_path, "pana_type_model.json")
        
        self.digit_model.save_model(digit_path)
        self.type_model.save_model(type_path)
        print(f"Models successfully saved to {dir_path}/")
        
    def load_models(self, dir_path="trained_models"):
        """Loads pre-trained XGBoost models from disk."""
        import os
        digit_path = os.path.join(dir_path, "pana_digit_model.json")
        type_path = os.path.join(dir_path, "pana_type_model.json")
        
        if os.path.exists(digit_path) and os.path.exists(type_path):
            self.digit_model.load_model(digit_path)
            self.type_model.load_model(type_path)
            self.is_trained = True
            print(f"Models successfully loaded from {dir_path}/")
        else:
            print(f"Error: Model files not found in {dir_path}/")

    def fit(self, df, is_morning=True):
        X, y_digit, y_type, _ = build_deep_pana_features(df, is_morning=is_morning)
        if len(X) < 10: return
        
        # Train independently
        self.digit_model.fit(X, y_digit)
        self.type_model.fit(X, y_type)
        self.is_trained = True
        
    def predict_220(self, df, is_morning=True, top_n=15):
        """
        Returns a sorted list of the top predicted 3-digit Panas for the next draw.
        """
        if not self.is_trained:
            return []
            
        _, _, _, last_row = build_deep_pana_features(df, is_morning=is_morning)
        
        # 1. Get raw probabilities
        digit_probs = self.digit_model.predict_proba(last_row)[0]
        type_probs = self.type_model.predict_proba(last_row)[0]
        
        # Normalize in case classes were missing during training
        d_p = np.zeros(10)
        for i, c in enumerate(self.digit_model.classes_): d_p[c] = digit_probs[i]
        
        t_p = np.zeros(3)
        for i, c in enumerate(self.type_model.classes_): t_p[c] = type_probs[i]
        
        type_names = {0: 'SP', 1: 'DP', 2: 'TP'}
        
        # 2. Cross-reference the Matrix
        pana_scores = []
        for digit in range(10):
            prob_d = d_p[digit]
            for t_idx in range(3):
                t_name = type_names[t_idx]
                prob_t = t_p[t_idx]
                
                # Retrieve the theoretical panas for this combination (e.g. Digit 9 + SP)
                valid_panas = get_panas_for_digit(digit, t_name)
                if not valid_panas: continue
                
                # Split probability across the valid panas uniformly
                individual_prob = (prob_d * prob_t) / len(valid_panas)
                
                for p_str in valid_panas:
                    pana_scores.append({
                        'Pana': p_str,
                        'Digit': digit,
                        'Type': t_name,
                        'Probability': individual_prob * 100
                    })
                    
        # Sort descending
        pana_scores.sort(key=lambda x: x['Probability'], reverse=True)
        return pana_scores[:top_n]
