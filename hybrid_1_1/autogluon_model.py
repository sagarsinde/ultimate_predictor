import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    TabularPredictor = None

class AutoGluonModel:
    """
    AutoGluon TabularPredictor wrapper for the hybrid_1_1 ensemble.
    Automates feature engineering, hyperparameter tuning, and ensembling.
    """
    def __init__(self):
        self.predictor = None
        self.label_col = 'target'
        
    def fit(self, X, y, sequence=None, dow_sequence=None):
        if TabularPredictor is None:
            return
            
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        # Ensure we have a 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Convert to DataFrame as AutoGluon requires tabular structure
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        df[self.label_col] = y
        
        # Train AutoGluon, limit time to avoid hanging in backtests.
        # Since this evaluates multiple times in walk-forward, we use a 30s limit per window.
        try:
            self.predictor = TabularPredictor(label=self.label_col, verbosity=0).fit(
                train_data=df, 
                time_limit=30,
                presets='good_quality'
            )
        except Exception:
            self.predictor = None
        
    def predict_proba(self, X, last_digits=None, current_dow=None):
        if self.predictor is None:
            return np.ones(10) / 10.0
            
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        
        try:
            # predict_proba returns a DataFrame of class probabilities for multi-class
            probs_df = self.predictor.predict_proba(df)
            
            # Extract probabilities, ensuring all 10 classes are represented
            probs = np.zeros(10)
            for i in range(10):
                if i in probs_df.columns:
                    probs[i] = probs_df[i].iloc[0]
                    
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(10) / 10.0
                
            return probs
        except Exception:
            return np.ones(10) / 10.0
