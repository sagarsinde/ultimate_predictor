import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
try:
    import catboost as cb
except ImportError:
    cb = None
import warnings
warnings.filterwarnings('ignore')

def inject_dummies(X, y):
    """
    Ensure all 100 classes (0-99) exist in the training slice.
    """
    present = set(np.unique(y))
    missing = set(range(100)) - present
    sample_weights = np.ones(len(y))
    
    if missing:
        mean_row = X.mean(axis=0, keepdims=True)
        dummy_X = np.repeat(mean_row, len(missing), axis=0)
        dummy_y = np.array(sorted(missing), dtype=int)
        dummy_w = np.full(len(missing), 1e-6) # near zero weight

        X = np.vstack([X, dummy_X])
        y = np.concatenate([y, dummy_y])
        sample_weights = np.concatenate([sample_weights, dummy_w])
        
    return X, y, sample_weights

class XGBoostJodiModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        X, y, sample_weights = inject_dummies(X, y)

        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=100,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            tree_method='hist',
            device='cuda',
            random_state=42,
            verbosity=0
        )
        
        try:
            # Applying your exact Isotonic Calibration via TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            self.model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=tscv)
            self.model.fit(X, y, sample_weight=sample_weights)
        except Exception:
            # Fallback if dataset is too small for TimeSeriesSplit or CUDA fails
            self.model = base_model
            try:
                self.model.fit(X, y, sample_weight=sample_weights)
            except Exception:
                self.model.set_params(device='cpu')
                self.model.fit(X, y, sample_weight=sample_weights)

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(100)
        if hasattr(self.model, 'classes_'):
            for i, cls in enumerate(self.model.classes_):
                probs[int(cls)] = raw[i]
        else:
            probs = raw
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(100) / 100.0
        return probs

class CatBoostJodiModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        if cb is None:
            raise ImportError("CatBoost not installed")
        
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        X, y, sample_weights = inject_dummies(X, y)

        self.model = cb.CatBoostClassifier(
            iterations=150,
            depth=5,
            learning_rate=0.05,
            loss_function='MultiClass',
            verbose=0,
            random_seed=42,
            allow_writing_files=False,
            task_type='GPU'
        )
        try:
            self.model.fit(X, y, sample_weight=sample_weights)
        except Exception:
            self.model = cb.CatBoostClassifier(
                iterations=150, depth=5, learning_rate=0.05,
                loss_function='MultiClass', verbose=0, random_seed=42,
                allow_writing_files=False, task_type='CPU'
            )
            self.model.fit(X, y, sample_weight=sample_weights)

    def predict_proba(self, X):
        if cb is None:
            return np.ones(100) / 100.0
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(100)
        for i, cls in enumerate(self.model.classes_):
            probs[int(cls)] = raw[i]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(100) / 100.0
        return probs

MODEL_TYPES = {
    'xgb': XGBoostJodiModel,
    'cat': CatBoostJodiModel
}
