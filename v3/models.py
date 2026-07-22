"""
Standardized Unified Models Registry for v2, v3, and hybrid_1_1
"""

import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

# Fallback import for AutoGluon if it exists
try:
    from hybrid_1_1.autogluon_model import AutoGluonModel
except ImportError:
    AutoGluonModel = None


# --- HELPER FUNCS ---
def get_path(dir_path, name, ext):
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"{name}.{ext}")


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

class XGBoostModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y, sequence=None):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        present = set(np.unique(y))
        missing = set(range(10)) - present
        sample_weights = np.ones(len(y))

        if missing:
            mean_row = X.mean(axis=0, keepdims=True)
            dummy_X = np.repeat(mean_row, len(missing), axis=0)
            dummy_y = np.array(sorted(missing), dtype=int)
            dummy_w = np.full(len(missing), 1e-6)
            X = np.vstack([X, dummy_X])
            y = np.concatenate([y, dummy_y])
            sample_weights = np.concatenate([sample_weights, dummy_w])

        base = xgb.XGBClassifier(
            objective='multi:softprob', num_class=10, eval_metric='mlogloss',
            max_depth=5, learning_rate=0.04, n_estimators=200, tree_method='hist',
            random_state=42, verbosity=0
        )
        
        n_splits = min(5, max(2, len(X) // 10))
        min_class_count = min(np.bincount(y, minlength=10))

        if len(X) < 50 or min_class_count < n_splits:
            base.fit(X, y, sample_weight=sample_weights)
            self.model = base
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            try:
                calibrated = CalibratedClassifierCV(estimator=base, method='isotonic', cv=tscv)
                calibrated.fit(X, y, sample_weight=sample_weights)
                self.model = calibrated
            except Exception:
                base.fit(X, y, sample_weight=sample_weights)
                self.model = base

    def predict_proba(self, X, last_digits=None, current_dow=None):
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(10)
        classes = getattr(self.model, 'classes_', np.arange(10))
        for i, cls in enumerate(classes):
            probs[int(cls)] = raw[i]
        
        total = probs.sum()
        return probs / total if total > 0 else np.ones(10) / 10.0

    def save_models(self, dir_path, name):
        # Using joblib for XGBoost to cleanly support CalibratedClassifierCV wrapper
        joblib.dump(self.model, get_path(dir_path, name, "pkl"))

    def load_models(self, dir_path, name):
        self.model = joblib.load(get_path(dir_path, name, "pkl"))


class LightGBMModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y, sequence=None):
        if lgb is None: return
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            objective='multiclass', num_class=10, random_state=42, n_jobs=-1, verbose=-1
        )
        self.model.fit(X, y)

    def predict_proba(self, X, last_digits=None, current_dow=None):
        if self.model is None: return np.ones(10) / 10.0
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(10)
        classes = getattr(self.model, 'classes_', np.arange(10))
        for i, cls in enumerate(classes):
            probs[int(cls)] = raw[i]
        total = probs.sum()
        return probs / total if total > 0 else np.ones(10) / 10.0

    def save_models(self, dir_path, name):
        joblib.dump(self.model, get_path(dir_path, name, "pkl"))

    def load_models(self, dir_path, name):
        self.model = joblib.load(get_path(dir_path, name, "pkl"))


class RandomForestModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y, sequence=None):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        self.model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        self.model.fit(X, y)

    def predict_proba(self, X, last_digits=None, current_dow=None):
        if self.model is None: return np.ones(10) / 10.0
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(10)
        classes = getattr(self.model, 'classes_', np.arange(10))
        for i, cls in enumerate(classes):
            probs[int(cls)] = raw[i]
        total = probs.sum()
        return probs / total if total > 0 else np.ones(10) / 10.0

    def save_models(self, dir_path, name):
        joblib.dump(self.model, get_path(dir_path, name, "pkl"))

    def load_models(self, dir_path, name):
        self.model = joblib.load(get_path(dir_path, name, "pkl"))


class CatBoostModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y, sequence=None):
        if cb is None: return
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        
        self.model = cb.CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            loss_function='MultiClass', verbose=0, random_seed=42, allow_writing_files=False
        )
        self.model.fit(X, y)

    def predict_proba(self, X, last_digits=None, current_dow=None):
        if self.model is None: return np.ones(10) / 10.0
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(10)
        classes = getattr(self.model, 'classes_', np.arange(10))
        for i, cls in enumerate(classes):
            probs[int(cls)] = raw[i]
        total = probs.sum()
        return probs / total if total > 0 else np.ones(10) / 10.0

    def save_models(self, dir_path, name):
        self.model.save_model(get_path(dir_path, name, "cbm"))

    def load_models(self, dir_path, name):
        self.model = cb.CatBoostClassifier()
        self.model.load_model(get_path(dir_path, name, "cbm"))


class LSTMModel:
    def __init__(self):
        self.model = None
        self.lookback = 14
        self.device = 'cuda' if (torch and torch.cuda.is_available()) else 'cpu'
        
        if torch is not None:
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
                    self.fc = nn.Linear(32, 10)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])
            self.NetClass = Net
        else:
            self.NetClass = None

    def fit(self, X, y, sequence=None):
        if torch is None or sequence is None or len(sequence) < self.lookback + 2: return
        seq = np.array(sequence, dtype=int)
        
        X_seqs, y_targets = [], []
        for i in range(self.lookback, len(seq)):
            X_seqs.append(seq[i - self.lookback : i])
            y_targets.append(seq[i])
            
        X_seqs = torch.tensor(X_seqs, dtype=torch.long)
        y_targets = torch.tensor(y_targets, dtype=torch.long)
        X_onehot = F.one_hot(X_seqs, num_classes=10).float()
        
        self.model = self.NetClass().to(self.device)
        dataset = torch.utils.data.TensorDataset(X_onehot, y_targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        for epoch in range(15):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X=None, last_digits=None, current_dow=None):
        if self.model is None or not last_digits or len(last_digits) < self.lookback:
            return np.ones(10) / 10.0
            
        recent = last_digits[-self.lookback:]
        x_tensor = torch.tensor([recent], dtype=torch.long)
        x_onehot = F.one_hot(x_tensor, num_classes=10).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_onehot)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    def save_models(self, dir_path, name):
        if self.model is not None:
            torch.save(self.model.state_dict(), get_path(dir_path, name, "pt"))

    def load_models(self, dir_path, name):
        if torch is not None:
            self.model = self.NetClass().to(self.device)
            self.model.load_state_dict(torch.load(get_path(dir_path, name, "pt")))


class MarkovModel:
    def __init__(self):
        self.transition = None

    def fit(self, X, y, sequence=None):
        if sequence is None: return
        self.transition = np.full((10, 10, 10), 1.0) # 2nd Order Laplace
        for i in range(len(sequence) - 2):
            prev, curr, nxt = int(sequence[i]), int(sequence[i + 1]), int(sequence[i + 2])
            self.transition[prev][curr][nxt] += 1
        row_sums = self.transition.sum(axis=2, keepdims=True)
        self.transition = self.transition / row_sums

    def predict_proba(self, X=None, last_digits=None, current_dow=None):
        if self.transition is None or not last_digits or len(last_digits) < 2:
            return np.ones(10) / 10.0
        prev, curr = int(last_digits[-2]), int(last_digits[-1])
        return self.transition[prev][curr].copy()

    def save_models(self, dir_path, name):
        joblib.dump(self.transition, get_path(dir_path, name, "pkl"))

    def load_models(self, dir_path, name):
        self.transition = joblib.load(get_path(dir_path, name, "pkl"))


class FrequencyModel:
    def __init__(self):
        self.probs = None

    def fit(self, X, y, sequence=None):
        if sequence is None: return
        counts = np.zeros(10)
        for d in sequence:
            counts[int(d)] += 1
        self.probs = (counts + 1.0) / (counts.sum() + 10.0)

    def predict_proba(self, X=None, last_digits=None, current_dow=None):
        if self.probs is None: return np.ones(10) / 10.0
        return self.probs.copy()

    def save_models(self, dir_path, name):
        joblib.dump(self.probs, get_path(dir_path, name, "pkl"))

    def load_models(self, dir_path, name):
        self.probs = joblib.load(get_path(dir_path, name, "pkl"))


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

MODEL_TYPES = {
    'xgb': XGBoostModel,
    'lgb': LightGBMModel,
    'rf': RandomForestModel,
    'cat': CatBoostModel,
    'lstm': LSTMModel,
    'markov': MarkovModel,
    'freq': FrequencyModel,
}

if AutoGluonModel is not None:
    MODEL_TYPES['ag'] = AutoGluonModel

FEATURE_MODELS = {'xgb', 'lgb', 'rf', 'cat', 'ag'}
SEQUENCE_MODELS = {'markov', 'freq', 'lstm'}
