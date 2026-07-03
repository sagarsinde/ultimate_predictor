"""
v2/models.py — Model Definitions

Four model types, each outputting P(digit=d) for d in {0..9}:
  1. XGBoost (gradient boosting with isotonic calibration)
  2. Random Forest
  3. Markov Chain (transition matrix, no features needed)
  4. Frequency (smoothed digit counts, no features needed)

All models implement a common interface:
  .fit(X, y, sequence=None)
  .predict_proba(X, last_digit=None) -> np.array of shape (10,)
"""

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    import catboost as cb
except ImportError:
    cb = None


class XGBoostModel:
    """XGBoost with isotonic calibration via TimeSeriesSplit."""

    def __init__(self):
        self.model = None
        self.classes_ = np.arange(10)

    def fit(self, X, y, sequence=None):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # XGBoost with num_class=10 requires all 10 classes in training data.
        # Small windows may be missing some digits. Fix: inject one dummy
        # sample per missing class (mean feature vector, tiny sample_weight).
        present = set(np.unique(y))
        missing = set(range(10)) - present
        sample_weights = np.ones(len(y))

        if missing:
            mean_row = X.mean(axis=0, keepdims=True)
            dummy_X = np.repeat(mean_row, len(missing), axis=0)
            dummy_y = np.array(sorted(missing), dtype=int)
            dummy_w = np.full(len(missing), 1e-6)  # near-zero weight

            X = np.vstack([X, dummy_X])
            y = np.concatenate([y, dummy_y])
            sample_weights = np.concatenate([sample_weights, dummy_w])

        base = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=10,
            eval_metric='mlogloss',
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            tree_method='hist',
            random_state=42,
            verbosity=0,
        )
        # Need enough samples per class for calibration
        n_splits = min(5, max(2, len(X) // 10))
        min_class_count = min(np.bincount(y, minlength=10))

        if len(X) < 50 or min_class_count < n_splits:
            try:
                base.fit(X, y, sample_weight=sample_weights)
            except Exception:
                base = xgb.XGBClassifier(
                    objective='multi:softprob', num_class=10,
                    max_depth=2, n_estimators=50, tree_method='hist',
                    random_state=42, verbosity=0,
                )
                base.fit(X, y, sample_weight=sample_weights)
            self.model = base
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            try:
                calibrated = CalibratedClassifierCV(
                    estimator=base, method='isotonic', cv=tscv
                )
                calibrated.fit(X, y, sample_weight=sample_weights)
                self.model = calibrated
            except Exception:
                base.fit(X, y, sample_weight=sample_weights)
                self.model = base

    def predict_proba(self, X, last_digit=None):
        raw = self.model.predict_proba(X)[0]
        # Ensure we have probabilities for all 10 digits
        probs = np.zeros(10)
        if hasattr(self.model, 'classes_'):
            for i, cls in enumerate(self.model.classes_):
                probs[int(cls)] = raw[i]
        else:
            probs = raw
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(10) / 10.0
        return probs


class RandomForestModel:
    """Random Forest classifier."""

    def __init__(self):
        self.model = None

    def fit(self, X, y, sequence=None):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # Ensure all 10 classes are present
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

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y, sample_weight=sample_weights)

    def predict_proba(self, X, last_digit=None):
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(10)
        for i, cls in enumerate(self.model.classes_):
            probs[int(cls)] = raw[i]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(10) / 10.0
        return probs


class MarkovModel:
    """
    First-order Markov Chain with Laplace smoothing.
    P(next_digit | last_digit) from transition counts.
    Does NOT use the feature matrix — operates on raw sequences.
    """

    def __init__(self):
        self.transition = None  # shape (10, 10)

    def fit(self, X, y, sequence=None, dow_sequence=None):
        """
        Args:
            X: ignored (interface compatibility)
            y: ignored
            sequence: 1D array of digit values (the raw sequence to learn from)
        """
        if sequence is None:
            raise ValueError("MarkovModel requires 'sequence' argument")

        # Build transition matrix with Laplace smoothing (alpha=1)
        alpha = 1.0
        self.transition = np.full((10, 10), alpha)

        for i in range(len(sequence) - 1):
            curr = int(sequence[i])
            nxt = int(sequence[i + 1])
            self.transition[curr][nxt] += 1

        # Normalize rows
        row_sums = self.transition.sum(axis=1, keepdims=True)
        self.transition = self.transition / row_sums

    def predict_proba(self, X=None, last_digit=None, current_dow=None):
        """
        Args:
            X: ignored
            last_digit: the most recent digit in the sequence
        Returns:
            probs: P(next_digit=d | last_digit) for d in 0..9
        """
        if last_digit is None:
            # Fallback: uniform
            return np.ones(10) / 10.0
        return self.transition[int(last_digit)].copy()


class FrequencyModel:
    """
    Simple smoothed frequency model.
    P(digit=d) = (count(d) + alpha) / (total + 10*alpha)
    Surprisingly powerful on short windows.
    """

    def __init__(self):
        self.probs = None

    def fit(self, X, y, sequence=None, dow_sequence=None):
        """
        Args:
            X: ignored
            y: ignored
            sequence: 1D array of digit values
        """
        if sequence is None:
            raise ValueError("FrequencyModel requires 'sequence' argument")

        alpha = 1.0  # Laplace smoothing
        counts = np.zeros(10)
        for d in sequence:
            counts[int(d)] += 1

        self.probs = (counts + alpha) / (counts.sum() + 10 * alpha)

    def predict_proba(self, X=None, last_digit=None, current_dow=None):
        if self.probs is None:
            return np.ones(10) / 10.0
        return self.probs.copy()


class DowFrequencyModel:
    """
    Day of Week Frequency Model.
    Calculates frequencies for each digit grouped by day of the week.
    """

    def __init__(self):
        self.probs_by_dow = {}

    def fit(self, X, y, sequence=None, dow_sequence=None):
        if sequence is None or dow_sequence is None:
            raise ValueError("DowFrequencyModel requires 'sequence' and 'dow_sequence'")
        
        alpha = 1.0
        counts = {i: np.zeros(10) for i in range(7)}
        
        for d, dow in zip(sequence, dow_sequence):
            counts[int(dow)][int(d)] += 1
            
        for i in range(7):
            self.probs_by_dow[i] = (counts[i] + alpha) / (counts[i].sum() + 10 * alpha)

    def predict_proba(self, X=None, last_digit=None, current_dow=None):
        if current_dow is None or current_dow not in self.probs_by_dow:
            return np.ones(10) / 10.0
        return self.probs_by_dow[int(current_dow)].copy()


class CatBoostModel:
    """CatBoost classifier for categorical sequence data."""

    def __init__(self):
        self.model = None

    def fit(self, X, y, sequence=None, dow_sequence=None):
        if cb is None:
            raise ImportError("catboost is not installed")
            
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

        self.model = cb.CatBoostClassifier(
            iterations=150,
            depth=4,
            learning_rate=0.05,
            loss_function='MultiClass',
            verbose=0,
            random_seed=42,
            allow_writing_files=False
        )
        self.model.fit(X, y, sample_weight=sample_weights)

    def predict_proba(self, X, last_digit=None, current_dow=None):
        if cb is None:
            return np.ones(10) / 10.0
            
        raw = self.model.predict_proba(X)[0]
        probs = np.zeros(10)
        for i, cls in enumerate(self.model.classes_):
            probs[int(cls)] = raw[i]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(10) / 10.0
        return probs




# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_TYPES = {
    'xgb': XGBoostModel,
    'rf': RandomForestModel,
    'cat': CatBoostModel,
    'markov': MarkovModel,
    'freq': FrequencyModel,
    'dowfreq': DowFrequencyModel,
}

# Models that need the feature matrix
FEATURE_MODELS = {'xgb', 'rf', 'cat'}

# Models that need the raw digit sequence
SEQUENCE_MODELS = {'markov', 'freq', 'dowfreq'}
