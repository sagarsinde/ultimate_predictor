import pandas as pd
from typing import Tuple

def get_pana_type(pana_str: str) -> str:
    """Classify a Pana as SP, DP, or TP based on unique digits."""
    pana_str = str(pana_str).strip()
    if len(pana_str) != 3 or not pana_str.isdigit():
        return 'UNKNOWN'
        
    unique_digits = len(set(pana_str))
    if unique_digits == 3:
        return 'SP'
    elif unique_digits == 2:
        return 'DP'
    elif unique_digits == 1:
        return 'TP'
    return 'UNKNOWN'

def build_pana_features(pana_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build momentum features for Pana Type prediction.
    Target: 0=SP, 1=DP, 2=TP
    """
    df = pd.DataFrame({'pana': pana_series})
    df['type'] = df['pana'].apply(get_pana_type)
    
    valid_mask = df['type'] != 'UNKNOWN'
    df = df[valid_mask].copy()
    
    type_map = {'SP': 0, 'DP': 1, 'TP': 2}
    df['target'] = df['type'].map(type_map)
    
    features = pd.DataFrame(index=df.index)
    
    is_sp = (df['type'] == 'SP').astype(int)
    is_dp = (df['type'] == 'DP').astype(int)
    is_tp = (df['type'] == 'TP').astype(int)
    
    features['sp_freq_10'] = is_sp.rolling(10, min_periods=1).mean()
    features['dp_freq_10'] = is_dp.rolling(10, min_periods=1).mean()
    features['tp_freq_30'] = is_tp.rolling(30, min_periods=1).mean()
    
    features['sp_freq_30'] = is_sp.rolling(30, min_periods=1).mean()
    features['dp_freq_30'] = is_dp.rolling(30, min_periods=1).mean()
    
    features['days_since_dp'] = df.groupby(is_dp.cumsum()).cumcount()
    features['days_since_tp'] = df.groupby(is_tp.cumsum()).cumcount()
    
    # Shift target so we predict *tomorrow* using *today's* features
    y = df['target'].shift(-1).dropna()
    X = features.loc[y.index]
    
    last_row = features.iloc[-1:]
    
    return X, y.astype(int), last_row
