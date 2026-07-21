import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def get_pana_type(pana_str: str) -> str:
    """Classify a Pana as SP (Single), DP (Double), or TP (Triple)."""
    # Clean it just in case
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

def get_pana_digit(pana_str: str) -> int:
    """Calculate the main digit from a Pana (sum modulo 10)."""
    try:
        return sum(int(d) for d in str(pana_str)) % 10
    except Exception:
        return -1

def generate_pana_universe() -> Dict[str, List[str]]:
    """Generate all theoretical valid Panas grouped by Type and Digit."""
    # A valid Pana typically has digits in ascending order, but since we just
    # want the cross-product mapping, we'll iterate through 000 to 999 
    # and map them. (In actual games, digits are sorted ascending, 
    # e.g. 126 not 621 or 261).
    
    universe = {'SP': {}, 'DP': {}, 'TP': {}}
    for i in range(10):
        universe['SP'][i] = []
        universe['DP'][i] = []
        universe['TP'][i] = []
        
    for d1 in range(10):
        for d2 in range(d1, 10):
            for d3 in range(d2, 10):
                p_str = f"{d1}{d2}{d3}"
                p_type = get_pana_type(p_str)
                p_digit = get_pana_digit(p_str)
                
                if p_type in universe:
                    universe[p_type][p_digit].append(p_str)
                    
    return universe

def build_pana_features(pana_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build momentum features for Pana Type prediction.
    Target: 0=SP, 1=DP, 2=TP
    """
    df = pd.DataFrame({'pana': pana_series})
    df['type'] = df['pana'].apply(get_pana_type)
    
    # Filter out unknowns for training
    valid_mask = df['type'] != 'UNKNOWN'
    df = df[valid_mask].copy()
    
    type_map = {'SP': 0, 'DP': 1, 'TP': 2}
    df['target'] = df['type'].map(type_map)
    
    # Features
    features = pd.DataFrame(index=df.index)
    
    # Rolling frequencies
    is_sp = (df['type'] == 'SP').astype(int)
    is_dp = (df['type'] == 'DP').astype(int)
    
    features['sp_freq_10'] = is_sp.rolling(10, min_periods=1).mean()
    features['dp_freq_10'] = is_dp.rolling(10, min_periods=1).mean()
    features['sp_freq_30'] = is_sp.rolling(30, min_periods=1).mean()
    features['dp_freq_30'] = is_dp.rolling(30, min_periods=1).mean()
    
    # Days since last DP / TP
    features['days_since_dp'] = df.groupby(is_dp.cumsum()).cumcount()
    features['days_since_tp'] = df.groupby((df['type'] == 'TP').astype(int).cumsum()).cumcount()
    
    # Shift target so we predict *tomorrow* using *today's* features
    y = df['target'].shift(-1).dropna()
    X = features.loc[y.index]
    
    # Return features for the last row separately for live prediction
    last_row = features.iloc[-1:]
    
    return X, y.astype(int), last_row
