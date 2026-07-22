import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Tuple, Dict

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

# Load Astro Dates once to avoid I/O overhead
_ASTRO_DATES = set()
def _load_astro_dates():
    global _ASTRO_DATES
    if _ASTRO_DATES: return
    try:
        path = os.path.join(os.path.dirname(__file__), '..', 'astrology', 'drikpanchang_dates.json')
        with open(path, 'r', encoding='utf-8') as f:
            dp_data = json.load(f)
        for year, data in dp_data.items():
            for d_str in data.get('Amavasya', []) + data.get('Purnima', []):
                parts = d_str.split(',')
                if len(parts) >= 2:
                    clean_str = (parts[0] + ',' + parts[1]).strip()
                    try:
                        dt = datetime.strptime(clean_str, '%B %d, %Y')
                        _ASTRO_DATES.add(dt.strftime('%Y-%m-%d'))
                    except ValueError:
                        pass
    except Exception:
        pass

def build_pana_features(pana_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Legacy compatibility for Pana Type prediction."""
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
    features['sp_freq_30'] = is_sp.rolling(30, min_periods=1).mean()
    features['dp_freq_30'] = is_dp.rolling(30, min_periods=1).mean()
    
    features['days_since_dp'] = df.groupby(is_dp.cumsum()).cumcount()
    features['days_since_tp'] = df.groupby(is_tp.cumsum()).cumcount()
    
    y = df['target'].shift(-1).dropna()
    X = features.loc[y.index]
    last_row = features.iloc[-1:]
    
    return X, y.astype(int), last_row

def build_deep_pana_features(df: pd.DataFrame, is_morning: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Builds comprehensive ML features for Daily Pana predictions, including Astrological Fallback matrices.
    Returns: X, y_digit, y_type, last_row (for inference)
    """
    _load_astro_dates()
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    pana_col = 'Morning_Panna' if 'Morning_Panna' in df.columns else 'Morning_pana'
    if not is_morning:
        pana_col = 'Evening_Panna' if 'Evening_Panna' in df.columns else 'Evening_pana'
        
    num_col = 'Morning_number' if is_morning else 'Evening_number'
    
    # Filter valid rows
    df = df[(df[num_col] != '*') & (df[pana_col].notna())].copy()
    
    features = pd.DataFrame(index=df.index)
    
    # Target columns
    df['pana_type'] = df[pana_col].apply(get_pana_type)
    type_map = {'SP': 0, 'DP': 1, 'TP': 2}
    
    y_type = df['pana_type'].map(type_map)
    y_digit = df[num_col].astype(int)
    
    # Feature 1: Astrological Flags (Activates Dealer Fallback weights naturally)
    features['is_astro'] = df['Date'].dt.strftime('%Y-%m-%d').isin(_ASTRO_DATES).astype(int)
    
    # Feature 2: Day of Week (cyclical)
    dow = df['Date'].dt.dayofweek
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    # Feature 3: Type Momentum (Human Fantasy chasing specific types)
    is_sp = (df['pana_type'] == 'SP').astype(int)
    is_dp = (df['pana_type'] == 'DP').astype(int)
    features['sp_freq_14'] = is_sp.rolling(14, min_periods=1).mean()
    features['dp_freq_14'] = is_dp.rolling(14, min_periods=1).mean()
    
    # Feature 4: Digit Momentum (Human Fantasy vs Dealer Fallback)
    # Track the hit frequency of the current digit in the last 14 days
    for d in range(10):
        is_d = (y_digit == d).astype(int)
        features[f'digit_{d}_freq14'] = is_d.rolling(14, min_periods=1).mean()
        
    # Shift targets to predict *tomorrow* using *today's* rolling state
    y_type_shifted = y_type.shift(-1)
    y_digit_shifted = y_digit.shift(-1)
    
    valid_mask = y_type_shifted.notna() & y_digit_shifted.notna()
    
    X = features[valid_mask]
    y_t = y_type_shifted[valid_mask].astype(int)
    y_d = y_digit_shifted[valid_mask].astype(int)
    
    last_row = features.iloc[-1:]
    
    return X, y_d, y_t, last_row
