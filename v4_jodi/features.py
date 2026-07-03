import os
import pandas as pd
import numpy as np

ALL_FEATURE_GROUPS = [
    'pressure',
    'lags',
    'day_of_week',
]

MARKET_CONFIG = {
    'kalyan': {'file': 'combined_data.csv'},
    'mb': {'file': 'main_bazar_dataset.csv'}
}

def load_raw_data(market):
    config = MARKET_CONFIG.get(market.lower())
    if not config:
        raise ValueError(f"Unknown market: {market}")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, config['file'])
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def get_window_size(market, label):
    """
    Since Jodi has 100 classes, we need bigger windows.
    labels: '3m', '6m', '12m', 'full'
    """
    days_per_month = 26 if market.lower() == 'kalyan' else 22
    if label == '3m': return int(days_per_month * 3)
    if label == '6m': return int(days_per_month * 6)
    if label == '12m': return int(days_per_month * 12)
    if label == 'full': return None
    raise ValueError(f"Unknown window label: {label}")

def slice_window(df, window_size):
    if window_size is None or window_size >= len(df):
        return df.copy()
    return df.iloc[-window_size:].copy()

def build_features(df, active_groups=None):
    if active_groups is None:
        active_groups = ALL_FEATURE_GROUPS.copy()

    df = df.copy()
    df['Jodi_Target'] = df['Morning_number'].astype(int) * 10 + df['Evening_number'].astype(int)

    features = pd.DataFrame(index=df.index)
    features['_date'] = df['Date']

    if 'pressure' in active_groups:
        for digit in range(10):
            hit_m = (df['Morning_number'] == digit).astype(int)
            features[f'M_Days_Since_{digit}'] = hit_m.groupby((hit_m == 1).cumsum()).cumcount()
            
            hit_e = (df['Evening_number'] == digit).astype(int)
            features[f'E_Days_Since_{digit}'] = hit_e.groupby((hit_e == 1).cumsum()).cumcount()

    if 'lags' in active_groups:
        features['Lag1_Jodi'] = df['Jodi_Target'].shift(1).fillna(-1).astype(int)
        features['Lag2_Jodi'] = df['Jodi_Target'].shift(2).fillna(-1).astype(int)

    if 'day_of_week' in active_groups:
        features['Day_of_Week_Num'] = df['Date'].dt.dayofweek

    valid_idx = features.dropna().index
    features = features.loc[valid_idx]
    y = df.loc[valid_idx, 'Jodi_Target'].astype(int)

    return features, y, active_groups
