"""
v2/features.py — Multi-Window Feature Builder

Builds strictly causal features from a raw dataset.
Supports window slicing (1M, 2M, 3M, full).
Every feature group is toggleable for ablation testing.
Works for both Kalyan and Main Bazar datasets.
"""

import pandas as pd
import numpy as np
from collections import Counter

# ---------------------------------------------------------------------------
# Market configurations
# ---------------------------------------------------------------------------
MARKET_CONFIG = {
    'kalyan': {
        'csv': 'true_kalyan_main_dataset.csv',
        'day_col': 'Day_of_Week',
        'playing_days_per_week': 6,  # Mon-Sat
        'draws_per_month': 26,
    },
    'mb': {
        'csv': 'main_bazar_dataset.csv',
        'day_col': 'Day',
        'playing_days_per_week': 5,  # Mon-Fri
        'draws_per_month': 22,
    }
}

# ---------------------------------------------------------------------------
# Feature group registry — each group can be toggled on/off for ablation
# ---------------------------------------------------------------------------
ALL_FEATURE_GROUPS = [
    'lags',
    'days_since_hit',
    'hits_last_7',
    'day_of_week',
    'morning_evening_corr',
    'hot_cold_ratio',
    'gap_velocity',
]


def load_raw_data(market: str) -> pd.DataFrame:
    """Load raw CSV for a market, standardize columns."""
    cfg = MARKET_CONFIG[market]
    df = pd.read_csv(cfg['csv'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Standardize day-of-week column name
    if cfg['day_col'] != 'Day_of_Week' and cfg['day_col'] in df.columns:
        df = df.rename(columns={cfg['day_col']: 'Day_of_Week'})

    return df


def slice_window(df: pd.DataFrame, window_draws: int = None) -> pd.DataFrame:
    """Slice dataframe to last N draws. None = full history."""
    if window_draws is None or window_draws >= len(df):
        return df.copy()
    return df.iloc[-window_draws:].reset_index(drop=True)


def get_window_size(market: str, window_label: str) -> int:
    """Convert window label to number of draws."""
    dpm = MARKET_CONFIG[market]['draws_per_month']
    mapping = {
        '1m': dpm * 1,
        '2m': dpm * 2,
        '3m': dpm * 3,
        'full': None,  # signals "use everything"
    }
    return mapping[window_label]


def build_features(
    df: pd.DataFrame,
    active_groups: list = None,
) -> tuple:
    """
    Build feature matrix from a dataframe.

    Args:
        df: Raw dataframe with Date, Morning_number, Evening_number columns.
        active_groups: List of feature group names to include.
                       If None, all groups are active.

    Returns:
        X: Feature matrix (pd.DataFrame)
        y_morning: Morning target series
        y_evening: Evening target series
        group_columns: Dict mapping group_name -> list of column names
    """
    if active_groups is None:
        active_groups = ALL_FEATURE_GROUPS.copy()

    df = df.copy()
    morning = df['Morning_number'].astype(int).values
    evening = df['Evening_number'].astype(int).values
    n = len(df)

    group_columns = {}
    feature_dict = {i: {} for i in range(n)}

    # ----- GROUP: lags (previous 1-7 results) -----
    if 'lags' in active_groups:
        cols = []
        for lag in range(1, 8):
            m_col = f'M_lag_{lag}'
            e_col = f'E_lag_{lag}'
            cols.extend([m_col, e_col])
            for i in range(n):
                if i >= lag:
                    feature_dict[i][m_col] = morning[i - lag]
                    feature_dict[i][e_col] = evening[i - lag]
                else:
                    feature_dict[i][m_col] = np.nan
                    feature_dict[i][e_col] = np.nan
        group_columns['lags'] = cols

    # ----- GROUP: days_since_hit (per digit 0-9) -----
    if 'days_since_hit' in active_groups:
        cols = []
        for d in range(10):
            cols.extend([f'M_gap_{d}', f'E_gap_{d}'])

        m_gap = {d: 0 for d in range(10)}
        e_gap = {d: 0 for d in range(10)}

        for i in range(n):
            for d in range(10):
                feature_dict[i][f'M_gap_{d}'] = m_gap[d]
                feature_dict[i][f'E_gap_{d}'] = e_gap[d]
            # Update gaps AFTER recording (strictly causal)
            for d in range(10):
                m_gap[d] = 0 if morning[i] == d else m_gap[d] + 1
                e_gap[d] = 0 if evening[i] == d else e_gap[d] + 1

        group_columns['days_since_hit'] = cols

    # ----- GROUP: hits_last_7 (digit frequency in prior 7 draws) -----
    if 'hits_last_7' in active_groups:
        cols = []
        for d in range(10):
            cols.extend([f'M_hits7_{d}', f'E_hits7_{d}'])

        for i in range(n):
            if i >= 7:
                m_window = morning[i-7:i]
                e_window = evening[i-7:i]
                m_counts = Counter(m_window)
                e_counts = Counter(e_window)
            else:
                m_counts = Counter()
                e_counts = Counter()

            for d in range(10):
                feature_dict[i][f'M_hits7_{d}'] = m_counts.get(d, 0)
                feature_dict[i][f'E_hits7_{d}'] = e_counts.get(d, 0)

        group_columns['hits_last_7'] = cols

    # ----- GROUP: day_of_week (cyclical sin/cos) -----
    if 'day_of_week' in active_groups:
        cols = ['DOW_sin', 'DOW_cos']
        dow_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5}

        for i in range(n):
            if 'Day_of_Week' in df.columns:
                dow_num = dow_map.get(df.iloc[i]['Day_of_Week'], 0)
            else:
                dow_num = df.iloc[i]['Date'].dayofweek
            feature_dict[i]['DOW_sin'] = np.sin(2 * np.pi * dow_num / 7.0)
            feature_dict[i]['DOW_cos'] = np.cos(2 * np.pi * dow_num / 7.0)

        group_columns['day_of_week'] = cols

    # ----- GROUP: morning_evening_corr -----
    # Conditional frequency: given morning=X, what's the distribution of evening?
    # Uses expanding window (all data before current row)
    if 'morning_evening_corr' in active_groups:
        cols = [f'ME_corr_{d}' for d in range(10)]

        # Expanding conditional count: P(evening=d | morning=m_today's_lag1)
        joint_counts = np.zeros((10, 10), dtype=float)  # joint_counts[m][e]

        for i in range(n):
            if i >= 1:
                last_m = morning[i - 1]
                row_total = joint_counts[last_m].sum()
                if row_total > 0:
                    probs = joint_counts[last_m] / row_total
                else:
                    probs = np.ones(10) / 10.0
                for d in range(10):
                    feature_dict[i][f'ME_corr_{d}'] = probs[d]
            else:
                for d in range(10):
                    feature_dict[i][f'ME_corr_{d}'] = 0.1

            # Update joint counts AFTER recording (causal)
            joint_counts[morning[i]][evening[i]] += 1

        group_columns['morning_evening_corr'] = cols

    # ----- GROUP: hot_cold_ratio -----
    # Digit frequency in last 14 draws vs expected 10%
    if 'hot_cold_ratio' in active_groups:
        cols = []
        for d in range(10):
            cols.extend([f'M_hotcold_{d}', f'E_hotcold_{d}'])

        for i in range(n):
            lookback = min(i, 14)
            if lookback >= 3:
                m_window = morning[i-lookback:i]
                e_window = evening[i-lookback:i]
                m_counts = Counter(m_window)
                e_counts = Counter(e_window)
                expected = lookback / 10.0
                for d in range(10):
                    feature_dict[i][f'M_hotcold_{d}'] = (m_counts.get(d, 0) / expected) - 1.0
                    feature_dict[i][f'E_hotcold_{d}'] = (e_counts.get(d, 0) / expected) - 1.0
            else:
                for d in range(10):
                    feature_dict[i][f'M_hotcold_{d}'] = 0.0
                    feature_dict[i][f'E_hotcold_{d}'] = 0.0

        group_columns['hot_cold_ratio'] = cols

    # ----- GROUP: gap_velocity -----
    # Is the gap for a digit accelerating or decelerating?
    # gap_velocity = current_gap - gap_at_last_appearance
    if 'gap_velocity' in active_groups:
        cols = []
        for d in range(10):
            cols.extend([f'M_gapvel_{d}', f'E_gapvel_{d}'])

        m_prev_gap = {d: 0 for d in range(10)}
        m_curr_gap = {d: 0 for d in range(10)}
        e_prev_gap = {d: 0 for d in range(10)}
        e_curr_gap = {d: 0 for d in range(10)}

        for i in range(n):
            for d in range(10):
                feature_dict[i][f'M_gapvel_{d}'] = m_curr_gap[d] - m_prev_gap[d]
                feature_dict[i][f'E_gapvel_{d}'] = e_curr_gap[d] - e_prev_gap[d]

            # Update AFTER recording (causal)
            for d in range(10):
                if morning[i] == d:
                    m_prev_gap[d] = m_curr_gap[d]
                    m_curr_gap[d] = 0
                else:
                    m_curr_gap[d] += 1

                if evening[i] == d:
                    e_prev_gap[d] = e_curr_gap[d]
                    e_curr_gap[d] = 0
                else:
                    e_curr_gap[d] += 1

        group_columns['gap_velocity'] = cols

    # ----- Assemble feature matrix -----
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')

    # Drop rows with NaN (first 7 rows due to lags)
    valid_mask = ~feature_df.isna().any(axis=1)
    feature_df = feature_df[valid_mask].reset_index(drop=True)

    y_morning_out = pd.Series(morning[valid_mask.values], name='Morning_number').astype(int)
    y_evening_out = pd.Series(evening[valid_mask.values], name='Evening_number').astype(int)

    # Also keep dates for time-based splitting
    dates_out = df['Date'].values[valid_mask.values]
    feature_df['_date'] = dates_out

    return feature_df, y_morning_out, y_evening_out, group_columns


def build_prediction_features(
    df: pd.DataFrame,
    active_groups: list = None,
) -> pd.DataFrame:
    """
    Build features and return ONLY the last row's feature vector.
    Used for inference (predicting the next draw).
    """
    feature_df, _, _, group_columns = build_features(df, active_groups)
    last_features = feature_df.iloc[[-1]].copy()

    # Drop internal date column
    if '_date' in last_features.columns:
        last_features = last_features.drop(columns=['_date'])

    return last_features, group_columns
