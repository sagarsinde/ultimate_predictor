import pandas as pd
import numpy as np

def process_and_sort_cards(df, card_cols_map):
    """
    Standardizes column names, drops missing/star values, 
    and applies horizontal sorting to card groups.
    """
    # Rename columns to match our standard target structure
    df = df.rename(columns=card_cols_map)
    
    m_cols = ['Morning_card1', 'Morning_card2', 'Morning_card3']
    e_cols = ['Evening_number1', 'Evening_number2', 'Evening_number3']
    all_cards = m_cols + e_cols
    
    # Drop rows containing '*' or NaN in any card columns
    for col in all_cards:
        if col in df.columns:
            df = df[df[col].astype(str).str.strip() != '*']
            df = df.dropna(subset=[col])
            df[col] = df[col].astype(int)
            
    # Ensure rows are complete before horizontal sorting
    df = df.dropna(subset=all_cards)
    
    # Apply horizontal sorting to group identical physical combinations
    m_sorted = np.sort(df[m_cols].values, axis=1)
    e_sorted = np.sort(df[e_cols].values, axis=1)
    
    df[m_cols] = m_sorted
    df[e_cols] = e_sorted
    
    # Recalculate mathematical targets to prevent scraper errors
    df['Morning_number'] = (df['Morning_card1'] + df['Morning_card2'] + df['Morning_card3']) % 10
    df['Evening_number'] = (df['Evening_number1'] + df['Evening_number2'] + df['Evening_number3']) % 10
    
    req_cols = ['Date', 'Day_of_Week'] + all_cards + ['Morning_number', 'Evening_number']
    return df[req_cols]

if __name__ == "__main__":
    # Load raw sets
    df_old = pd.read_csv('dataset.csv')
    df_new = pd.read_csv('draw_data.csv')
    
    # Map for dataset.csv (already close to target)
    map_old = {
        'Morning_card1': 'Morning_card1', 'Morning_card2': 'Morning_card2', 'Morning_card3': 'Morning_card3',
        'Evening_number1': 'Evening_number1', 'Evening_number2': 'Evening_number2', 'Evening_number3': 'Evening_number3'
    }
    
    # Map for draw_data.csv (needs remapping)
    map_new = {
        'Morning_Card1': 'Morning_card1', 'Morning_Card2': 'Morning_card2', 'Morning_Card3': 'Morning_card3',
        'Evening_Card1': 'Evening_number1', 'Evening_Card2': 'Evening_number2', 'Evening_Card3': 'Evening_number3'
    }
    
    print("Processing older dataset (2020+)...")
    df_old_clean = process_and_sort_cards(df_old, map_old)
    
    print("Processing newer dataset (up to 2026)...")
    df_new_clean = process_and_sort_cards(df_new, map_new)
    
    print("Merging datasets together...")
    combined = pd.concat([df_old_clean, df_new_clean], ignore_index=True)
    
    # Parse dates properly checking both standard string formats
    combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
    combined = combined.dropna(subset=['Date'])
    
    # Sort chronologically and drop duplicates based on unique drawing dates
    combined = combined.sort_values('Date').reset_index(drop=True)
    combined = combined.drop_duplicates(subset=['Date'])
    
    # Generate the sequential Draw_Index for unbroken time-series features
    combined['Draw_Index'] = range(1, len(combined) + 1)
    
    # Format Date cleanly for CSV export
    combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
    
    output_filename = 'combined_data.csv'
    combined.to_csv(output_filename, index=False)
    print(f"Done! Saved {len(combined)} completely clean rows to {output_filename}.")