import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

def load_data(filepath='draw_data.csv'):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    for prefix in ['Morning', 'Evening']:
        for i in [1, 2, 3]:
            col = f'{prefix}_Card{i}'
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Calculate Total Sum (0-27)
        df[f'{prefix}_Total_Sum'] = df[f'{prefix}_Card1'] + df[f'{prefix}_Card2'] + df[f'{prefix}_Card3']
        # Calculate the final winning digit (0-9)
        df[f'{prefix}_Number'] = df[f'{prefix}_Total_Sum'] % 10
        
        # Create a string representation of the panel (e.g., "3-8-9") for frequency counting
        # Ensure it's sorted so that 3-8-9 and 9-3-8 are treated identically if order doesn't matter
        # Wait, if cards are drawn in sequence, order might matter. Let's keep exact order first.
        df[f'{prefix}_Panel'] = df[f'{prefix}_Card1'].astype(str) + "-" + df[f'{prefix}_Card2'].astype(str) + "-" + df[f'{prefix}_Card3'].astype(str)
        
    return df

def analyze_card_frequencies(df):
    """
    1. Card-Level Frequency Analysis (The Deck Check)
    """
    print("\n=======================================================")
    print("1. CARD-LEVEL FREQUENCY ANALYSIS (THE DECK CHECK)")
    print("=======================================================")
    
    for prefix in ['Morning', 'Evening']:
        print(f"\n--- {prefix} Draw Card Frequencies ---")
        card_counts = pd.concat([df[f'{prefix}_Card1'], df[f'{prefix}_Card2'], df[f'{prefix}_Card3']]).value_counts().sort_index()
        total_cards = card_counts.sum()
        
        for num, count in card_counts.items():
            percentage = (count / total_cards) * 100
            print(f"Card {num}: {count} times ({percentage:.2f}%)")
            
        # Optional: plot it
        plt.figure(figsize=(10, 4))
        sns.barplot(x=card_counts.index, y=card_counts.values, palette='viridis')
        plt.title(f'{prefix} Draw: Frequency of Individual Cards (0-9)')
        plt.xlabel('Card Value')
        plt.ylabel('Total Occurrences')
        plt.savefig(f'{prefix.lower()}_card_frequency.png')
        plt.close()

def analyze_bell_curve(df):
    """
    2. Theoretical vs. Actual Bell Curve (The Sum Check)
    """
    print("\n=======================================================")
    print("2. TOTAL SUM DISTRIBUTION (THE BELL CURVE CHECK)")
    print("=======================================================")
    
    for prefix in ['Morning', 'Evening']:
        print(f"\n--- {prefix} Total Sum Distribution (0-27) ---")
        sum_counts = df[f'{prefix}_Total_Sum'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 5))
        sns.barplot(x=sum_counts.index, y=sum_counts.values, palette='magma')
        plt.title(f'{prefix} Draw: Total Sum (0-27) Distribution')
        plt.xlabel('Total Sum')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{prefix.lower()}_sum_distribution.png')
        plt.close()
        
        print("Plotted Sum Distribution. Check the saved images to see if the curve is perfectly symmetrical or skewed.")
        
        # Which sum leads to 0?
        zeros = df[df[f'{prefix}_Number'] == 0]
        zero_sums = zeros[f'{prefix}_Total_Sum'].value_counts().sort_index()
        print(f"When {prefix} final digit was 0, the sums were:")
        for s, count in zero_sums.items():
            print(f"  Sum = {s}: {count} times")

def analyze_jodi_leakage(df):
    """
    3. The Jodi Leakage Test (Morning vs. Evening Independence)
    """
    print("\n=======================================================")
    print("3. JODI LEAKAGE TEST (MORNING vs EVENING CORRELATION)")
    print("=======================================================")
    
    # Pearson Correlation between Total Sums
    correlation = df['Morning_Total_Sum'].corr(df['Evening_Total_Sum'])
    print(f"Pearson Correlation (Morning Sum vs Evening Sum): {correlation:.4f}")
    
    if abs(correlation) > 0.05:
        print(">>> WARNING: There is a statistically measurable correlation between Morning and Evening!")
        print(">>> This suggests the Evening deck is influenced by the Morning draw.")
    else:
        print(">>> Morning and Evening sums appear to be mathematically independent.")
        
    # Mutual Information Score
    X = df[['Morning_Total_Sum']]
    y = df['Evening_Total_Sum']
    mi = mutual_info_regression(X, y, random_state=42)[0]
    print(f"Mutual Information Score: {mi:.4f} (Higher means more dependence)")

def analyze_top_panels(df):
    """
    4. Top Panel Occurrences
    """
    print("\n=======================================================")
    print("4. TOP PANEL OCCURRENCES (3-CARD COMBOS)")
    print("=======================================================")
    
    for prefix in ['Morning', 'Evening']:
        print(f"\n--- {prefix} Most Frequent Panels ---")
        top_panels = df[f'{prefix}_Panel'].value_counts().head(10)
        print(top_panels)

if __name__ == "__main__":
    df = load_data('draw_data.csv')
    print(f"Loaded {len(df)} draw records for Deep Analysis.\n")
    
    analyze_card_frequencies(df)
    analyze_bell_curve(df)
    analyze_jodi_leakage(df)
    analyze_top_panels(df)
    print("\nDeep Analysis Complete. All charts saved to the current directory.")
