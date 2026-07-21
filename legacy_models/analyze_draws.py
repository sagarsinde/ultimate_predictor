import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest

def create_template(filepath='template.csv'):
    """
    Creates an empty template CSV file with the required column names.
    """
    columns = [
        'Date', 'Day_of_Week',
        'Morning_Card1', 'Morning_Card2', 'Morning_Card3', 'Jodi_Number',
        'Evening_Card1', 'Evening_Card2', 'Evening_Card3'
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(filepath, index=False)
    print(f"Template CSV created at: {filepath}")

def load_and_process_data(filepath):
    """
    Loads historical draw data and calculates the Morning, Evening, and Jodi Numbers.
    """
    df = pd.read_csv(filepath)
    
    # Drop rows with missing card values to ensure clean calculation
    card_cols = [
        'Morning_Card1', 'Morning_Card2', 'Morning_Card3',
        'Evening_Card1', 'Evening_Card2', 'Evening_Card3'
    ]
    df = df.dropna(subset=card_cols).copy()
    
    # Convert card columns to integer so they sum correctly
    for col in card_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Calculate Morning Number: sum of Morning cards, keep last digit
    df['Morning_Sum'] = df['Morning_Card1'] + df['Morning_Card2'] + df['Morning_Card3']
    df['Morning_Number'] = df['Morning_Sum'] % 10
    
    # Calculate Evening Number: sum of Evening cards, keep last digit
    df['Evening_Sum'] = df['Evening_Card1'] + df['Evening_Card2'] + df['Evening_Card3']
    df['Evening_Number'] = df['Evening_Sum'] % 10
    
    # Convert the extracted Jodi_Number to an integer, or calculate it to verify
    # A Jodi Number combines the Morning Number and Evening Number (e.g., 8 and 9 -> 89)
    df['Jodi_Number_Calculated'] = df['Morning_Number'] * 10 + df['Evening_Number']
    
    # Convert Date to datetime object if possible for timeline analysis
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        
    return df

def analyze_frequencies(df):
    """
    Analyzes hot and cold numbers for Morning Number, Evening Number, and Jodi Number.
    """
    results = {}
    
    for col, name in zip(['Morning_Number', 'Evening_Number', 'Jodi_Number_Calculated'], 
                         ['Morning Number', 'Evening Number', 'Jodi Number']):
        
        freqs = df[col].value_counts().sort_values(ascending=False)
        
        results[name] = {
            'hot': freqs.head(5),
            'cold': freqs.tail(5)
        }
        
        print(f"\n--- {name} Frequencies ---")
        print("Top 5 HOT Numbers:")
        print(results[name]['hot'].to_string())
        print("\nTop 5 COLD Numbers:")
        print(results[name]['cold'].to_string())
        
    return results

def test_uniform_distribution(df):
    """
    Performs Chi-Square Goodness-of-Fit test on Jodi Number distribution.
    Tests if the drawn numbers (00-99) follow a perfectly uniform distribution (1% each).
    """
    print("\n--- Chi-Square Goodness-of-Fit Test (Jodi Number) ---")
    
    # Count frequencies of all 100 possible outcomes (0-99)
    # Reindex ensures that numbers that never appeared are counted as 0
    actual_counts = df['Jodi_Number_Calculated'].value_counts().reindex(range(100), fill_value=0)
    
    # Expected counts for perfectly uniform distribution
    total_draws = len(df)
    expected_counts = [total_draws / 100.0] * 100
    
    chi_stat, p_value = chisquare(f_obs=actual_counts, f_exp=expected_counts)
    
    print(f"Chi-Square Statistic: {chi_stat:.4f}")
    print(f"P-Value: {p_value:.4e}")
    
    # Standard significance level alpha = 0.05
    if p_value < 0.05:
        print("Result: Reject the null hypothesis.")
        print("Conclusion: The distribution of Jodi Numbers is statistically SIGNIFICANTLY BIASED and not uniform.")
        print("There is evidence of physical bias (e.g., inadequate shuffling).")
    else:
        print("Result: Fail to reject the null hypothesis.")
        print("Conclusion: The distribution appears uniform. No significant evidence of physical bias detected.")

def detect_anomalies(df):
    """
    Uses Isolation Forest to detect highly anomalous draw patterns based on the 
    raw drawn cards and their resulting values.
    """
    print("\n--- Anomaly Detection (Isolation Forest) ---")
    
    # Features to use for anomaly detection
    features = [
        'Morning_Card1', 'Morning_Card2', 'Morning_Card3',
        'Evening_Card1', 'Evening_Card2', 'Evening_Card3',
        'Morning_Number', 'Evening_Number', 'Jodi_Number_Calculated'
    ]
    
    X = df[features]
    
    # Isolation forest configuration: 
    # contamination represents the expected proportion of outliers (e.g., 2%)
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    
    # Fit and predict. 1 = Inlier, -1 = Outlier
    df['Anomaly'] = model.fit_predict(X)
    df['Anomaly_Score'] = model.decision_function(X) # lower score means more anomalous
    
    anomalies = df[df['Anomaly'] == -1].sort_values('Anomaly_Score')
    
    print(f"Detected {len(anomalies)} highly unusual draws out of {len(df)} total draws.")
    
    if len(anomalies) > 0:
        print("\nTop 5 Most Anomalous Draws:")
        cols_to_show = ['Date', 'Jodi_Number', 'Anomaly_Score'] if 'Date' in df.columns else ['Jodi_Number', 'Anomaly_Score']
        print(anomalies.head()[cols_to_show])

def plot_visualizations(df):
    """
    Generates bar charts and heatmaps for the frequency distributions.
    """
    # 1. Bar Chart for Morning Number
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Morning_Number', data=df, order=range(10), palette='viridis')
    plt.title('Frequency Distribution: Morning Number')
    plt.xlabel('Number (0-9)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('frequency_morning.png')
    plt.show()
    
    # 2. Bar Chart for Evening Number
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Evening_Number', data=df, order=range(10), palette='magma')
    plt.title('Frequency Distribution: Evening Number')
    plt.xlabel('Number (0-9)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('frequency_evening.png')
    plt.show()

    # 3. Heatmap for Jodi Number (00-99)
    # We create a 10x10 matrix where rows=Morning_Number (tens digit) and cols=Evening_Number (ones digit)
    heatmap_data = pd.crosstab(df['Morning_Number'], df['Evening_Number'])
    
    # Ensure it's exactly 10x10 even if some numbers never appeared
    heatmap_data = heatmap_data.reindex(index=range(10), columns=range(10), fill_value=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Frequency'})
    plt.title('Heatmap: Jodi Number Frequencies')
    plt.xlabel('Evening Number')
    plt.ylabel('Morning Number')
    plt.tight_layout()
    plt.savefig('heatmap_jodi_prize.png')
    plt.show()


if __name__ == "__main__":
    # 1. Specify your dataset file
    dataset_file = 'draw_data.csv'
    
    # Run this once to generate the template, then you can comment it out
    # create_template('draw_data.csv') 
    
    try:
        # Load the data
        print(f"Loading data from {dataset_file}...")
        df_draws = load_and_process_data(dataset_file)
        print(f"Successfully loaded {len(df_draws)} records.\n")
        
        # Run Analytics
        analyze_frequencies(df_draws)
        test_uniform_distribution(df_draws)
        detect_anomalies(df_draws)
        
        # Plot Visualizations
        print("\nGenerating Visualizations...")
        plot_visualizations(df_draws)
        print("Visualizations saved as PNG files in the current directory.")
        
    except FileNotFoundError:
        print(f"Error: {dataset_file} not found.")
        print("Generating a template CSV for you...")
        create_template(dataset_file)
        print("Please fill out the template with your historical data and run the script again.")
