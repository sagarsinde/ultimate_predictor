import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')

def load_and_prep(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create the true 2-digit Jodi (e.g., Morning 6, Evening 3 -> "63")
    df['Jodi'] = df['Morning_number'].astype(str) + df['Evening_number'].astype(str)
    
    # Calculate Day of Week and Day of Month
    df['Day_of_Week'] = df['Date'].dt.day_name()
    # Order the days correctly for heatmaps
    df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
        ordered=True)
        
    df['Day_of_Month'] = df['Date'].dt.day
    
    return df

def temporal_heatmap(df):
    print("\n--- 1. Temporal & Calendar Engine ---")
    
    # Day of Week Heatmap
    dow_counts = pd.crosstab(df['Day_of_Week'], df['Jodi'])
    
    plt.figure(figsize=(20, 5))
    sns.heatmap(dow_counts, cmap='YlOrRd', linewidths=.5)
    plt.title("Jodi Frequency by Day of Week")
    plt.xlabel("Jodi (00-99)")
    plt.ylabel("Day of Week")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig('jodi_heatmap_dayofweek.png', dpi=300)
    plt.close()
    
    # Day of Month Heatmap
    dom_counts = pd.crosstab(df['Day_of_Month'], df['Jodi'])
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(dom_counts, cmap='viridis', linewidths=.5)
    plt.title("Jodi Frequency by Day of Month (1st-31st)")
    plt.xlabel("Jodi (00-99)")
    plt.ylabel("Day of Month")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig('jodi_heatmap_dayofmonth.png', dpi=300)
    plt.close()
    
    print("Saved Temporal Heatmaps: jodi_heatmap_dayofweek.png, jodi_heatmap_dayofmonth.png")

def graph_theory_centrality(df):
    print("\n--- 2. Graph Theory & PageRank Engine ---")
    
    G = nx.DiGraph()
    
    jodis = df['Jodi'].tolist()
    # Create edges from one day to the exact next day
    transitions = [(jodis[i], jodis[i+1]) for i in range(len(jodis)-1)]
    
    for u, v in transitions:
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
            
    # Calculate PageRank (Google's algorithm for finding the most central/important nodes)
    pagerank = nx.pagerank(G, weight='weight')
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 'Central' Jodis (The Mathematical Gravity Wells):")
    for jodi, score in sorted_pr[:5]:
        print(f"  Jodi {jodi}: PageRank Score {score:.5f}")
        
    # Plot top subgraph (plotting all 100 nodes is an unreadable mess, so we plot the top 15)
    top_nodes = [x[0] for x in sorted_pr[:15]]
    sub_G = G.subgraph(top_nodes)
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(sub_G, seed=42)
    nx.draw_networkx_nodes(sub_G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(sub_G, pos, edge_color='gray', arrows=True, alpha=0.5)
    nx.draw_networkx_labels(sub_G, pos, font_size=12, font_family='sans-serif', font_weight='bold')
    
    plt.title("Transition Network (Top 15 Central Jodis)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('jodi_network_graph.png', dpi=300)
    plt.close()
    print("Saved Network Graph: jodi_network_graph.png")

def fft_signal_processing(df):
    print("\n--- 3. Signal Processing Engine (FFT) ---")
    
    # To run an FFT, we need a continuous numerical signal. 
    # Jodis are strings ("05"), let's convert them to their integer values (0-99).
    signal = df['Jodi'].astype(int).values
    
    N = len(signal)
    T = 1.0 # 1 day spacing between draws
    
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    
    # Find dominant frequencies (ignore the 0 frequency / DC offset)
    amplitudes[0] = 0
    top_indices = np.argsort(amplitudes)[-3:][::-1]
    
    print("Dominant Cyclic Frequencies Detected:")
    for idx in top_indices:
        freq = xf[idx]
        if freq > 0:
            period = 1 / freq
            print(f"  Cycle Length: {period:.1f} days (Amplitude: {amplitudes[idx]:.2f})")
    
    # Plot periods from 2 days up to 100 days
    plt.figure(figsize=(12, 5))
    plt.plot(1 / xf[1:], amplitudes[1:], color='purple')
    plt.title("FFT Spectrum: Hidden Cyclic Periods of Jodis")
    plt.xlabel("Period (Days)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('jodi_fft_spectrum.png', dpi=300)
    plt.close()
    print("Saved FFT Spectrum: jodi_fft_spectrum.png")

def copula_dependency(df):
    print("\n--- 4. Copula & Dependency Check ---")
    
    m_nums = df['Morning_number']
    e_nums = df['Evening_number']
    
    # Spearman rank correlation checks for monotonic relationships
    corr = m_nums.corr(e_nums, method='spearman')
    print(f"Spearman Rank Correlation (Morning vs Evening): {corr:.4f}")
    if abs(corr) < 0.05:
        print("  -> The digits appear to be relatively independent.")
    else:
        print("  -> WARNING: There is measurable mathematical dependency between Morning and Evening.")
    
    # Joint distribution heatmap (Empirical Copula)
    joint_counts = pd.crosstab(m_nums, e_nums, normalize='all')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(joint_counts, cmap='coolwarm', annot=True, fmt=".3f", linewidths=.5)
    plt.title("Joint Probability Matrix (Empirical Dependency)")
    plt.xlabel("Evening Digit (0-9)")
    plt.ylabel("Morning Digit (0-9)")
    plt.savefig('jodi_joint_probability.png', dpi=300)
    plt.close()
    print("Saved Joint Probability Matrix: jodi_joint_probability.png")

def main():
    filepath = 'combined_data.csv'
    try:
        df = load_and_prep(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Ensure dataset is in the directory.")
        return
        
    print(f"Loaded {len(df)} draw records. Beginning Deep Jodi Analysis...")
    
    temporal_heatmap(df)
    graph_theory_centrality(df)
    fft_signal_processing(df)
    copula_dependency(df)
    
    print("\n=======================================================")
    print("DEEP ANALYSIS COMPLETE. ALL CHARTS SAVED.")
    print("=======================================================\n")

if __name__ == '__main__':
    main()
