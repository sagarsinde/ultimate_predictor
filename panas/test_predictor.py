import pandas as pd
from panas.model import HybridPanaPredictor

print("Loading Madhur dataset...")
df = pd.read_csv('data/madhur_dataset.csv')

print("Training Hybrid Pana Predictor (Morning)...")
predictor = HybridPanaPredictor()
predictor.fit(df, is_morning=True)

print("Predicting Top 15 Morning Panas for Next Draw...")
top_panas = predictor.predict_220(df, is_morning=True, top_n=15)

print("\n=== TOP 15 PREDICTED MORNING PANAS ===")
for i, p in enumerate(top_panas):
    print(f"{i+1}. Pana: {p['Pana']} | Digit: {p['Digit']} | Type: {p['Type']} | Prob: {p['Probability']:.4f}%")
    
print("\nDone.")
