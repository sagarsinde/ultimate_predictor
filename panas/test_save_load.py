import pandas as pd
from panas.model import HybridPanaPredictor

print("1. Loading Data...")
df = pd.read_csv('data/madhur_dataset.csv')

print("\n2. Training New Model...")
predictor = HybridPanaPredictor()
predictor.fit(df, is_morning=True)

print("\n3. Saving Model to Disk...")
predictor.save_models(dir_path="trained_models/madhur_morning")

print("\n4. Loading Model from Disk into a NEW Predictor...")
loaded_predictor = HybridPanaPredictor()
loaded_predictor.load_models(dir_path="trained_models/madhur_morning")

print("\n5. Testing Prediction with Loaded Model...")
top_panas = loaded_predictor.predict_220(df, is_morning=True, top_n=3)
for p in top_panas:
    print(f"Pana: {p['Pana']} | Prob: {p['Probability']:.4f}%")

print("\nSuccessfully tested Save & Load functionality!")
