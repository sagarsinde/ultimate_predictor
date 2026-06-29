import pandas as pd
from feature_engineering_mb import generate_features

def main():
    print("--- Phase 1: Feature Store Pipeline ---")
    input_file = 'main_bazar_dataset.csv'
    output_file = 'main_bazar_feature_store.csv'
    
    print(f"Loading raw dataset: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Run parser first.")
        return
        
    print(f"Raw rows loaded: {len(df)}")
    
    # Generate the perfectly causal feature state
    feature_df = generate_features(df)
    
    print(f"Feature calculation complete. Final rows: {len(feature_df)}")
    print(f"Total features generated: {len(feature_df.columns)}")
    
    print("Saving strictly-causal Data Lake to disk...")
    feature_df.to_csv(output_file, index=False)
    print(f"Success! Master Feature Store saved to: {output_file}")
    
if __name__ == '__main__':
    main()
