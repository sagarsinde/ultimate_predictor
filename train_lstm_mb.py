import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. Define the Neural Architecture
class SattaLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, num_classes=10):
        super(SattaLSTM, self).__init__()
        # The LSTM Cell
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        # Separate output heads for Morning and Evening
        self.fc_m = nn.Linear(hidden_size, num_classes)
        self.fc_e = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Pass through LSTM
        out, _ = self.lstm(x)
        # Extract the final time-step state
        last_out = out[:, -1, :] 
        # Pass to fully connected output layers
        return self.fc_m(last_out), self.fc_e(last_out)

def create_sequences(features, labels_m, labels_e, seq_length=30):
    xs, ym, ye = [], [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:(i + seq_length)])
        ym.append(labels_m[i + seq_length])
        ye.append(labels_e[i + seq_length])
    return np.array(xs), np.array(ym), np.array(ye)

def main():
    print("--- Phase 7: Deep Learning LSTM Engine (Main Bazar) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Accelerator: {device.type.upper()}")
    
    df = pd.read_csv('main_bazar_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Extract the Vault (Train only on 2019-2025)
    train_df = df[df['Date'] < '2026-01-01'].copy()
    
    raw_m = train_df['Morning_number'].astype(int).values
    raw_e = train_df['Evening_number'].astype(int).values
    
    # 3. One-Hot Encode (Translate numbers into machine language)
    encoder = OneHotEncoder(sparse_output=False, categories=[range(10)])
    m_encoded = encoder.fit_transform(raw_m.reshape(-1, 1))
    e_encoded = encoder.fit_transform(raw_e.reshape(-1, 1))
    
    # Combine to 20-dimensional vector
    daily_vectors = np.hstack([m_encoded, e_encoded])
    
    # 4. Generate 30-Day Memory Sequences
    print("Constructing 30-Day Memory Sequences...")
    seq_length = 30
    X, y_m, y_e = create_sequences(daily_vectors, raw_m, raw_e, seq_length=seq_length)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    ym_tensor = torch.tensor(y_m, dtype=torch.long)
    ye_tensor = torch.tensor(y_e, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, ym_tensor, ye_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 5. Initialize the Brain
    model = SattaLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Train the Neural Network
    epochs = 150
    print("\nIgniting Neural Network Training Loop...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_ym, batch_ye in loader:
            batch_x, batch_ym, batch_ye = batch_x.to(device), batch_ym.to(device), batch_ye.to(device)
            
            # Forward pass
            out_m, out_e = model(batch_x)
            
            # Calculate Loss for both draws
            loss_m = criterion(out_m, batch_ym)
            loss_e = criterion(out_e, batch_ye)
            loss = loss_m + loss_e
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | AI Loss: {total_loss/len(loader):.4f}")
            
    # 7. Save Weights
    print("\nSaving PyTorch Neural Weights to disk...")
    torch.save(model.state_dict(), 'mb_lstm_model.pth')
    print("Done! LSTM Engine is fully trained.")

if __name__ == '__main__':
    main()
