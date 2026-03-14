# NexusAI Deep Learning Trainer (Python Bridge)
# This script is designed to take historical price data exported from MT5,
# train a Deep Multi-Layer Perceptron (MLP), and output the MQL5 array weights.

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Architecture matches NexusAI_Pro.mq5
NUM_INPUTS = 8
NUM_HIDDEN_1 = 6
NUM_HIDDEN_2 = 4
NUM_OUTPUTS = 1

def generate_dummy_data(samples=10000):
    """
    Generates dummy financial data for testing the pipeline 
    before connecting a real MT5 CSV export.
    Features: ATR_Norm, MACD_Hist, RSI, Dist_UP, Dist_DN, Momentum, MACD_Main, Volatility_Exp
    """
    np.random.seed(42)
    X = np.random.randn(samples, NUM_INPUTS)
    
    # Create some dummy rules for a binary target (1 = Buy, 0 = Sell)
    # E.g., if Momentum > 0 and RSI < 0 (oversold) -> Buy
    y = ((X[:, 5] > 0) & (X[:, 2] < -0.5)).astype(int)
    
    # Add some noise
    flip_mask = np.random.rand(samples) < 0.1
    y[flip_mask] = 1 - y[flip_mask]
    
    return pd.DataFrame(X), pd.Series(y)

import os

# ... (architecture defines)

def train_and_export():
    print("NexusAI: Starting Neural Network Training...")
    
    csv_path = r"C:\Users\skark\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NexusAI_Export_EURUSD.csv"
    
    if os.path.exists(csv_path):
        print(f"Loading real market data from: {csv_path}")
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['Target']).values
        y = df['Target'].values
        # Ensure we only take the last ROLLING_WINDOW if we want, or use everything.
        print(f"Dataset Size: {len(df)} samples.")
    else:
        print("Real data CSV not found. Running with mock data for pipeline verification...")
        X, y = generate_dummy_data()
        X = X.values
        y = y.values
    
    # Even if Z-Score normalized in MQL5, we scale here to ensure the training batch is stable.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tanh = np.tanh(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tanh, y, test_size=0.2, random_state=42)
    
    # Building the exact architecture as our MQL5 EA
    mlp = MLPClassifier(
        hidden_layer_sizes=(NUM_HIDDEN_1, NUM_HIDDEN_2),
        activation='tanh', 
        solver='adam',
        max_iter=2000, # Increased for real data
        random_state=42
    )
    
    print("Training the Deep MLP Model...")
    mlp.fit(X_train, y_train)
    
    score = mlp.score(X_test, y_test)
    print(f"Model Accuracy on Test Set: {score:.2%}")
    
    # --- Weight Extraction for MQL5 ---
    # Scikit-learn stores weights in mlp.coefs_ and biases in mlp.intercepts_
    W1 = mlp.coefs_[0]
    B1 = mlp.intercepts_[0]
    W2 = mlp.coefs_[1]
    B2 = mlp.intercepts_[1]
    W3 = mlp.coefs_[2]
    B3 = mlp.intercepts_[2]
    
    print("\n--- MQL5 EXPORT CODE ---")
    print("Copy and paste this initialize function into NexusAI_Pro.mq5:\n")
    
    mql_code = "void InitWeights()\n{\n"
    
    # Layer 1
    for i in range(NUM_INPUTS):
        for j in range(NUM_HIDDEN_1):
            mql_code += f"   W1[{i}][{j}] = {W1[i][j]:.5f};\n"
    for j in range(NUM_HIDDEN_1):
        mql_code += f"   B1[{j}] = {B1[j]:.5f};\n"
        
    mql_code += "\n"
    # Layer 2
    for i in range(NUM_HIDDEN_1):
        for j in range(NUM_HIDDEN_2):
            mql_code += f"   W2[{i}][{j}] = {W2[i][j]:.5f};\n"
    for j in range(NUM_HIDDEN_2):
        mql_code += f"   B2[{j}] = {B2[j]:.5f};\n"
        
    mql_code += "\n"
    # Output Layer
    for i in range(NUM_HIDDEN_2):
        for j in range(NUM_OUTPUTS):
            mql_code += f"   W3[{i}][{j}] = {W3[i][j]:.5f};\n"
    for j in range(NUM_OUTPUTS):
        mql_code += f"   B3[{j}] = {B3[j]:.5f};\n"
        
    mql_code += "}\n"
    
    print(mql_code)
    
    # Save code to file for easy copy-pasting
    with open("mql5_weights.txt", "w") as f:
        f.write(mql_code)
    print("Export saved to mql5_weights.txt")

if __name__ == "__main__":
    train_and_export()
