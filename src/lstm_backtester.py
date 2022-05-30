# %%
from sklearn.preprocessing import MinMaxScaler
import wandb
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from lstm_backend import LSTMRNN
# %%
# --- GLOBAL CONSTANTS --- #
CONFIG_FILEPATH = '../data/btc_lstm_config.json'
MODEL_STATE_FILEPATH = '../data/btc_lstm.pt'
DATASET_FILEPATH = '../data/BTC.feather'

# --- LOAD MODEL --- #
config = json.load(open(CONFIG_FILEPATH))
model = LSTMRNN(
    num_classes=1,
    input_size=1,
    hidden_size=config['hidden_layers'],
    num_layers=config['num_layers']
)
model.load_state_dict(torch.load(MODEL_STATE_FILEPATH, map_location='cpu'))
# Set to eval mode
model.eval()

def predict(sample, scaler):
    with torch.no_grad():
        # Predict
        pred = model(torch.Tensor(np.array([X[0]])))
        # De-scale to original value and return
        return scaler.inverse_transform(pred).item()
# %%
# --- PREPARE DATA --- #
# Load dataframe
df = pd.read_feather(DATASET_FILEPATH)
# Min-Max scale closing values
scaler = MinMaxScaler()
closing_values = scaler.fit_transform(df['close'].values.reshape(-1, 1))
X, y, dates = [], [], []
# Split into rolling windows of 110 hours
L = 110  # Sequence Length, evenly divides 41,470
for i in range(df.shape[0]-L):
    X.append(df.iloc[i:(i+L)].close)
    y.append(df.iloc[i+L].close)
    dates.append(df.iloc[i+L].date)

# --- TRAIN / VAL / TEST split --- #
X = np.array(X)
y = np.array(y)
# Reshape 
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], 1)


# %%
# ---- BACKTEST ---- #
profit_limit = 1/100
loss_limit = 1-(2/100)
assets = {'USDT': 1000, 'BTC': 0}
positions = []  # List of buys and sells
net_worth = []  # List of net worths

# Iterate through windows
for i in tqdm(range(X.shape[0])):
    current_price = y[i][0]
    # Predict next price
    next_price = predict(X[i], scaler)

    # If last index, liquidate all crypto_asset
    if i == X.shape[0]-1:
        assets['USDT'] += assets['BTC'] * current_price
        assets['BTC'] = 0
        positions.append([current_price, 'SELL', dates[i]])
        break

    # If any USDT to buy
    if assets['USDT'] > 0:
        # If next predicted price exceeds profit limit, buy
        if next_price >= profit_limit * current_price:
            assets['BTC'] += assets['USDT'] / current_price
            assets['USDT'] = 0
            positions.append([current_price, 'BUY', dates[i]])
    
    # If any BTC to sell
    elif assets['BTC'] > 0:
        # If current price exceeds last buy price by profit limit
        profit_met = current_price >= loss_limit * positions[-1][0]
        # If current price is below last buy price by loss limit
        loss_met = current_price <= loss_limit * positions[-1][0]
        if profit_met or loss_met:
            assets['USDT'] += assets['BTC'] * current_price
            assets['BTC'] = 0
            positions.append([current_price, 'SELL', dates[i]])
    
    # Calculate net worth
    net_worth.append(assets['USDT'] + assets['BTC'] * current_price)


# %%
for currency, value in assets.items():
    print(f'{currency}: {value:,.6f}')
# %%
