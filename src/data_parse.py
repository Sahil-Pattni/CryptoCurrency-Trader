# %%
import pandas as pd
import numpy as np

df = pd.read_csv('data/ADA.csv')
# Sort by date (ascending)
df.sort_values(by='unix', inplace=True, ascending=True)

# --- Add SMAs --- #
df['SMA_7'] = df.close.rolling(7).mean()
df['SMA_25'] = df.close.rolling(25).mean()
df['SMA_99'] = df.close.rolling(99).mean()

# Drop NaN values
df.dropna(inplace=True)
# %%
# Reset index
df.reset_index(inplace=True, drop=True)
# Export to Feather
df.to_feather('data/ADA.feather')
# %%
