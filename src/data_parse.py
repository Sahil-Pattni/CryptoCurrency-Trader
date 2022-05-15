# %%
import pandas as pd
import numpy as np

df = pd.read_csv('../data/ADA.csv')
# Sort by date (ascending)
df.sort_values(by='unix', inplace=True, ascending=True)
# Drop NaN columns
df.dropna(axis=1, inplace=True)
# --- Add SMAs --- #
df['SMA_7'] = df.close.rolling(7).mean()
df['SMA_25'] = df.close.rolling(25).mean()
df['SMA_99'] = df.close.rolling(99).mean()
# %%
df.dropna(inplace=True)

# Round SMA columns
round_digit = 5
df['SMA_7'] = df['SMA_7'].round(round_digit)
df['SMA_25'] = df['SMA_25'].round(round_digit)
df['SMA_99'] = df['SMA_99'].round(round_digit)

# Valid buy if SMA_7 > SMA_25 & SMA 7 > SMA 99
df['buy'] = np.where((df.SMA_7 > df.SMA_25) & (df.SMA_7 > df.SMA_99), 1, 0)
# %%
# Reset index
df.reset_index(inplace=True, drop=True)
# Export to Feather
df.to_feather('../data/ADA.feather')
# %%
