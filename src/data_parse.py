# %%
import pandas as pd
import numpy as np

CRYPTO = 'BTC'

df = pd.read_csv(f'../data/{CRYPTO}.csv')
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
# Reset index
df.reset_index(inplace=True, drop=True)

# %%
def change_date(date):
    _date = date.split(' ')
    time = _date[1].split('-')
    if len(time) != 2:
        return date
    if time[1] == 'PM':
        time[0] = (int(time[0]) + 12) % 24
    time[0] = int(time[0])
    return f'{_date[0]} + {time[0]:.0f}:00'

# df.date = df.date.apply(change_date)
# %%
# df['date'] = pd.to_datetime(df['date'])
# Export to Feather
df.to_feather(f'../data/{CRYPTO}.feather')