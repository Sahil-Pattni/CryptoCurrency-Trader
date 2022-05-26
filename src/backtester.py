# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

CRYPTO = 'BTC'
df_ = pd.read_feather(f'../data/{CRYPTO}.feather')
train = df_.iloc[:int(len(df_)*.5)]
test = df_.iloc[int(len(df_)*.5):]
# %%


def backtest(df, initial_balance, buys_required, profit_limit, loss_limit, plot_fig=False):
    assets = {'USDT': initial_balance, CRYPTO: 0}
    assert(assets['USDT'] == 1000)
    positions = []
    net_worth = []
    successive_buys_seen = 0
    for index, row in df.iterrows():
        # If last index, liquidate all crypto_asset
        if index == df.index[-1]:
            usdt_earned = assets[CRYPTO] * row.close
            assets['USDT'] += usdt_earned
            assets[CRYPTO] = 0
            break
        # Only scan for valid buys if USDT to buy
        if assets['USDT'] > 0:
            if row.buy == 1:
                successive_buys_seen += 1
                # If 3 buys in a row, buy crypto_asset
                if successive_buys_seen == buys_required:
                    assets[CRYPTO] += assets['USDT'] / row.close
                    assets['USDT'] = 0
                    successive_buys_seen = 0
                    positions.append([row.close, 'BUY'])
            # If no valid buy, reset counter
            elif row.buy == 0 and successive_buys_seen > 0:
                successive_buys_seen = 0
        # If no USDT, scan for 2% profit to sell
        else:
            profit_met = row.close >= (1+(profit_limit/100)) * positions[-1][0]
            stop_limit_met = row.close <= (
                1-(loss_limit/100)) * positions[-1][0]
            if profit_met or stop_limit_met:
                assets['USDT'] += assets[CRYPTO] * row.close
                assets[CRYPTO] = 0
                profit_percentage = (
                    row.close - positions[-1][0]) / positions[-1][0] * 100
                positions.append(
                    [row.close, 'SELL', f'{profit_percentage:.2f}%'])

        # Add to net worth
        net_worth.append(assets['USDT'] + assets[CRYPTO] * row.close)

    # Round off assets
    assets['USDT'] = round(assets['USDT'], 4)
    assets[CRYPTO] = round(assets[CRYPTO], 4)

    if plot_fig:
        spacer = 50
        print(f'Simulation complete.')
        _, ax = plt.subplots(figsize=(16, 9))
        ax.plot(df.date[1:][::spacer], net_worth[::spacer], '--',
                color='0.5', label='Net Worth (USDT)')
        ax.set_ylabel('USDT')
        ax.set_xlabel('2022')
        ax.set_xticks(df.date[1:][::spacer*100])
        ax.set_xticklabels(df.date[1:][::spacer*100], rotation=45)
        ax.set_title(f'Backtest {CRYPTO}/USDT')
        ax.legend()
        plt.show()

    return assets['USDT'], assets[CRYPTO], positions, net_worth


# %%
buy_range = list(range(1, 5))
buy_range.extend(list(range(5, 55, 5)))

initial_balance = 1000 
logs = []
for buys_required in tqdm(buy_range):
    for profit_limit in np.arange(0.25, 2, 0.25):
        for loss_limit in np.arange(0.25, 2, 0.25):
            usdt, crypto_asset, positions, net_worth = backtest(
                train, initial_balance, buys_required, profit_limit, loss_limit)
            logs.append([buys_required, profit_limit, loss_limit, usdt])


# %%

def performance_report(log, USDT=1000):
    return f'RETURN: USDT {log[-1]:,.2f} ({(log[-1]-{USDT})/{USDT} * 100:,.2f})%, BUYS: {log[0]}, PROFI LIMITT: {log[1]:,.2f}%, LOSS LIMIT: {log[2]:,.2f}%'
logs.sort(key=lambda x: x[-1], reverse=True)
print(f'Training:')
print(performance_report(logs[0]))
usdt, _, positions, net_worth = backtest(
    test, 1000, logs[0][0], logs[0][1], logs[0][2], plot_fig=True)# %%
net_worth
# %%
