import pandas as pd
from ta.momentum import RSIIndicator


df = pd.read_csv('AUDUSD_5min_Jan2023_Sep2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.drop(columns=['volume','average','barCount','high','low'])
df = df.sort_values('date').reset_index(drop=True)
df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
df = df.dropna()
df = df.reset_index(drop=True)
print(df)

df_train = df[df['date'] <= pd.to_datetime('2024-11-30').tz_localize('UTC')].copy()
df_test = df[df['date'] >= pd.to_datetime('2024-12-01').tz_localize('UTC')].copy()
print("*"*50)
print(df_train)
print("*"*50)
print(df_test)

# Defining upper and lwer bounds for RSI
L = []
U = []

for i in range(20, 21):
    L.append(i)

for j in range(65, 66):
    U.append(j)


def buy_strat(df, l, u):
    df['signal'] = 0
    df.loc[df['RSI'] < l, 'signal'] = 1
    df.loc[df['RSI'] > u, 'signal'] = -1
    df['execution_signal'] = df['signal'].shift(1)
    #df = df[df['signal'] != 0]
    trades = []
    current_position = 0 # 0 = flat, 1 = long, -1 = short

    for index, row in df.iterrows():
        sig = row['execution_signal']
        
        # If we get a Buy signal and we aren't already Long
        if sig == 1 and current_position != 1:
            trades.append(row)
            current_position = 1
            
        # If we get a Sell signal and we aren't already Short
        elif sig == -1 and current_position != -1:
            trades.append(row)
            current_position = -1

    df_trades = pd.DataFrame(trades).reset_index(drop=True)
    return df_trades

for l in L:
    for u in U:
        df_trades = buy_strat(df_train, l, u)
        print(df_trades)


