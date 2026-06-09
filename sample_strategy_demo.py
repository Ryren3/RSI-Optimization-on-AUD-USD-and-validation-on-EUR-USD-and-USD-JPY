import pandas as pd
import numpy as np

# Simple demo showing how signals -> positions -> returns -> strategy_return
dates = pd.date_range('2024-01-01', periods=10, freq='D')
price = np.array([100, 101, 100, 102, 103, 101, 102, 104, 103, 105], dtype=float)

# Calculate open-to-open returns (pct change)
returns = pd.Series(price).pct_change().fillna(0)

# Example signals generated at bar close (0 hold, 1 long, -1 short)
signals = pd.Series([0, 1, 0, 0, -1, 0, 1, 0, 0, 0])

# Convert signals -> positions: replace 0 with NaN, shift to next bar (enter next open), forward-fill to hold
position = signals.replace(0, np.nan).shift(1).ffill().fillna(0)

# Strategy returns = position * returns
strategy_return = position * returns

# Cumulative equity starting from 1.0
equity = (1 + strategy_return).cumprod()

df = pd.DataFrame({
    'date': dates,
    'price': price,
    'return': returns.round(6),
    'signal': signals,
    'position': position,
    'strategy_return': strategy_return.round(6),
    'equity': equity.round(6)
})

print('\nDemo: signals -> position -> returns -> strategy_return -> equity')
print(df.to_string(index=False))
print(f"\nFinal cumulative return (strategy): {equity.iloc[-1]-1:.2%}")
