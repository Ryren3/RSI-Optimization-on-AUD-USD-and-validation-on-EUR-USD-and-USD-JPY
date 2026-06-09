import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

# Configuration Constants
RSI_WINDOW = 14
BARS_PER_YEAR = 288 * 252   # 5-minute bars
TRANSACTION_COST = 0.0001  # 1 pip spread
TRAIN_END_DATE = '2024-11-30'
TEST_START_DATE = '2024-12-01'

# Load data
df = pd.read_csv('AUDUSD_5min_Jan2023_Sep2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.drop(columns=['volume','average','barCount'])
df = df.sort_values('date').reset_index(drop=True)

# Split data BEFORE calculating RSI to avoid data leakage
train_end_date = pd.to_datetime(TRAIN_END_DATE).tz_localize('UTC')
test_start_date = pd.to_datetime(TEST_START_DATE).tz_localize('UTC')

df_train = df[df['date'] <= train_end_date].copy()
df_test = df[df['date'] >= test_start_date].copy()

# Calculate RSI separately for each set
df_train['RSI'] = RSIIndicator(close=df_train['close'], window=RSI_WINDOW).rsi()
df_test['RSI'] = RSIIndicator(close=df_test['close'], window=RSI_WINDOW).rsi()

# Remove NaN rows
df_train = df_train.dropna()
df_test = df_test.dropna()

print("Full Dataset:")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"Total bars: {len(df)}")

print(f"\nTraining Set: {df_train['date'].min()} to {df_train['date'].max()} ({len(df_train)} bars)")
print(f"Testing Set: {df_test['date'].min()} to {df_test['date'].max()} ({len(df_test)} bars)")

# Parameter ranges for grid search
L = list(range(20, 46))
U = list(range(65, 91))


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate maximum drawdown from cumulative returns series
    """
    wealth = (1 + cumulative_returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown


def backtest_rsi_strategy(data, lower, upper):
    """
    Backtest RSI mean-reversion strategy
    
    Returns: cumulative_return, sharpe_ratio, max_drawdown, sortino_ratio
    """
    df = data.copy()

    # Generate signals
    df['signal'] = 0
    df.loc[df['RSI'] < lower, 'signal'] = 1   # Oversold -> Buy
    df.loc[df['RSI'] > upper, 'signal'] = -1  # Overbought -> Sell

    # Execute next bar (avoid look-ahead bias)
    df['position'] = df['signal'].replace(0, np.nan).shift(1) 
    df['position'] = df['position'].ffill().fillna(0)
  
    # Returns: Open-to-open (enter at next bar's open)
    df['return'] = df['open'].shift(-1) / df['open'] - 1
    df['strategy_return'] = df['position'] * df['return']

    # Add transaction costs (1 pip)
    df['trade'] = ((df['position'] != df['position'].shift(1)) & 
                   (df['position'].shift(1).notna())).astype(int)
    df['strategy_return'] = df['strategy_return'] - (df['trade'] * TRANSACTION_COST)
    
    df['strategy_return'] = df['strategy_return'].fillna(0)

    # Performance metrics
    cumulative_return = (1 + df['strategy_return']).prod() - 1

    # Calculate annualized mean return
    mu = df['strategy_return'].mean() * BARS_PER_YEAR
    
    # Sharpe ratio (annualized)
    sigma = df['strategy_return'].std() * np.sqrt(BARS_PER_YEAR)
    sharpe = mu / sigma if sigma > 0 else 0

    # Sortino ratio (annualized) - CORRECTED VERSION
    # Use semi-deviation: downside deviation from target return (0)
    target_return = 0
    downside_diff = np.minimum(df['strategy_return'] - target_return, 0)
    downside_deviation = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(BARS_PER_YEAR)
    
    if downside_deviation > 0:
        sortino = mu / downside_deviation
    else:
        sortino = np.inf if mu > 0 else 0
    
    # Maximum drawdown
    max_dd = calculate_max_drawdown(df['strategy_return'])

    if df['position'].iloc[-1] != 0:
        last_return = (df['close'].iloc[-1] / df['open'].iloc[-1]) - 1
        df.loc[df.index[-1], 'strategy_return'] += df.loc[df.index[-1], 'position'] * last_return


    return cumulative_return, sharpe, max_dd, sortino


# ============================================
# PHASE 1: OPTIMIZATION ON TRAINING DATA
# ============================================
print("\n" + "="*60)
print("PHASE 1: PARAMETER OPTIMIZATION (Training Data 2023-2024)")
print("="*60)

train_results = []

for lower in L:
    for upper in U:
        if lower < upper:
            cum_return, sharpe, max_drawdown, sortino = backtest_rsi_strategy(df_train, lower, upper)
            train_results.append({
                'lower': lower,
                'upper': upper,
                'cumulative_return': cum_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino
            })

train_results_df = pd.DataFrame(train_results)

# Find best parameters on training data
best_params_train = train_results_df.loc[train_results_df['cumulative_return'].idxmax()]
print("\nBest Parameters (Training - by Cumulative Return):")
print(f"Lower RSI: {best_params_train['lower']}, Upper RSI: {best_params_train['upper']}")
print(f"Cumulative Return: {best_params_train['cumulative_return']:.2%}")
print(f"Sharpe Ratio: {best_params_train['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {best_params_train['sortino_ratio']:.2f}")
print(f"Max Drawdown: {best_params_train['max_drawdown']:.2%}")

best_sharpe_train = train_results_df.loc[train_results_df['sharpe_ratio'].idxmax()]
print("\nBest Parameters (Training - by Sharpe Ratio):")
print(f"Lower RSI: {best_sharpe_train['lower']}, Upper RSI: {best_sharpe_train['upper']}")
print(f"Cumulative Return: {best_sharpe_train['cumulative_return']:.2%}")
print(f"Sharpe Ratio: {best_sharpe_train['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {best_sharpe_train['sortino_ratio']:.2f}")
print(f"Max Drawdown: {best_sharpe_train['max_drawdown']:.2%}")

# Also check best by Sortino
best_sortino_train = train_results_df.loc[train_results_df['sortino_ratio'].idxmax()]
print("\nBest Parameters (Training - by Sortino Ratio):")
print(f"Lower RSI: {best_sortino_train['lower']}, Upper RSI: {best_sortino_train['upper']}")
print(f"Cumulative Return: {best_sortino_train['cumulative_return']:.2%}")
print(f"Sharpe Ratio: {best_sortino_train['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {best_sortino_train['sortino_ratio']:.2f}")
print(f"Max Drawdown: {best_sortino_train['max_drawdown']:.2%}")

print("\nTop 10 Strategies (Training Data):")
print(train_results_df.nlargest(10, 'cumulative_return')[
    ['lower', 'upper', 'cumulative_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown']
])

# ============================================
# PHASE 2: VALIDATION ON TEST DATA
# ============================================
print("\n" + "="*60)
print("PHASE 2: OUT-OF-SAMPLE TESTING (Test Data 2025)")
print("="*60)

# Extract best parameters
best_lower = int(best_params_train['lower'])
best_upper = int(best_params_train['upper'])

# Test on out-of-sample data
test_cum_return, test_sharpe, test_max_dd, test_sortino = backtest_rsi_strategy(
    df_test, best_lower, best_upper
)

print(f"\nOut-of-Sample Test Results (RSI {best_lower}/{best_upper}):")
print(f"Cumulative Return: {test_cum_return:.2%}")
print(f"Sharpe Ratio: {test_sharpe:.2f}")
print(f"Sortino Ratio: {test_sortino:.2f}")
print(f"Max Drawdown: {test_max_dd:.2%}")

# ============================================
# PHASE 3: COMPARISON
# ============================================
print("\n" + "="*60)
print("PERFORMANCE COMPARISON: Training vs Testing")
print("="*60)

comparison = pd.DataFrame({
    'Metric': ['Cumulative Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
    'Training (2023-2024)': [
        f"{best_params_train['cumulative_return']:.2%}",
        f"{best_params_train['sharpe_ratio']:.2f}",
        f"{best_params_train['sortino_ratio']:.2f}",
        f"{best_params_train['max_drawdown']:.2%}"
    ],
    'Testing (2025)': [
        f"{test_cum_return:.2%}",
        f"{test_sharpe:.2f}",
        f"{test_sortino:.2f}",
        f"{test_max_dd:.2%}"
    ],
    'Difference': [
        f"{(test_cum_return - best_params_train['cumulative_return']):.2%}",
        f"{(test_sharpe - best_params_train['sharpe_ratio']):.2f}",
        f"{(test_sortino - best_params_train['sortino_ratio']):.2f}",
        f"{(test_max_dd - best_params_train['max_drawdown']):.2%}"
    ]
})

print("\n", comparison.to_string(index=False))

# Performance degradation check
return_degradation = (test_cum_return - best_params_train['cumulative_return']) / abs(best_params_train['cumulative_return']) * 100
sharpe_degradation = (test_sharpe - best_params_train['sharpe_ratio']) / abs(best_params_train['sharpe_ratio']) * 100 if best_params_train['sharpe_ratio'] != 0 else 0

print("\n" + "="*60)
print("VALIDATION ASSESSMENT")
print("="*60)
print(f"Return Degradation: {return_degradation:.1f}%")
print(f"Sharpe Degradation: {sharpe_degradation:.1f}%")

if test_cum_return > 0 and test_sharpe > 0:
    if return_degradation > -30:
        print("✓ PASSED: Strategy shows reasonable out-of-sample performance")
    else:
        print("⚠ WARNING: Significant performance degradation detected")
else:
    print("✗ FAILED: Strategy performs poorly on out-of-sample data")

# ============================================
# OPTIONAL: Test alternative parameter choices
# ============================================
print("\n" + "="*60)
print("BONUS: Testing Alternative Parameters on Test Data")
print("="*60)

# Test top 5 parameter combinations from training
top_5_params = train_results_df.nlargest(5, 'cumulative_return')

alternative_results = []
for idx, row in top_5_params.iterrows():
    lower = int(row['lower'])
    upper = int(row['upper'])
    test_return, test_sharpe_alt, test_dd, test_sort = backtest_rsi_strategy(
        df_test, lower, upper
    )
    
    alternative_results.append({
        'lower': lower,
        'upper': upper,
        'train_return': row['cumulative_return'],
        'test_return': test_return,
        'train_sharpe': row['sharpe_ratio'],
        'test_sharpe': test_sharpe_alt,
        'train_sortino': row['sortino_ratio'],
        'test_sortino': test_sort,
        'test_max_dd': test_dd
    })

alt_df = pd.DataFrame(alternative_results)
print("\nTop 5 Training Parameters - Test Performance:")
print(alt_df.to_string(index=False))

# Find which performed best on test data
best_test_idx = alt_df['test_return'].idxmax()
best_on_test = alt_df.iloc[best_test_idx]
print(f"\nBest performer on test data: RSI {int(best_on_test['lower'])}/{int(best_on_test['upper'])}")
print(f"Test Return: {best_on_test['test_return']:.2%}")
print(f"Test Sharpe: {best_on_test['test_sharpe']:.2f}")
print(f"Test Sortino: {best_on_test['test_sortino']:.2f}")

print("\n" + "="*60)
print("NOTES ON CORRECTED VERSION")
print("="*60)
print("✓ Fixed: Sortino ratio now properly calculated using semi-deviation")
print("✓ Fixed: Sortino ratio now properly stored and displayed")
print("✓ Fixed: RSI calculated separately on train/test to avoid data leakage")
print("✓ Fixed: Transaction cost logic improved")
print("✓ Added: Magic numbers converted to named constants")