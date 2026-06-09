import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta

# Load data
df = pd.read_csv('AUDUSD_5min_Jan2023_Sep2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.drop(columns=['volume','average','barCount'])
df = df.sort_values('date').reset_index(drop=True)
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
df = df.dropna()
df = df.reset_index(drop=True)

print("="*80)
print("WALK-FORWARD OPTIMIZATION FOR RSI STRATEGY")
print("="*80)
print(f"Dataset: {df['date'].min()} to {df['date'].max()}")
print(f"Total bars: {len(df)}")
print(f"Total days: {(df['date'].max() - df['date'].min()).days}")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot AUD/USD price
ax1.plot(df['date'], df['close'], color='blue', linewidth=0.8)
ax1.axvspan('2022-12-27', '2024-11-30', alpha=0.2, color='orange', 
            label='Training Period (Downward Trend)')
ax1.axvspan('2024-12-01', '2025-09-30', alpha=0.2, color='green', 
            label='Testing Period (Upward Trend)')
ax1.set_title('AUD/USD Price: Regime Shift Between Training and Testing')
ax1.legend()

# Plot RSI
ax2.plot(df['date'], df['rsi'], color='purple', linewidth=0.8)
ax2.axhline(y=21, color='green', linestyle='--', label='Lower Threshold (21)')
ax2.axhline(y=77, color='red', linestyle='--', label='Upper Threshold (77)')
ax2.set_title('RSI with Optimised Thresholds')
ax2.legend()

plt.tight_layout()
plt.savefig('regime_shift.png', dpi=300)


# ============================================
# CONFIGURATION
# ============================================
L = list(range(20, 48, 3))
U = list(range(65, 92, 3))

TRAIN_MONTHS = 6
TEST_MONTHS = 3
STEP_MONTHS = 3

print(f"\nWalk-Forward Settings:")
print(f"  Training period: {TRAIN_MONTHS} months")
print(f"  Testing period: {TEST_MONTHS} months")
print(f"  Step size: {STEP_MONTHS} months")
print(f"  Parameter grid: Lower {L[0]}-{L[-1]}, Upper {U[0]}-{U[-1]}")

# ============================================
# HELPER FUNCTIONS
# ============================================
def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown"""
    wealth = (1 + cumulative_returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return drawdown.min()


def backtest_rsi_strategy(data, lower, upper):
    """Backtest RSI strategy on given data"""
    df_copy = data.copy()

    # Generate signals
    df_copy['signal'] = 0
    df_copy.loc[df_copy['rsi'] < lower, 'signal'] = 1
    df_copy.loc[df_copy['rsi'] > upper, 'signal'] = -1

    # Execute next bar
    df_copy['position'] = df_copy['signal'].replace(0, np.nan).shift(1)
    df_copy['position'] = df_copy['position'].ffill().fillna(0)

    # Calculate returns
    df_copy['return'] = df_copy['open'].shift(-1) / df_copy['open'] - 1
    df_copy['strategy_return'] = df_copy['position'] * df_copy['return']

    # Transaction costs (1 pip)
    TRANSACTION_COST = 0.0001
    df_copy['trade'] = ((df_copy['position'] != df_copy['position'].shift(1)) &
                     df_copy['position'].shift(1).notna()).astype(int)

    df_copy['strategy_return'] = df_copy['strategy_return'] - (df_copy['trade'] * TRANSACTION_COST)
    df_copy['strategy_return'] = df_copy['strategy_return'].fillna(0)

    # ---- Metrics ----
    cumulative_return = (1 + df_copy['strategy_return']).prod() - 1

    mu = df_copy['strategy_return'].mean() * 288*252
    sigma = df_copy['strategy_return'].std() * np.sqrt(288*252)
    sharpe = mu / sigma if sigma > 0 else 0

    # Sortino: penalise only returns below 0 (semi-deviation, not std of negatives)
    downside_diff = np.minimum(df_copy['strategy_return'], 0)
    downside_deviation = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(288*252)
    sortino = mu / downside_deviation if downside_deviation > 0 else 0

    max_dd = calculate_max_drawdown(df_copy['strategy_return'])

    return {
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'strategy_returns': df_copy['strategy_return']
    }


def optimize_parameters(train_data, param_grid_L, param_grid_U, metric='sharpe_ratio'):
    """Find best parameters on training data"""
    results = []

    for lower in param_grid_L:
        for upper in param_grid_U:
            if lower < upper:
                metrics = backtest_rsi_strategy(train_data, lower, upper)
                results.append({
                    'lower': lower,
                    'upper': upper,
                    'cumulative_return': metrics['cumulative_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'sortino_ratio': metrics['sortino_ratio'],
                    'max_drawdown': metrics['max_drawdown']
                })

    results_df = pd.DataFrame(results)

    if metric == 'cumulative_return':
        best = results_df.loc[results_df['cumulative_return'].idxmax()]
    elif metric == 'sortino_ratio':
        best = results_df.loc[results_df['sortino_ratio'].idxmax()]
    else:
        best = results_df.loc[results_df['sharpe_ratio'].idxmax()]

    return int(best['lower']), int(best['upper']), results_df


def get_date_windows(start_date, end_date, train_months, test_months, step_months):
    """Generate train/test date windows for walk-forward"""
    windows = []
    current_start = start_date

    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        windows.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })

        current_start = current_start + pd.DateOffset(months=step_months)

    return windows


# ============================================
# WALK-FORWARD OPTIMIZATION
# ============================================
print("\n" + "="*80)
print("EXECUTING WALK-FORWARD OPTIMIZATION")
print("="*80)

windows = get_date_windows(
    df['date'].min(),
    df['date'].max(),
    TRAIN_MONTHS,
    TEST_MONTHS,
    STEP_MONTHS
)

print(f"\nTotal walk-forward windows: {len(windows)}")

wfo_results = []
all_test_returns = []

for i, window in enumerate(windows):
    print(f"\n{'='*80}")
    print(f"Window {i+1}/{len(windows)}")
    print(f"{'='*80}")

    train_data = df[(df['date'] >= window['train_start']) &
                    (df['date'] < window['train_end'])].copy()
    test_data  = df[(df['date'] >= window['test_start']) &
                    (df['date'] < window['test_end'])].copy()

    print(f"Train: {window['train_start'].date()} to {window['train_end'].date()} ({len(train_data)} bars)")
    print(f"Test:  {window['test_start'].date()} to {window['test_end'].date()} ({len(test_data)} bars)")

    if len(train_data) < 1000 or len(test_data) < 500:
        print("⚠️  Insufficient data, skipping window")
        continue

    # Optimize on training data
    print("\nOptimizing parameters on training data...")
    best_lower, best_upper, train_results = optimize_parameters(
        train_data, L, U, metric='sharpe_ratio'
    )

    train_best = train_results.loc[train_results['sharpe_ratio'].idxmax()]
    print(f"Best parameters: RSI {best_lower}/{best_upper}")
    print(f"  Train Return:  {train_best['cumulative_return']:.2%}")
    print(f"  Train Sharpe:  {train_best['sharpe_ratio']:.2f}")
    print(f"  Train Sortino: {train_best['sortino_ratio']:.2f}")
    print(f"  Train Max DD:  {train_best['max_drawdown']:.2%}")

    # Test on out-of-sample data
    print("\nTesting on out-of-sample data...")
    test_metrics = backtest_rsi_strategy(test_data, best_lower, best_upper)

    print(f"Out-of-sample results:")
    print(f"  Test Return:   {test_metrics['cumulative_return']:.2%}")
    print(f"  Test Sharpe:   {test_metrics['sharpe_ratio']:.2f}")
    print(f"  Test Sortino:  {test_metrics['sortino_ratio']:.2f}")
    print(f"  Test Max DD:   {test_metrics['max_drawdown']:.2%}")

    return_degradation = ((test_metrics['cumulative_return'] - train_best['cumulative_return']) /
                          abs(train_best['cumulative_return']) * 100) if train_best['cumulative_return'] != 0 else 0
    print(f"  Return Degradation: {return_degradation:.1f}%")

    wfo_results.append({
        'window': i+1,
        'train_start': window['train_start'],
        'train_end': window['train_end'],
        'test_start': window['test_start'],
        'test_end': window['test_end'],
        'best_lower': best_lower,
        'best_upper': best_upper,
        'train_return': train_best['cumulative_return'],
        'train_sharpe': train_best['sharpe_ratio'],
        'train_sortino': train_best['sortino_ratio'],
        'train_max_dd': train_best['max_drawdown'],
        'test_return': test_metrics['cumulative_return'],
        'test_sharpe': test_metrics['sharpe_ratio'],
        'test_sortino': test_metrics['sortino_ratio'],
        'test_max_dd': test_metrics['max_drawdown'],
        'return_degradation': return_degradation
    })

    all_test_returns.extend(test_metrics['strategy_returns'].values)


# ============================================
# AGGREGATE RESULTS
# ============================================
print("\n" + "="*80)
print("WALK-FORWARD OPTIMIZATION SUMMARY")
print("="*80)

wfo_df = pd.DataFrame(wfo_results)

if len(wfo_df) > 0:
    print("\nPer-Window Results:")
    print(wfo_df[['window', 'best_lower', 'best_upper',
                  'train_return', 'test_return',
                  'train_sharpe', 'test_sharpe',
                  'train_sortino', 'test_sortino',
                  'return_degradation']].to_string(index=False))

    print("\n" + "="*80)
    print("OVERALL PERFORMANCE METRICS")
    print("="*80)

    avg_train_return  = wfo_df['train_return'].mean()
    avg_test_return   = wfo_df['test_return'].mean()
    avg_train_sharpe  = wfo_df['train_sharpe'].mean()
    avg_test_sharpe   = wfo_df['test_sharpe'].mean()
    avg_train_sortino = wfo_df['train_sortino'].mean()
    avg_test_sortino  = wfo_df['test_sortino'].mean()

    print(f"\nAverage Training Return:  {avg_train_return:.2%}")
    print(f"Average Test Return:      {avg_test_return:.2%}")
    print(f"Average Training Sharpe:  {avg_train_sharpe:.2f}")
    print(f"Average Test Sharpe:      {avg_test_sharpe:.2f}")
    print(f"Average Training Sortino: {avg_train_sortino:.2f}")
    print(f"Average Test Sortino:     {avg_test_sortino:.2f}")
    print(f"Average Return Degradation: {wfo_df['return_degradation'].mean():.1f}%")

    combined_test_return = (1 + pd.Series(all_test_returns)).prod() - 1
    print(f"\nCombined Out-of-Sample Return: {combined_test_return:.2%}")

    positive_test_windows = (wfo_df['test_return'] > 0).sum()
    print(f"\nWindows with Positive Test Returns: {positive_test_windows}/{len(wfo_df)} ({positive_test_windows/len(wfo_df)*100:.1f}%)")

    print("\n" + "="*80)
    print("PARAMETER STABILITY")
    print("="*80)
    print(f"Lower threshold range: {wfo_df['best_lower'].min()} to {wfo_df['best_lower'].max()}")
    print(f"Upper threshold range: {wfo_df['best_upper'].min()} to {wfo_df['best_upper'].max()}")
    print(f"Most common lower: {wfo_df['best_lower'].mode().values[0] if len(wfo_df['best_lower'].mode()) > 0 else 'N/A'}")
    print(f"Most common upper: {wfo_df['best_upper'].mode().values[0] if len(wfo_df['best_upper'].mode()) > 0 else 'N/A'}")

    print("\n" + "="*80)
    print("STRATEGY ASSESSMENT")
    print("="*80)

    if avg_test_return > 0 and avg_test_sharpe > 0.5:
        print(" PASSED: Strategy shows consistent out-of-sample profitability")
    elif avg_test_return > 0 and avg_test_sharpe > 0:
        print("MARGINAL: Strategy is barely profitable, needs improvement")
    else:
        print(" FAILED: Strategy does not show consistent profitability")

    if wfo_df['return_degradation'].mean() > -50:
        print(" Acceptable degradation from training to testing")
    else:
        print(" High degradation suggests overfitting")

    if positive_test_windows / len(wfo_df) > 0.6:
        print(" Good consistency across market regimes")
    else:
        print(" Poor consistency - strategy is regime-dependent")

    # ============================================
    # VISUALIZATION
    # ============================================
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    # Plot 1: Train vs Test Returns
    axes[0, 0].bar(wfo_df['window'] - 0.2, wfo_df['train_return'], width=0.4, alpha=0.7, label='Train')
    axes[0, 0].bar(wfo_df['window'] + 0.2, wfo_df['test_return'],  width=0.4, alpha=0.7, label='Test')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Window')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Train vs Test Returns per Window')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sharpe Ratios
    axes[0, 1].plot(wfo_df['window'], wfo_df['train_sharpe'], 'o-', label='Train Sharpe')
    axes[0, 1].plot(wfo_df['window'], wfo_df['test_sharpe'],  's-', label='Test Sharpe')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 1].axhline(y=1, color='g', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel('Window')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Sharpe Ratios per Window')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Sortino Ratios  <-- NEW
    axes[1, 0].plot(wfo_df['window'], wfo_df['train_sortino'], 'o-', label='Train Sortino')
    axes[1, 0].plot(wfo_df['window'], wfo_df['test_sortino'],  's-', label='Test Sortino')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 0].axhline(y=1, color='g', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Window')
    axes[1, 0].set_ylabel('Sortino Ratio')
    axes[1, 0].set_title('Sortino Ratios per Window')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Sharpe vs Sortino comparison on test data  <-- NEW
    x = wfo_df['window']
    axes[1, 1].plot(x, wfo_df['test_sharpe'],  'o-', label='Test Sharpe')
    axes[1, 1].plot(x, wfo_df['test_sortino'], 's-', label='Test Sortino')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Window')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_title('Test Sharpe vs Sortino per Window')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Parameter Evolution
    axes[2, 0].plot(wfo_df['window'], wfo_df['best_lower'], 'o-', label='Lower')
    axes[2, 0].plot(wfo_df['window'], wfo_df['best_upper'], 's-', label='Upper')
    axes[2, 0].set_xlabel('Window')
    axes[2, 0].set_ylabel('RSI Threshold')
    axes[2, 0].set_title('Optimal Parameters per Window')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Combined Out-of-Sample Equity Curve
    cumulative = (1 + pd.Series(all_test_returns)).cumprod()
    axes[2, 1].plot(cumulative.values)
    axes[2, 1].set_xlabel('Bar')
    axes[2, 1].set_ylabel('Cumulative Return')
    axes[2, 1].set_title('Combined Out-of-Sample Equity Curve')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('walk_forward_optimization_results.png', dpi=300, bbox_inches='tight')
    print(" Visualization saved as 'walk_forward_optimization_results.png'")

else:
    print(" No valid windows generated. Check your data and window settings.")