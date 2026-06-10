import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

# ============================================
# CONFIGURATION
# ============================================
LOWER_RSI = 21
UPPER_RSI = 77
BARS_PER_DAY = 288
TRADING_DAYS = 252
BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS   # 72,576
TRANSACTION_COST = 0.0001                      # 1 pip

print("="*80)
print("MULTI-CURRENCY RSI STRATEGY BACKTEST")
print("="*80)
print(f"Strategy: Long-Short RSI Mean Reversion")
print(f"Parameters: RSI {LOWER_RSI}/{UPPER_RSI}")
print(f"Long when RSI < {LOWER_RSI}, Short when RSI > {UPPER_RSI}")
print("="*80)

# ============================================
# HELPER FUNCTIONS
# ============================================
def load_and_prepare_data(filename, currency_pair):
    """Load CSV and calculate RSI"""
    print(f"\nLoading {currency_pair} data from {filename}...")

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])

    cols_to_drop = [col for col in ['volume', 'average', 'barCount'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df.sort_values('date', inplace=True)

    # RSI calculated on the full dataset here is correct for a single-period
    # backtest — there is no train/test split so there is no leakage.
    # dropna() removes the first 14 bars where RSI is undefined.
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df = df.dropna().reset_index(drop=True)

    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total bars: {len(df):,}")
    print(f"  Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")

    return df


def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from returns series"""
    wealth = (1 + cumulative_returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return drawdown.min()


def backtest_rsi_strategy(data, lower, upper, currency_pair):
    """
    Backtest RSI long-short strategy

    Strategy:
    - Go LONG  when RSI < lower threshold
    - Go SHORT when RSI > upper threshold
    - Hold position until opposite signal
    """
    df = data.copy()

    # Generate signals
    df['signal'] = 0
    df.loc[df['rsi'] < lower, 'signal'] = 1    # Buy signal
    df.loc[df['rsi'] > upper, 'signal'] = -1   # Sell signal

    # Execute trades at next bar open (avoid look-ahead bias)
    df['position'] = df['signal'].replace(0, np.nan).shift(1)
    df['position'] = df['position'].ffill().fillna(0)

    # Returns: open-to-open, next bar
    df['return'] = df['open'].shift(-1) / df['open'] - 1
    df['strategy_return'] = df['position'] * df['return']
    df['strategy_return'] = df['strategy_return'].fillna(0)

    # Transaction costs: only charge when position genuinely changes
    # (exclude first bar where shift produces NaN → not a real trade)
    df['trade'] = ((df['position'] != df['position'].shift(1)) &
                    df['position'].shift(1).notna()).astype(int)
    df['strategy_return'] = df['strategy_return'] - (df['trade'] * TRANSACTION_COST)
    df['strategy_return'] = df['strategy_return'].fillna(0)

    # ---- Performance metrics ----
    cumulative_return = (1 + df['strategy_return']).prod() - 1

    # Annualised mean return
    mu = df['strategy_return'].mean() * BARS_PER_YEAR

    # Sharpe ratio: penalises all volatility (up and down)
    sigma = df['strategy_return'].std() * np.sqrt(BARS_PER_YEAR)
    sharpe_ratio = mu / sigma if sigma > 0 else 0

    # Sortino ratio: penalises only downside volatility (semi-deviation)
    downside_diff = np.minimum(df['strategy_return'], 0)
    downside_deviation = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(BARS_PER_YEAR)
    sortino_ratio = mu / downside_deviation if downside_deviation > 0 else 0

    # Maximum drawdown
    max_dd = calculate_max_drawdown(df['strategy_return'])

    # Trading activity
    total_trades = (df['position'].diff() != 0).sum()
    long_periods  = (df['position'] == 1).sum()
    short_periods = (df['position'] == -1).sum()
    flat_periods  = (df['position'] == 0).sum()

    # Win rate
    profitable_periods  = (df['strategy_return'] > 0).sum()
    total_active_periods = len(df[df['strategy_return'] != 0])
    win_rate = profitable_periods / total_active_periods if total_active_periods > 0 else 0

    # Buy & hold for comparison
    buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

    return {
        'currency_pair':   currency_pair,
        'cumulative_return': cumulative_return,
        'sharpe_ratio':    sharpe_ratio,
        'sortino_ratio':   sortino_ratio,
        'max_drawdown':    max_dd,
        'total_trades':    total_trades,
        'long_periods':    long_periods,
        'short_periods':   short_periods,
        'flat_periods':    flat_periods,
        'win_rate':        win_rate,
        'buy_hold_return': buy_hold_return,
        'equity_curve':    (1 + df['strategy_return']).cumprod(),
        'dates':           df['date'],
        'positions':       df['position'],
        'returns':         df['strategy_return']
    }


def print_results(results):
    """Print formatted results"""
    print(f"\n{'='*80}")
    print(f"{results['currency_pair']} RESULTS")
    print(f"{'='*80}")
    print(f"Cumulative Return:     {results['cumulative_return']:>10.2%}")
    print(f"Buy & Hold Return:     {results['buy_hold_return']:>10.2%}")
    print(f"Excess Return:         {results['cumulative_return'] - results['buy_hold_return']:>10.2%}")
    print(f"Sharpe Ratio:          {results['sharpe_ratio']:>10.2f}")
    print(f"Sortino Ratio:         {results['sortino_ratio']:>10.2f}")
    print(f"Maximum Drawdown:      {results['max_drawdown']:>10.2%}")
    print(f"Win Rate:              {results['win_rate']:>10.1%}")
    print(f"\nTrading Activity:")
    print(f"  Total Position Changes: {results['total_trades']:>6}")
    n = len(results['returns'])
    print(f"  Long Periods:           {results['long_periods']:>6} ({results['long_periods']/n*100:.1f}%)")
    print(f"  Short Periods:          {results['short_periods']:>6} ({results['short_periods']/n*100:.1f}%)")
    print(f"  Flat Periods:           {results['flat_periods']:>6} ({results['flat_periods']/n*100:.1f}%)")


# ============================================
# LOAD DATA AND RUN BACKTESTS
# ============================================
currency_pairs = {
    'AUD/USD': 'AUDUSD_5min_Jan2023_Sep2025.csv',
    'EUR/USD': 'EURUSD_5min_Jan2023_Sep2025.csv',
    'JPY/USD': 'USDJPY2_5min_Jan2023_Sep2025.csv'
}

all_results = {}

for pair_name, filename in currency_pairs.items():
    try:
        df = load_and_prepare_data(filename, pair_name)
        results = backtest_rsi_strategy(df, LOWER_RSI, UPPER_RSI, pair_name)
        all_results[pair_name] = results
        print_results(results)
    except FileNotFoundError:
        print(f"\n⚠️  Warning: {filename} not found. Skipping {pair_name}.")
    except Exception as e:
        print(f"\n❌ Error processing {pair_name}: {str(e)}")

# ============================================
# COMPARISON TABLE
# ============================================
if len(all_results) > 0:
    print("\n" + "="*80)
    print("CROSS-CURRENCY COMPARISON")
    print("="*80)

    comparison_df = pd.DataFrame({
        'Currency':          [r['currency_pair']      for r in all_results.values()],
        'Cumulative Return': [f"{r['cumulative_return']:.2%}" for r in all_results.values()],
        'Sharpe Ratio':      [f"{r['sharpe_ratio']:.2f}"      for r in all_results.values()],
        'Sortino Ratio':     [f"{r['sortino_ratio']:.2f}"     for r in all_results.values()],
        'Max Drawdown':      [f"{r['max_drawdown']:.2%}"      for r in all_results.values()],
        'Buy & Hold':        [f"{r['buy_hold_return']:.2%}"   for r in all_results.values()],
        'Win Rate':          [f"{r['win_rate']:.1%}"          for r in all_results.values()],
        'Trades':            [r['total_trades']               for r in all_results.values()]
    })

    print("\n", comparison_df.to_string(index=False))

    # ============================================
    # VISUALIZATION
    # ============================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    n_pairs = len(all_results)
    fig, axes = plt.subplots(2, n_pairs, figsize=(6*n_pairs, 10))

    if n_pairs == 1:
        axes = axes.reshape(2, 1)

    for idx, (pair_name, results) in enumerate(all_results.items()):
        # Top row: Equity curves
        ax1 = axes[0, idx]
        ax1.plot(results['equity_curve'].values, linewidth=2, label='Strategy')
        ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title(
            f'{pair_name} Equity Curve\n'
            f'Return: {results["cumulative_return"]:.2%} | '
            f'Sharpe: {results["sharpe_ratio"]:.2f} | '
            f'Sortino: {results["sortino_ratio"]:.2f}'
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Bottom row: Return distributions
        ax2 = axes[1, idx]
        returns_pct = results['returns'] * 100
        ax2.hist(returns_pct, bins=100, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{pair_name} Return Distribution')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('multi_currency_backtest_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'multi_currency_backtest_results.png'")

    # ============================================
    # PERFORMANCE COMPARISON BAR CHARTS
    # ============================================
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))   # expanded to 4 panels

    pairs     = list(all_results.keys())
    returns   = [all_results[p]['cumulative_return'] * 100 for p in pairs]
    sharpes   = [all_results[p]['sharpe_ratio']           for p in pairs]
    sortinos  = [all_results[p]['sortino_ratio']          for p in pairs]
    drawdowns = [all_results[p]['max_drawdown'] * 100     for p in pairs]

    # Panel 1: Cumulative returns
    colors = ['green' if r > 0 else 'red' for r in returns]
    axes2[0].bar(pairs, returns, color=colors, alpha=0.7)
    axes2[0].axhline(y=0, color='black', linewidth=1)
    axes2[0].set_ylabel('Return (%)')
    axes2[0].set_title('Cumulative Returns')
    axes2[0].grid(True, alpha=0.3, axis='y')
    axes2[0].tick_params(axis='x', rotation=45)

    # Panel 2: Sharpe ratios
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    axes2[1].bar(pairs, sharpes, color=colors, alpha=0.7)
    axes2[1].axhline(y=0, color='black', linewidth=1)
    axes2[1].axhline(y=1, color='orange', linewidth=1, linestyle='--', alpha=0.7, label='Threshold (1.0)')
    axes2[1].set_ylabel('Sharpe Ratio')
    axes2[1].set_title('Sharpe Ratios')
    axes2[1].grid(True, alpha=0.3, axis='y')
    axes2[1].tick_params(axis='x', rotation=45)
    axes2[1].legend()

    # Panel 3: Sortino ratios
    colors = ['green' if s > 0 else 'red' for s in sortinos]
    axes2[2].bar(pairs, sortinos, color=colors, alpha=0.7)
    axes2[2].axhline(y=0, color='black', linewidth=1)
    axes2[2].axhline(y=1, color='orange', linewidth=1, linestyle='--', alpha=0.7, label='Threshold (1.0)')
    axes2[2].set_ylabel('Sortino Ratio')
    axes2[2].set_title('Sortino Ratios')
    axes2[2].grid(True, alpha=0.3, axis='y')
    axes2[2].tick_params(axis='x', rotation=45)
    axes2[2].legend()

    # Panel 4: Max drawdowns
    axes2[3].bar(pairs, drawdowns, color='red', alpha=0.7)
    axes2[3].set_ylabel('Max Drawdown (%)')
    axes2[3].set_title('Maximum Drawdowns')
    axes2[3].grid(True, alpha=0.3, axis='y')
    axes2[3].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('multi_currency_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Comparison chart saved as 'multi_currency_performance_comparison.png'")

    # ============================================
    # FINAL ASSESSMENT
    # ============================================
    print("\n" + "="*80)
    print("STRATEGY ASSESSMENT ACROSS CURRENCIES")
    print("="*80)

    avg_return  = np.mean([r['cumulative_return'] for r in all_results.values()])
    avg_sharpe  = np.mean([r['sharpe_ratio']      for r in all_results.values()])
    avg_sortino = np.mean([r['sortino_ratio']      for r in all_results.values()])
    profitable_pairs = sum([1 for r in all_results.values() if r['cumulative_return'] > 0])

    print(f"\nAverage Return:     {avg_return:.2%}")
    print(f"Average Sharpe:     {avg_sharpe:.2f}")
    print(f"Average Sortino:    {avg_sortino:.2f}")
    print(f"Profitable Pairs:   {profitable_pairs}/{len(all_results)}")

    print("\n" + "-"*80)

    if avg_return > 0.1 and avg_sharpe > 0.5 and profitable_pairs == len(all_results):
        print("✅ EXCELLENT: Strategy works consistently across all currency pairs!")
        print(f"   The RSI {LOWER_RSI}/{UPPER_RSI} parameters appear to be robust.")
    elif avg_return > 0 and profitable_pairs >= len(all_results) * 0.66:
        print("⚠️  GOOD: Strategy is profitable on most pairs but shows variability.")
        print("   Consider currency-specific optimization or regime filters.")
    elif profitable_pairs > 0:
        print("⚠️  MIXED: Strategy works on some pairs but not others.")
        print("   This suggests currency-specific market dynamics.")
    else:
        print("❌ POOR: Strategy does not generalize well to other currency pairs.")
        print("   The AUD/USD optimization may have overfit to specific conditions.")

    print("="*80)

else:
    print("\n❌ No results to display. Please check your data files.")