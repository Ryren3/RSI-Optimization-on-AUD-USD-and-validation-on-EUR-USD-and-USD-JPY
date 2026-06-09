import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
import seaborn as sns

# ============================================
# CONFIGURATION
# ============================================
BARS_PER_DAY  = 288
TRADING_DAYS  = 252
BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS   # 72,576
TRANSACTION_COST = 0.0001                      # 1 pip

# ============================================
# LOAD AND PREPARE DATA
# ============================================
print("="*80)
print("Q-LEARNING REINFORCEMENT LEARNING FOR RSI TRADING")
print("="*80)

df = pd.read_csv('AUDUSD_5min_Jan2023_Sep2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.drop(columns=['volume', 'average', 'barCount'])
df = df.sort_values(by='date').reset_index(drop=True)

# Split BEFORE calculating RSI to avoid data leakage into the test set.
# RSI on the full dataset would mean test-period RSI values were computed
# using training data prices — the 14-bar lookback crosses the boundary.
split_idx = int(len(df) * 0.7)
df_train_raw = df.iloc[:split_idx].copy().reset_index(drop=True)
df_test_raw  = df.iloc[split_idx:].copy().reset_index(drop=True)

# Calculate RSI separately on each split
df_train_raw['rsi'] = RSIIndicator(close=df_train_raw['close'], window=14).rsi()
df_test_raw['rsi']  = RSIIndicator(close=df_test_raw['close'],  window=14).rsi()

df_train = df_train_raw.dropna().reset_index(drop=True)
df_test  = df_test_raw.dropna().reset_index(drop=True)

print(f"\nDataset split:")
print(f"Train: {len(df_train)} bars ({df_train['date'].min().date()} to {df_train['date'].max().date()})")
print(f"Test:  {len(df_test)} bars  ({df_test['date'].min().date()} to {df_test['date'].max().date()})")

# ============================================
# Q-LEARNING AGENT
# ============================================
class QLearningAgent:
    def __init__(self, n_states=10, n_actions=3, learning_rate=0.1,
                 discount_factor=0.95, epsilon=0.95, epsilon_decay=0.999,
                 epsilon_min=0.01):
        """
        Q-Learning Agent for trading

        States:  n_states bins of RSI (0-100 divided equally)
        Actions: 0=Buy, 1=Sell, 2=Hold
        """
        self.n_states        = n_states
        self.n_actions       = n_actions
        self.learning_rate   = learning_rate
        self.discount_factor = discount_factor
        self.epsilon         = epsilon
        self.epsilon_decay   = epsilon_decay
        self.epsilon_min     = epsilon_min

        self.q_table     = np.zeros((n_states, n_actions))
        self.action_names = {0: 'Buy', 1: 'Sell', 2: 'Hold'}

    def discretize_state(self, rsi_value):
        """Convert RSI (0-100) to discrete state index (0 to n_states-1)"""
        rsi_value = max(0.0, min(100.0, rsi_value))
        state = int(rsi_value * self.n_states / 100)
        return min(state, self.n_states - 1)   # clamp top edge

    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        """
        Q-learning update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') - Q(s,a) ]
        """
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next]
        td_error  = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================
# TRADING ENVIRONMENT
# ============================================
class TradingEnvironment:
    def __init__(self, data, transaction_cost=TRANSACTION_COST):
        """
        Trading environment using close prices.

        Positions: 0=Flat, 1=Long, -1=Short
        """
        self.data             = data.reset_index(drop=True)
        self.transaction_cost = transaction_cost
        self.current_step     = 0
        self.position         = 0
        self.entry_price      = 0.0

    def reset(self):
        self.current_step = 0
        self.position     = 0
        self.entry_price  = 0.0
        return self._get_state()

    def _get_state(self):
        return self.data.loc[self.current_step, 'rsi']

    def step(self, action):
        """
        Execute action, advance one bar, return (next_state, reward, done).

        Reward = P&L of the position held over this bar, minus transaction
        cost when the position changes.  Rewards are scaled ×100 for the
        Q-learning update signal.
        """
        current_price = self.data.loc[self.current_step, 'close']

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if done:
            return None, 0, done

        next_price  = self.data.loc[self.current_step, 'close']
        next_state  = self._get_state()

        prev_position = self.position
        cost          = 0.0

        # Update position and record entry price
        if action == 0:        # Buy → go long
            if self.position != 1:
                cost          = self.transaction_cost
                self.position = 1
                self.entry_price = current_price
        elif action == 1:      # Sell → go short
            if self.position != -1:
                cost          = self.transaction_cost
                self.position = -1
                self.entry_price = current_price
        # action == 2: Hold — position unchanged

        # Reward = one-bar P&L of the NEW (or held) position
        if self.position == 1:
            reward = (next_price - current_price) / current_price
        elif self.position == -1:
            reward = (current_price - next_price) / current_price
        else:
            reward = 0.0

        reward = (reward - cost) * 100   # scale for Q-update signal

        return next_state, reward, done


# ============================================
# TRAINING
# ============================================
def train_agent(agent, env, episodes=10):
    print("\n" + "="*80)
    print("TRAINING Q-LEARNING AGENT")
    print("="*80)

    episode_rewards  = []
    episode_epsilons = []

    for episode in range(episodes):
        state_rsi = env.reset()
        state     = agent.discretize_state(state_rsi)
        total_reward = 0
        steps        = 0

        while True:
            action = agent.choose_action(state, training=True)
            next_state_rsi, reward, done = env.step(action)

            if done:
                break

            next_state = agent.discretize_state(next_state_rsi)
            agent.update_q_value(state, action, reward, next_state)

            state         = next_state
            total_reward += reward
            steps        += 1

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.epsilon)

        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"Episode {episode+1:>3}/{episodes} | "
              f"Total Reward: {total_reward:>10.2f} | "
              f"Avg (last 10): {avg_reward:>10.2f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Steps: {steps}")

    return episode_rewards, episode_epsilons


# ============================================
# BACKTESTING
# ============================================
def backtest_agent(agent, data, title="Backtest"):
    """Backtest trained agent and compute performance metrics."""
    env       = TradingEnvironment(data)
    state_rsi = env.reset()
    state     = agent.discretize_state(state_rsi)

    actions_taken = []
    positions     = []
    rewards       = []
    prices        = []
    dates         = []
    total_reward  = 0

    while True:
        action = agent.choose_action(state, training=False)

        actions_taken.append(action)
        positions.append(env.position)
        prices.append(data.loc[env.current_step, 'close'])
        dates.append(data.loc[env.current_step, 'date'])

        next_state_rsi, reward, done = env.step(action)

        if done:
            break

        rewards.append(reward)
        total_reward += reward

        state = agent.discretize_state(next_state_rsi)

    results_df = pd.DataFrame({
        'date':     dates,
        'price':    prices,
        'action':   actions_taken,
        'position': positions,
        'reward':   rewards + [0]
    })

    # Returns: close-to-close, position applied with 1-bar delay
    results_df['market_return']   = results_df['price'].pct_change()
    results_df['strategy_return'] = results_df['position'].shift(1) * results_df['market_return']
    results_df['strategy_return'] = results_df['strategy_return'].fillna(0)

    # Transaction costs on position changes
    results_df['trade'] = ((results_df['position'] != results_df['position'].shift(1)) &
                            results_df['position'].shift(1).notna()).astype(int)
    results_df['strategy_return'] = results_df['strategy_return'] - (results_df['trade'] * TRANSACTION_COST)

    cumulative_return = (1 + results_df['strategy_return']).prod() - 1

    # ---- Metrics (annualised with BARS_PER_YEAR) ----
    mu    = results_df['strategy_return'].mean() * BARS_PER_YEAR
    sigma = results_df['strategy_return'].std()  * np.sqrt(BARS_PER_YEAR)
    sharpe = mu / sigma if sigma > 0 else 0

    # Sortino: semi-deviation below zero
    downside_diff      = np.minimum(results_df['strategy_return'], 0)
    downside_deviation = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(BARS_PER_YEAR)
    sortino = mu / downside_deviation if downside_deviation > 0 else 0

    # Max drawdown
    wealth      = (1 + results_df['strategy_return']).cumprod()
    running_max = wealth.cummax()
    max_dd      = ((wealth - running_max) / running_max).min()

    action_counts = results_df['action'].value_counts()

    print(f"\n{'='*80}")
    print(f"{title.upper()} RESULTS")
    print(f"{'='*80}")
    print(f"Total Reward:      {total_reward:.2f}")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Sortino Ratio:     {sortino:.2f}")
    print(f"Max Drawdown:      {max_dd:.2%}")
    print(f"\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"  {agent.action_names[action]}: {count} ({count/len(results_df)*100:.1f}%)")

    return results_df, cumulative_return, sharpe, sortino, max_dd


# ============================================
# MAIN EXECUTION
# ============================================
agent = QLearningAgent(
    n_states=10,           # 10 bins → RSI 0-10, 10-20, ..., 90-100
    n_actions=3,
    learning_rate=0.01,
    discount_factor=0.99,
    epsilon=0.9,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

train_env = TradingEnvironment(df_train)

EPISODES = 200
episode_rewards, episode_epsilons = train_agent(agent, train_env, episodes=EPISODES)

train_results, train_return, train_sharpe, train_sortino, train_dd = backtest_agent(agent, df_train, "Training")
test_results,  test_return,  test_sharpe,  test_sortino,  test_dd  = backtest_agent(agent, df_test,  "Testing")

# ============================================
# VISUALIZATION
# ============================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. Training progress
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(episode_rewards, linewidth=2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.set_title('Training Progress: Reward per Episode')
ax1.grid(True, alpha=0.3)

# 2. Epsilon decay
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(episode_epsilons, color='orange', linewidth=2)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Epsilon')
ax2.set_title('Exploration Rate Decay')
ax2.grid(True, alpha=0.3)

# 3. Q-table heatmap — labels derived from actual n_states
ax3 = fig.add_subplot(gs[1, :])
bin_size = 100 // agent.n_states
state_labels = [f'{i*bin_size}-{(i+1)*bin_size}' for i in range(agent.n_states)]
sns.heatmap(agent.q_table.T, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=state_labels,
            yticklabels=['Buy', 'Sell', 'Hold'],
            cbar_kws={'label': 'Q-Value'},
            ax=ax3)
ax3.set_xlabel('RSI State')
ax3.set_ylabel('Action')
ax3.set_title('Learned Q-Table (State-Action Values)')

# 4. Training equity curve
ax4 = fig.add_subplot(gs[2, 0])
train_equity = (1 + train_results['strategy_return']).cumprod()
ax4.plot(train_equity.values, linewidth=2)
ax4.set_xlabel('Time Steps')
ax4.set_ylabel('Cumulative Return')
ax4.set_title(f'Training Equity Curve\nReturn: {train_return:.2%} | Sharpe: {train_sharpe:.2f} | Sortino: {train_sortino:.2f}')
ax4.grid(True, alpha=0.3)

# 5. Test equity curve
ax5 = fig.add_subplot(gs[2, 1])
test_equity = (1 + test_results['strategy_return']).cumprod()
ax5.plot(test_equity.values, linewidth=2, color='green')
ax5.set_xlabel('Time Steps')
ax5.set_ylabel('Cumulative Return')
ax5.set_title(f'Test Equity Curve\nReturn: {test_return:.2%} | Sharpe: {test_sharpe:.2f} | Sortino: {test_sortino:.2f}')
ax5.grid(True, alpha=0.3)

# 6. Performance comparison (now includes Sortino)
ax6 = fig.add_subplot(gs[2, 2])
metrics    = ['Return %', 'Sharpe', 'Sortino', 'Max DD %']
train_vals = [train_return*100, train_sharpe, train_sortino, train_dd*100]
test_vals  = [test_return*100,  test_sharpe,  test_sortino,  test_dd*100]

x     = np.arange(len(metrics))
width = 0.35
ax6.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
ax6.bar(x + width/2, test_vals,  width, label='Test',  alpha=0.8)
ax6.set_ylabel('Value')
ax6.set_title('Performance Comparison')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, rotation=15)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# 7. Action distribution — training
ax7 = fig.add_subplot(gs[3, 0])
train_action_dist = train_results['action'].value_counts().sort_index()
ax7.bar([agent.action_names[i] for i in train_action_dist.index],
        train_action_dist.values, color=['green', 'red', 'gray'])
ax7.set_ylabel('Count')
ax7.set_title('Training: Action Distribution')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Action distribution — testing
ax8 = fig.add_subplot(gs[3, 1])
test_action_dist = test_results['action'].value_counts().sort_index()
ax8.bar([agent.action_names[i] for i in test_action_dist.index],
        test_action_dist.values, color=['green', 'red', 'gray'])
ax8.set_ylabel('Count')
ax8.set_title('Testing: Action Distribution')
ax8.grid(True, alpha=0.3, axis='y')

# 9. Reward distribution
ax9 = fig.add_subplot(gs[3, 2])
ax9.hist(train_results['reward'], bins=50, alpha=0.5, label='Train')
ax9.hist(test_results['reward'],  bins=50, alpha=0.5, label='Test')
ax9.set_xlabel('Reward')
ax9.set_ylabel('Frequency')
ax9.set_title('Reward Distribution')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.savefig('q_learning_results.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'q_learning_results.png'")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

summary_df = pd.DataFrame({
    'Metric':   ['Cumulative Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
    'Training': [f'{train_return:.2%}', f'{train_sharpe:.2f}', f'{train_sortino:.2f}', f'{train_dd:.2%}'],
    'Testing':  [f'{test_return:.2%}',  f'{test_sharpe:.2f}',  f'{test_sortino:.2f}',  f'{test_dd:.2%}']
})

print("\n", summary_df.to_string(index=False))

degradation = (test_return - train_return) / abs(train_return) * 100 if train_return != 0 else 0
print(f"\nReturn Degradation: {degradation:.1f}%")

if test_return > 0 and test_sharpe > 0.5:
    print("\n✅ Q-Learning agent shows promising results!")
elif test_return > 0:
    print("\n⚠️  Q-Learning agent is marginally profitable")
else:
    print("\n❌ Q-Learning agent needs improvement")

print("\n" + "="*80)