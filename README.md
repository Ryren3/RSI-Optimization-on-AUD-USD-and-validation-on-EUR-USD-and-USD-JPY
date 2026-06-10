# Optimizing RSI thresholds on the AUD/USD and validating on the EUR/USD and USD/JPY.
--

##  Project Overview
This dissertation evaluates the structural stability, out-of-sample robustness, cross-currency generalization, and risk-adjusted performance of optimized RSI thresholds. 
The experimental framework utilizes high-frequency 5-minute Forex intraday data across three major currency pairs (**AUD/USD**, **EUR/USD**, and **USD/JPY**) spanning from
December 2022 to September 2025. The optimal threshold is found on the **AUD/USD** and validated on **EUR/USD**, and **USD/JPY**.

--

## Key Findings
* **In-Sample Superiority:** Tailored, optimized RSI thresholds significantly outperform conventional 30/70 baseline strategies during historical in-sample training.
* **Out-of-Sample Decay:** Trading performance deteriorates substantially when frozen optimized parameters are exposed to unseen, forward test windows.
* **Zero Cross-Asset Generalization:** RSI threshold configurations optimized on a specific currency pair fail completely when cross-validated on alternative pairs, proving asset-specific volatility signatures.
* **The AI Complexity Paradox:** Adaptive Reinforcement Learning (Q-Learning) heavily underperforms simpler, rigid, rule-based heuristics due to high-frequency microstructure noise.
* **Market Non-Stationarity:** Walk-Forward Optimization (WFO) confirmed structural parameter drift and severe regime dependency, demonstrating that optimal momentum thresholds are transient.

---

## Methodology & Architectural Paradigms

The study compares three distinct computational optimization paradigms:

| Methodology | Optimization Type | Core Operational Mechanics |
| :--- | :--- | :--- |
| **Grid Search** | Exhaustive Parameter Sweeping | Executes a brute-force sweep across a matrix of lower/upper boundaries to maximize historical cumulative returns. |
| **Walk-Forward Optimization** | Rolling Temporal Validation | Simulates realistic live-trading deployment by continuously training and testing across sliding historical windows. |
| **Reinforcement Learning** | Dynamic Policy Adaptation | Deploys a model-free Q-Learning agent to learn an optimal action-policy mapped directly to discretized RSI states. |

---

## Dataset & Technical Specifications

| Attribute | Specification |
| :--- | :--- |
| **Data Source** | Interactive Brokers (IBKR) API |
| **Temporal Frequency** | 5-Minute OHLCV Candles |
| **Sampling Window** | December 2022 – September 2025 |
| **Assets Evaluated** | AUD/USD, EUR/USD, USD/JPY |
| **Primary Indicator** | 14-Period Relative Strength Index (RSI) |

The RSI functions as a bounded momentum oscillator operating within a range of $[0, 100]$. The baseline benchmark strategy abides by conventional heuristics:
* $\text{RSI} < 30 \rightarrow \text{Oversold} \rightarrow \text{Trigger Buy Long}$
* $\text{RSI} > 70 \rightarrow \text{Overbought} \rightarrow \text{Trigger Sell Short}$

---

## Experimental Framework & Phase Analysis

### 🔍 1. Exhaustive Grid Search Optimization

The parameter space was isolated to analyze non-linear return surfaces across a 676-combination matrix:
* **Lower Threshold Boundary Layer:** Evaluated increments from $20$ to $45$.
* **Upper Threshold Boundary Layer:** Evaluated increments from $65$ to $90$.
* **Total Combinations:** $26 \times 26 = 676$ matrix iterations.

#### 🏆 In-Sample Optimization Peak (AUD/USD Training)
| Lower RSI | Upper RSI | Cumulative Return | Sharpe Ratio | Sortino Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **21** | **77** | **31.70%** | **1.43** | **2.03** | **-7.40%** |

#### 📉 Out-of-Sample Forward Performance Decay
| Metric | Forward Test Set Result |
| :--- | :---: |
| **Test Set Return** | 4.73% |
| **Sharpe Ratio** | 0.57 |
| **Sortino Ratio** | 0.81 |
| **Maximum Drawdown** | -7.99% |

#### 📌 Quantitative Interpretation
The stark compression in the Sharpe Ratio ($1.43 \rightarrow 0.57$) indicates **severe data over-fitting**. While the optimized $(21, 77)$ boundary retained marginal profitability out-of-sample, the return distribution surface flatten significantly, proving that exhaustive historical maximization captures fleeting structural patterns rather than persistent market inefficiencies.

[Insert Grid Search Heatmap Here]
Caption: Figure 1.1: Cumulative return topology across 676 RSI parameter combinations, demonstrating localized spikes in historical profitability.


---

### 2. Walk-Forward Optimization (WFO)

To counteract static parameter decay, a rolling walk-forward framework was implemented to mirror live deployment conditions. 

####  Sliding Window Architecture
* **In-Sample Training Sub-Window:** 6 Months (Parameter Selection)
* **Out-of-Sample Testing Sub-Window:** 3 Months (Forward Tracking)
* **Total Evaluations:** 9 Sequential Rolling Iterations

####  Empirical Rolling Results
| Window | Optimal In-Sample RSI | Out-of-Sample Test Return (%) |
| :---: | :---: | :---: |
| **1** | 23 / 77 | +4.91% |
| **2** | 20 / 74 | -7.37% |
| **3** | 20 / 80 | +5.38% |
| **4** | 20 / 83 | +2.39% |
| **5** | 20 / 74 | +3.14% |
| **6** | 29 / 83 | -11.23% |
| **7** | 23 / 68 | +7.09% |
| **8** | 23 / 68 | -1.42% |
| **9** | 41 / 71 | +0.39% |

#### Mathematical Insights & Parameter Drift
The walk-forward matrix provides definitive proof of **forex market non-stationarity**. The rapid fluctuation of optimal bounds (ranging from highly conservative
$20/83$ combinations to tight $41/71$ spaces) proves structural parameter drift. Identical parameter rules exposed to consecutive windows generated opposing return
distributions, establishing that changing macroeconomic regimes invalidate static mathematical optimization.

#### Generalized Operational Recommendation
To bypass window-specific overfitting, the dissertation aggregates rolling parameters to establish a **Generalized Threshold Alternative (Mean lower/upper bounds)**:
**`RSI (24 / 75)`**. This smoothed configuration neutralizes short-term regime dependency and yields more reliable, low-variance baseline parameters.

<img width="4470" height="4159" alt="image" src="https://github.com/user-attachments/assets/bdd69c20-70f9-4bdd-b4f0-50deffad50e3" />
Figure 2.1: Equity curve progression and parameter drift across 9 distinct sliding validation windows.

---

###  3. Reinforcement Learning via Model-Free Q-Learning

To transition from rigid boundaries to adaptive decision-making, a Q-Learning agent was built to map dynamic actions directly to momentum states. While the former two methods showed the one ideal set of RSI thresholds, RL model assigns the best action to be taken in each RSI state, thus making it more dynamic. 

####  Discretized State Space
The raw continuous $[0, 100]$ RSI space was mapped into 10 distinct mathematical bins:
$$\text{State } s \in \{0, 1, 2, \dots, 9\} \quad \text{where} \quad s = \lfloor \text{RSI} / 10 \rfloor$$

####  Action Space
$$\text{Action } a \in \{\text{Buy (Long)}, \text{Hold (Neutral)}, \text{Sell (Short)}\}$$

####  Step-Wise Reward Function
The agent’s feedback loops were driven by directional next-period logarithmic price changes:
$$r_{t+1} = \begin{cases} \ln(P_{t+1} / P_t) & \text{if Action} = \text{Buy} \\ -\ln(P_{t+1} / P_t) & \text{if Action} = \text{Sell} \\ 0 & \text{if Action} = \text{Hold} \end{cases}$$

####  Hyperparameter Configuration
* **Training Episodes:** 200
* **Learning Rate ($\alpha$):** 0.01
* **Discount Factor ($\gamma$):** 0.99
* **Exploration Policy:** $\epsilon$-greedy ($\epsilon_{init} = 0.9$, $\text{Decay Rate} = 0.995$)

####  Reinforcement Learning Performance Metrics
| Phase | Cumulative Return | Sharpe Ratio | Sortino Ratio | Maximum Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **Training Phase** | +6.25% | 0.36 | 0.50 | -11.61% |
| **Testing Phase** | **-6.16%** | **-0.66** | **-0.92** | **-13.24%** |

#### Algorithmic Interpretation
The Q-Learning agent produced the weakest out-of-sample performance across all configurations. This highlights a classic hurdle in financial machine learning: 
**low signal-to-noise ratios**. At a high-frequency 5-minute resolution, asset prices are dominated by marketplace microstructure noise (bid-ask bounce, execution fragmentation). This noise floods the reward function with variance, leading to extreme sample inefficiency and causing the tabular agent to overfit to random market static.

<img width="3554" height="795" alt="image" src="https://github.com/user-attachments/assets/09832578-2f52-4782-a353-c9451fef9e74" />
Figure 3.1: The optimal action state in each action state.

Fig 3.1 shows the optimal action in each RSI state, higher the Q-value for each state under each action more likely that action is more suited in that state. State 0-10 is the best state for buying, as shown with a high q-value of 0.06. 

---

## Cross-Model Performance Matrix

The final model selection table compiles performance metrics recorded over identical, out-of-sample test data:

| Optimization Framework | RSI Parameter Set | Test Return | Sharpe Ratio | Sortino Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Grid Search Optimization** | 21 / 77 | **+4.73%** | **0.57** | **0.81** | **-7.99%** |
| **Walk-Forward Framework** | 24 / 75 | +0.35% | 0.18 | 0.34 | -11.24% |
| **Tabular Q-Learning Agent** | Adaptive Policy | -14.94% | -0.12 | -0.36 | -12.07% |

---

## Cross-Currency Stress Testing & Validation

To test for geographical and asset-class generalizability, the top performing configuration derived from AUD/USD ($21 / 77$) was cross-validated on alternative, out-of-sample currency pairs:

| Target Pair | Validation Return | Sharpe Ratio | Sortino Ratio | Maximum Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **EUR/USD** | -8.13% | -0.36 | -0.51 | -13.33% |
| **USD/JPY** | -25.89% | -0.99 | -1.39 | -34.84% |

#### Critical Validation Takeaway
Cross-currency testing confirms that **optimal momentum thresholds are asset-specific**. Mathematical boundaries optimized for a commodity-linked pair (AUD/USD) collapsed when exposed to heavy funding currencies (USD/JPY). This confirms that independent currency pairs operate within unique volatility regimes, invalidating the concept of a generalized global RSI threshold setting.

---

## Project Limitations & Research Boundaries

* **Single-Feature Isolation:** The pipeline relies entirely on the RSI oscillator, creating an information deficit regarding structural trend direction and market context.
* **Absence of Volatility Adapters:** Threshold filters remain fixed and do not dynamically scale based on changing volatility measures (e.g., Average True Range or Bollinger Band expansions).
* **Tabular State Constraints:** Discretizing continuous data into 10 rigid tabular bins strips the reinforcement learning agent of fine-grained momentum details.
* **Microstructure Dominated Horizon:** Operating at a 5-minute sampling interval forces the models to process heavy exchange order-matching noise, obfuscating broad structural trends.
* **Deterministic Modeling Deficit:** The system lacks probabilistic uncertainty modules (e.g., Conformal Predictions or Bayesian frameworks) to handle tail risk events.

---

## Future Research Directions

1. **Advanced RL algorithms:** Transitioning from tabular formats to Deep Q-Networks (DQN) to process dense, high-dimensional, continuous feature spaces.
2. **Context-Aware Multi-Indicator Systems:** Compounding the momentum pipeline with trend indicators (e.g., Ichimoku Cloud matrices) and volatility components.
3. **Volatility-Adjusted Alpha Bands:** Building dynamic RSI lines that expand and contract as a function of running rolling asset volatility.
4. **Deep Sequence Modeling:** Implementing Long Short-Term Memory (LSTM) layers or Transformer blocks ahead of the reinforcement learning policy to capture deep historical dependencies.

---
