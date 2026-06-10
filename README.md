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



