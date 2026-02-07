<div align="center">
  <img src="static/images/paper_logo.png" width="120px" alt="Strat-LLM Logo"/>

  # Strat-LLM: Stratified Strategy Alignment for LLM-based Stock Trading with Real-time Multimodal Signals

  [![Project Website](https://img.shields.io/badge/🌐-Project%20Website-0066CC?style=for-the-badge&logo=github)](https://Strat-LLM.github.io/)
  [![Paper](https://img.shields.io/badge/📄-IJCNN%20Submission-B31B1B?style=for-the-badge)](https://Strat-LLM.github.io/)
  [![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

</div>

---

## 📌 Overview

> **Strat-LLM** is a comprehensive framework designed to investigate the reliability boundaries of Large Language Models (LLMs) in financial decision-making. It introduces **Stratified Strategy Alignment** to modulate model autonomy across dynamic market environments.

At its core, the project operates in a **Live-Forward 2025 Framework**, strictly mimicking the chronological flow of information to preclude data leakage and look-ahead bias. By integrating **sequential numerical market streams** with **static textual annual reports**, Strat-LLM evaluates how strategic constraints influence trading performance using state-of-the-art models like **Qwen3**.

> [!IMPORTANT]
> **Anonymous IJCNN Submission**
> This repository contains code and datasets for review purposes. Full open-source release will follow upon acceptance.

---

## ✨ Key Features

### 🚀 Core Capabilities

#### 🛡️ Stratified Strategy Alignment
Implements three distinct levels of autonomy to audit decision rationales:
1.  **Free Mode:** Unrestricted Role-Playing (Native Financial Intuition).
2.  **Guided Mode:** **Human-AI Collaboration** (Strategies provided as reference with dynamic adjustment).
3.  **Strict Mode:** Hard-constrained strategic execution.

#### 📊 Live-Forward 2025 Setup
Rejects traditional random-split backtesting. The framework strictly adheres to a timeline from **Jan 1, 2025, to Sept 30, 2025**, ensuring no future data leaks into the decision process.

#### 📈 Multimodal Signal Fusion
Synthesizes high-frequency numerical data (**Daily/Minute-level Price Data**) with low-frequency textual insights (**Real-time News** & **Annual Report Summaries**) to drive trading decisions.

#### 🧪 Cross-Market Stress Tests
Evaluates performance divergence across distinct market environments:
* **A-share Market:** High volatility, inefficient (Alpha-driven).
* **U.S. Market:** High efficiency, institutional dominance (Beta-driven).

> **Key Finding:** *The Alignment Tax* — Strict constraints limit upside in inefficient markets but provide essential robustness in efficient ones.

---

## 🛠️ Getting Started

### 1. Requirements

| Requirement | Specification |
| :--- | :--- |
| **Python** | `3.13` |
| **Framework** | PyTorch (Latest Stable) |

Install the necessary dependencies (including **scikit-learn** and **matplotlib** for strategy evaluation and visualization):

```bash
pip install torch transformers pandas numpy tushare yfinance scikit-learn matplotlib accelerate
```

### ⚙️ 2. Configuration
Configure your market environment and alignment strategy in `config.py`. The system supports switching between A-share and U.S. market data streams.

<details>
<summary><b>📝 Click to view configuration example</b></summary>

```python
### 3. Configuration

The framework uses `ExperimentConfig` and `ModelConfig` dataclasses for centralized management. You can customize these settings in `config.py` or pass them dynamically.

```python
# config.py

# Corresponds to ExperimentConfig class
EXPERIMENT_CONFIG = {
    # Market Selection
    'market_type': 'us_stock',          # Options: 'us_stock', 'a_stock'
    
    # Strategy Alignment (The Core Feature)
    'strategy_mode': 'strict_strategies', # Options: 'free_strategies', 'guided_strategies', 'strict_strategies'
    'attitude': 'aggressive',             # Options: 'conservative', 'neutral', 'aggressive'
    
    # Data Granularity
    'data_level': 'full_data',          # Options: 'price_only', 'price_news', 'full_data' (includes Annual Reports)
    
    # Time & Cycle
    'cycle_type': 18,                   # 18 (Short Cycle) or 93 (Long Cycle)
    'start_date': "2025-01-01",
    'end_date': "2025-06-30"
}

# ================= Model Settings =================
# Corresponds to ModelConfig class
LLM_CONFIG = {
    'model_name': 'Qwen/Qwen3-32B',     # Primary model 
    'api_base': "",
    'enable_thinking': True,            # Enable Chain-of-Thought reasoning
    'thinking_budget': 4096,            # Token budget for reasoning
    'temperature': 0.7,
    'max_tokens': 4096
}
```
</details>




### ▶️ 3. Execution
Run the live-forward simulation:

```python
python run_strat_simulation.py 
```

### 📂 4. Data & Framework Logic

### Multimodal Inputs
The framework integrates heterogeneous data streams to simulate a realistic trading environment:

| **Data Modality** | **Source** | **Description** |
| :--- | :--- | :--- |
| 🔢 **Numerical** | Alpha-360 / Qlib | Open, High, Low, Close, Volume + 360 Technical Factors |
| 📰 **News** | Real-time Streams | NLP-extracted sentiment and key events (2025 Live-forward) |
| 📄 **Knowledge** | Annual Reports | Static strategic outlooks and risk factors |
| 🌏 **Markets** | CSI 300 / S&P 500 | Top constituents covering A-Shares and U.S. Equities |

### Framework Logic: Stratified Strategy Alignment
To investigate the reliability boundaries of LLMs, we implement three progressive autonomy modes:

| **Mode** | **Autonomy Level** | **Logic Specification** |
| :--- | :--- | :--- |
| 🦅 **Free Mode** | High (Zero-shot) | Relies solely on the LLM's native financial intuition. No external constraints. |
| 🤝 **Guided Mode** | Medium (Co-pilot) | Strategies (S1-S4) provided as references. LLM can adjust based on news. |
| 🔒 **Strict Mode** | Low (Compliance) | **Mandatory adherence.** LLM must cite specific Strategy Rules (e.g., *Breakout Momentum*) to justify trades. |

> **Note:** The framework operates in a **Live-Forward** setting (2025), strictly preventing look-ahead bias by sequentially feeding data day-by-day.

### Theoretical Workflow
The Strat-LLM framework operates on a **T+1 Rolling Basis**, strictly adhering to a Long-Only Accumulation Protocol to isolate entry precision from exit noise.

#### 1. Multimodal State Construction
At trading day $T$, the agent observes a composite state vector $S_T$, integrating real-time and static streams:

$$S_T = \{ \mathbf{P}_{T}, \mathcal{N}_{T}, \mathcal{K}_{\text{static}} \}$$

* $\mathbf{P}_{T}$: Numerical market history (OHLCV + Technical Factors).
* $\mathcal{N}_{T}$: **Real-time News Streams** (Sentiment & Key Events via NLP).
* $\mathcal{K}_{\text{static}}$: Static Knowledge Base (Annual Reports & Strategic Outlooks).

#### 2. Stratified Strategy Alignment
The LLM generates a binary action $A_T \in \{0, 1\}$ based on the active Autonomy Mode ($\psi$):

$$
A_T = \pi_{\text{LLM}}(S_T, \psi) \rightarrow \begin{cases} 
1 & \text{(Buy/Accumulate)} \\\\
0 & \text{(Wait/Hold)} 
\end{cases}
$$

* **Free Mode:** $\pi$ relies on intrinsic intuition.
* **Strict Mode:** $\pi$ is constrained by expert rules ($S_1-S_4$), requiring specific rationale citations.

#### 3. Execution & Metacognitive Auditing
Unlike traditional RL rewards, we evaluate **Decision Consistency** and **Risk-Adjusted Alpha**.

**Key Metric 1: Cognitive Dissonance ($\sigma_C$)**
Measures the stability of the model's reasoning process under strict constraints. A spike in $\sigma_C$ indicates conflict between intuition and rules:

$$\sigma_{C} = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(\bar{C}_t - \bar{C})^2}$$

**Key Metric 2: Alignment Tax (Alpha)**
Quantifies the excess return sacrificed for compliance, derived from the CAPM model:

$$R_p - R_f = \alpha + \beta(R_m - R_f) + \epsilon$$

