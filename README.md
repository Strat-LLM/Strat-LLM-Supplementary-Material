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
1.  **Free Mode:** Unrestricted Role-Playing (High Creativity, High Risk).
2.  **Guided Mode:** Chain-of-Thought (CoT) with RAG support.
3.  **Strict Mode:** Hard-constrained strategic execution.

#### 📊 Live-Forward 2025 Setup
Rejects traditional random-split backtesting. The framework strictly adheres to a timeline from **Jan 1, 2025, to Dec 31, 2025**, ensuring no future data leaks into the decision process.

#### 📈 Multimodal Signal Fusion
Synthesizes high-frequency numerical data (**Alpha-360 factors**) with low-frequency textual insights (**Annual Report Summaries**) to drive trading decisions.

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
| **LLM Backend** | Qwen3 (via API or Local) |

Install the necessary dependencies:

```bash
pip install torch transformers pandas numpy tushare yfinance


### ⚙️ 2. Configuration
Configure your market environment and alignment strategy in config.py. The system supports switching between A-share and U.S. market data streams.

<details> <summary><b>📝 Click to view configuration example</b></summary>
# config.py

# Strategy Configuration
STRATEGY_CONFIG = {
    'market': 'US',          # Options: 'CN' (A-share), 'US' (S&P 500)
    'mode': 'Strict',        # Options: 'Free', 'Guided', 'Strict'
    'lookback_window': 30,   # Days of historical data
}

# Model Settings
LLM_CONFIG = {
    'model_name': 'Qwen/Qwen3-72B-Instruct', # Primary model used in 2025 experiments
    'temperature': 0.1,      # Low temp for trading consistency
    'max_tokens': 1024
}
</details>




### ▶️ 3. Execution
Run the live-forward simulation:
python run_strat_simulation.py --start_date 2025-01-01 --end_date 2025-12-31
📂 Data & Framework Logic
Multimodal Inputs
The framework utilizes a dual-stream data structure to simulate the trading desk environment:
Data Modality,Source,Description
🔢 Numerical,Alpha-360 / Qlib,"Open, High, Low, Close, Volume + 360 Technical Factors"
📄 Textual,Annual Reports,Summarized strategic outlooks and risk factors
🌏 Markets,CSI 300 / S&P 500,Top constituents by market cap
### Theoretical Workflow
The Strat-LLM decision process follows a sequential state-action workflow:

#### 1. State Construction
At time $t$, the agent observes a multimodal state $s_t$:

$$s_t = \{ \mathbf{P}_{t-H:t}, \mathcal{D}_{text} \}$$

Where $\mathbf{P}$ represents the numerical price history and $\mathcal{D}_{text}$ represents the static textual knowledge base.

#### 2. Strategy Alignment
The LLM generates a trading signal $a_t$ (Long/Short/Hold) conditioned on the alignment mode $\psi$:

$$a_t \sim \pi_{\theta}(a_t | s_t, \psi)$$

#### 3. Portfolio Execution
Positions are adjusted to maximize the reward function $r_t$ (Daily Returns):

$$r_t = w_t \cdot \frac{p_{t+1} - p_t}{p_t} - \text{TransactionCosts}$$

---

</div>

</div>