import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from config import ExperimentConfig, StrategyMode, OrderType
from broker import Broker, Order
from data_loader import DataLoader
from metrics import calculate_annualized_return, calculate_max_drawdown, calculate_sharpe_ratio


class StrategyAgent:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def get_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": "hold",
            "reason": "Agent logic hidden for privacy",
            "confidence": 0.0
        }


class SimulationRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.loader = DataLoader(config)
        self.broker = Broker(config)
        self.agent = StrategyAgent(config)
        self.results_path = config.result_path
        self.results_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        trading_days = self.loader.trading_days
        confidence_scores = []

        pbar = tqdm(total=len(trading_days), desc=f"Running {self.config.strategy_mode.value}")

        for day_idx, current_date in enumerate(trading_days):
            market_data = self.loader.get_market_data(day_idx)
            context_str, context_len = self.loader.get_context(day_idx)

            current_price = market_data['close']
            self.broker.update_daily_status(current_price, str(current_date))

            state = {
                "date": str(current_date),
                "market_data": market_data,
                "context": context_str,
                "portfolio": {
                    "cash": self.broker.cash,
                    "shares": self.broker.shares,
                    "equity": self.broker.current_equity,
                    "holdings_val": self.broker.shares * current_price
                }
            }

            decision = self.agent.get_decision(state)

            action = decision.get("action", "hold").lower()
            confidence = decision.get("confidence", 0.0)
            reason = decision.get("reason", "")

            confidence_scores.append(confidence)

            if action == "buy":
                max_shares = self.broker.get_max_buyable_shares(current_price)
                if max_shares > 0:
                    order = Order(
                        order_type=OrderType.BUY,
                        shares=max_shares,
                        price=current_price,
                        timestamp=str(current_date),
                        reason=reason,
                        confidence=confidence
                    )
                    self.broker.execute_order(order)

            elif action == "sell":
                sellable = self.broker.get_sellable_shares()
                if sellable > 0:
                    order = Order(
                        order_type=OrderType.SELL,
                        shares=sellable,
                        price=current_price,
                        timestamp=str(current_date),
                        reason=reason,
                        confidence=confidence
                    )
                    self.broker.execute_order(order)

            pbar.update(1)
            pbar.set_postfix({'Equity': f"{self.broker.current_equity:.2f}"})

        self._finalize_experiment(confidence_scores)

    def _finalize_experiment(self, confidence_scores: List[float]):
        metrics = self.broker.get_metrics()

        equity_curve = self.broker.history_equity
        returns = pd.Series(equity_curve).pct_change().dropna().values

        strat_metrics = {
            "final_equity": self.broker.current_equity,
            "total_return": (self.broker.current_equity - self.broker.initial_capital) / self.broker.initial_capital,
            "annualized_return": calculate_annualized_return(
                (self.broker.current_equity - self.broker.initial_capital) / self.broker.initial_capital,
                len(equity_curve)),
            "max_drawdown": calculate_max_drawdown(equity_curve),
            "sharpe_ratio": calculate_sharpe_ratio(returns),
            "cognitive_dissonance": np.std(confidence_scores) if confidence_scores else 0.0,
            "trade_count": metrics['total_trades']
        }

        df_history = pd.DataFrame(metrics['trade_history'])
        history_file = self.results_path / f"{self.config.get_experiment_id()}_trades.csv"
        df_history.to_csv(history_file, index=False)

        summary_file = self.results_path / f"{self.config.get_experiment_id()}_metrics.json"
        pd.Series(strat_metrics).to_json(summary_file)


if __name__ == "__main__":
    config = ExperimentConfig(
        strategy_mode=StrategyMode.STRICT_STRATEGIES
    )
    runner = SimulationRunner(config)
    runner.run()