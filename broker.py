from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Order:
    order_type: OrderType
    shares: int
    price: float
    timestamp: str = ""
    reason: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_type': self.order_type.value,
            'shares': self.shares,
            'price': self.price,
            'timestamp': self.timestamp,
            'reason': self.reason,
            'confidence': self.confidence,
        }


@dataclass
class TradeRecord:
    timestamp: str
    order_type: str
    shares: int
    price: float
    amount: float
    commission: float
    total_cost: float
    cash_after: float
    shares_after: int
    equity_after: float
    reason: str = ""
    confidence: float = 0.0


class Broker:
    def __init__(self, config):
        self.config = config
        self.market_config = config.market_config

        self.cash = config.initial_capital
        self.shares = 0
        self.initial_capital = config.initial_capital

        self.current_equity = config.initial_capital
        self.history_equity = [config.initial_capital]
        self.peak_equity = config.initial_capital
        self.max_drawdown = 0.0

        self.trade_history: List[TradeRecord] = []
        self.order_history: List[Order] = []

        self.commission_rate = self.market_config.commission_rate
        self.min_commission = self.market_config.min_commission
        self.stamp_duty = self.market_config.stamp_duty
        self.platform_fee = self.market_config.platform_fee
        self.settlement_fee = self.market_config.settlement_fee

        self.lot_size = self.market_config.lot_size
        self.t_plus_n = self.market_config.t_plus_n

        self.locked_shares = 0
        self.last_buy_date = None

    def calculate_commission(self, amount: float, is_sell: bool = False) -> float:
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp = amount * self.stamp_duty if is_sell else 0
        platform = self.platform_fee
        settlement = amount * self.settlement_fee
        total = commission + stamp + platform + settlement
        return round(total, 2)

    def get_max_buyable_shares(self, price: float) -> int:
        if price <= 0:
            return 0

        estimated_commission_rate = max(0.01, self.commission_rate + self.stamp_duty)
        available = self.cash / (1 + estimated_commission_rate)
        max_shares = int(available / price)

        if self.lot_size > 1:
            max_shares = (max_shares // self.lot_size) * self.lot_size

        return max_shares

    def get_sellable_shares(self) -> int:
        if self.t_plus_n == 1:
            return self.shares - self.locked_shares
        return self.shares

    def execute_order(self, order: Order) -> bool:
        self.order_history.append(order)

        if order.order_type == OrderType.HOLD:
            return True

        if order.order_type == OrderType.BUY:
            return self._execute_buy(order)
        elif order.order_type == OrderType.SELL:
            return self._execute_sell(order)
        return False

    def _execute_buy(self, order: Order) -> bool:
        cost = order.shares * order.price
        commission = self.calculate_commission(cost, is_sell=False)
        total_cost = cost + commission

        if total_cost > self.cash:
            return False

        self.cash -= total_cost
        self.shares += order.shares
        self.locked_shares += order.shares
        self.last_buy_date = order.timestamp

        self._record_trade(order, cost, commission, total_cost)
        return True

    def _execute_sell(self, order: Order) -> bool:
        if order.shares > self.get_sellable_shares():
            return False

        revenue = order.shares * order.price
        commission = self.calculate_commission(revenue, is_sell=True)
        net_revenue = revenue - commission

        self.cash += net_revenue
        self.shares -= order.shares

        self._record_trade(order, revenue, commission, -net_revenue)
        return True

    def _record_trade(self, order: Order, amount: float, commission: float, total_cost: float):
        self.current_equity = self.cash + (self.shares * order.price)

        record = TradeRecord(
            timestamp=order.timestamp,
            order_type=order.order_type.value,
            shares=order.shares,
            price=order.price,
            amount=amount,
            commission=commission,
            total_cost=total_cost,
            cash_after=self.cash,
            shares_after=self.shares,
            equity_after=self.current_equity,
            reason=order.reason,
            confidence=order.confidence
        )
        self.trade_history.append(record)

    def update_daily_status(self, current_price: float, timestamp: str):
        self.current_equity = self.cash + (self.shares * current_price)
        self.history_equity.append(self.current_equity)

        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        if self.t_plus_n == 1 and self.last_buy_date != timestamp:
            self.locked_shares = 0

    def get_metrics(self) -> Dict[str, Any]:
        buy_trades = [t for t in self.trade_history if t.order_type == "buy"]
        sell_trades = [t for t in self.trade_history if t.order_type == "sell"]

        total_commission = sum(t.commission for t in self.trade_history)

        winning_trades = sum(1 for t in sell_trades if t.total_cost < 0)

        return {
            'total_trades': len(self.trade_history),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commission': total_commission,
            'winning_trades': winning_trades,
            'trade_history': [t.__dict__ for t in self.trade_history[-10:]],
        }

    def reset(self):
        self.cash = self.initial_capital
        self.shares = 0
        self.current_equity = self.initial_capital
        self.history_equity = [self.initial_capital]
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.trade_history = []
        self.order_history = []
        self.locked_shares = 0
        self.last_buy_date = None