import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional
import warnings

def calculate_annualized_return(total_return: float, days: int) -> float:
    if days <= 0:
        return 0.0

    if days < 5:
        daily_return = total_return / days if days > 0 else 0
        annualized_ret = daily_return * 252
        return annualized_ret

    annual_factor = 252.0 / days

    if abs(total_return) > 5.0:
        log_return = np.log1p(total_return)
        annualized_log_return = log_return * annual_factor
        annualized_ret = np.expm1(annualized_log_return)
        return annualized_ret

    annualized_ret = (1 + total_return) ** annual_factor - 1

    return annualized_ret


def calculate_alpha_beta(agent_returns: List[float],
                         benchmark_returns: List[float]) -> Tuple[float, float]:
    if len(agent_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0, 1.0

    if len(agent_returns) != len(benchmark_returns):
        min_length = min(len(agent_returns), len(benchmark_returns))
        agent_returns = agent_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]

    try:
        beta, alpha, r_value, p_value, std_err = stats.linregress(
            benchmark_returns, agent_returns
        )

        annualized_alpha = alpha * 252

        return annualized_alpha, beta
    except:
        return 0.0, 1.0


def calculate_volatility_sharpe(daily_returns: List[float],
                                risk_free_rate: float = 0.0) -> Tuple[float, float]:
    if len(daily_returns) < 2:
        return 0.0, 0.0

    daily_vol = np.std(daily_returns, ddof=1)
    annual_vol = daily_vol * np.sqrt(252)

    mean_daily_return = np.mean(daily_returns)
    daily_rf = risk_free_rate / 252.0
    excess_return = mean_daily_return - daily_rf

    if daily_vol == 0:
        sharpe = 0.0
    else:
        sharpe = (excess_return / daily_vol) * np.sqrt(252)

    return annual_vol, sharpe


def calculate_max_drawdown(equity_values: List[float]) -> float:
    if not equity_values or len(equity_values) < 2:
        return 0.0

    peak = equity_values[0]
    max_dd = 0.0

    for value in equity_values:
        if value > peak:
            peak = value

        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown

    return max_dd


def calculate_sortino_ratio(daily_returns: List[float],
                            risk_free_rate: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0

    mean_return = np.mean(daily_returns)
    daily_rf = risk_free_rate / 252.0
    excess_return = mean_return - daily_rf

    daily_returns_np = np.array(daily_returns)
    excess_returns_series = daily_returns_np - daily_rf
    downside_excess = np.clip(excess_returns_series, -np.inf, 0)

    downside_variance = np.mean(downside_excess ** 2)
    downside_deviation = np.sqrt(downside_variance)

    if downside_deviation == 0:
        sortino = 0.0
    else:
        sortino = (excess_return / downside_deviation) * np.sqrt(252)

    return sortino


def calculate_calmar_ratio(annualized_return: float,
                           max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0

    return annualized_return / max_drawdown


def calculate_win_rate(trade_results: List[Dict[str, any]]) -> float:
    if not trade_results:
        return 0.0

    profitable_trades = 0
    total_trades = 0

    for trade in trade_results:
        action = trade.get('action')
        if action in ['BUY', 'SELL']:
            total_trades += 1
            net_proceeds = trade.get('net_proceeds', 0)
            if net_proceeds > 0:
                profitable_trades += 1

    if total_trades == 0:
        return 0.0

    return profitable_trades / total_trades


def calculate_profit_factor(trade_results: List[Dict[str, any]]) -> float:
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in trade_results:
        net_proceeds = trade.get('net_proceeds', 0)
        if net_proceeds > 0:
            gross_profit += net_proceeds
        elif net_proceeds < 0:
            gross_loss += abs(net_proceeds)

    if gross_loss == 0:
        return float('inf')

    return gross_profit / gross_loss


def calculate_trade_statistics(trade_results: List[Dict[str, any]]) -> Dict[str, float]:
    if not trade_results:
        return {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'hold_trades': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0
        }

    total_trades = len(trade_results)
    buy_trades = sum(1 for t in trade_results if t.get('action') == 'BUY')
    sell_trades = sum(1 for t in trade_results if t.get('action') == 'SELL')
    hold_trades = sum(1 for t in trade_results if t.get('action') in ['HOLD', 'HOLD_LONG', 'WAIT'])

    profits = [t.get('net_proceeds', 0) for t in trade_results]
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]

    avg_profit = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    largest_win = max(winning_trades) if winning_trades else 0
    largest_loss = min(losing_trades) if losing_trades else 0

    return {
        'total_trades': total_trades,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'hold_trades': hold_trades,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss
    }


def calculate_all_metrics(equity_values: List[float],
                          daily_returns: List[float],
                          trade_results: List[Dict[str, any]],
                          benchmark_returns: Optional[List[float]] = None,
                          risk_free_rate: float = 0.0) -> Dict[str, float]:
    if not equity_values or not daily_returns:
        return {}

    if equity_values[0] > 0:
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
    else:
        total_return = 0.0

    annualized_return = calculate_annualized_return(total_return, len(daily_returns))
    annual_vol, sharpe = calculate_volatility_sharpe(daily_returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(equity_values)
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate)
    calmar = calculate_calmar_ratio(annualized_return, max_drawdown)
    win_rate = calculate_win_rate(trade_results)
    profit_factor = calculate_profit_factor(trade_results)

    trade_stats = calculate_trade_statistics(trade_results)

    alpha, beta = 0.0, 1.0
    if benchmark_returns and len(benchmark_returns) >= 2:
        alpha, beta = calculate_alpha_beta(daily_returns, benchmark_returns)

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annual_volatility': annual_vol,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'alpha': alpha,
        'beta': beta,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        **trade_stats,
        'total_trading_days': len(daily_returns),
        'initial_equity': equity_values[0] if equity_values else 0.0,
        'final_equity': equity_values[-1] if equity_values else 0.0
    }