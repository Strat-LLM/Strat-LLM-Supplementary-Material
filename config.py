import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

class MarketType(Enum):
    US_STOCK = "us_stock"
    A_STOCK = "a_stock"
    HK_STOCK = "hk_stock"

class CycleType(Enum):
    SHORT_CYCLE = 18
    LONG_CYCLE = 93

class DataLevel(Enum):
    PRICE_ONLY = "price_only"
    PRICE_NEWS = "price_news"
    FULL_DATA = "full_data"

class StrategyMode(Enum):
    STRICT_STRATEGIES = "strict_strategies"
    GUIDED_STRATEGIES = "guided_strategies"
    FREE_STRATEGIES = "free_strategies"

class Attitude(Enum):
    CONSERVATIVE = "conservative"
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"

MODEL_ALIASES = {
}

@dataclass
class ModelConfig:
    model_name: str
    api_base: str = ""
    api_key: str = os.getenv("LLM_API_KEY", "")
    temperature: float = 0.7
    max_tokens: int = 4096

@dataclass
class ExperimentConfig:
    market_type: MarketType = MarketType.US_STOCK
    cycle_type: CycleType = CycleType.SHORT_CYCLE
    data_level: DataLevel = DataLevel.FULL_DATA
    strategy_mode: StrategyMode = StrategyMode.STRICT_STRATEGIES
    attitude: Attitude = Attitude.AGGRESSIVE
    initial_capital: float = 100000.0
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig(model_name="qwen3_80b"))
    
    base_path: Path = field(default_factory=lambda: Path(__file__).parent)
    data_path: Path = field(default_factory=lambda: Path("data"))
    result_path: Path = field(default_factory=lambda: Path("results"))

    def get_experiment_id(self) -> str:
        market_map = {'us_stock': 'us', 'a_stock': 'cn', 'hk_stock': 'hk'}
        cycle_map = {18: '15', 93: '90'}
        data_map = {'price_only': 'p', 'price_news': 'n', 'full_data': 'f'}
        strategy_map = {'strict_strategies': 's', 'guided_strategies': 'g', 'free_strategies': 'x'}
        attitude_map = {'conservative': 'c', 'neutral': 'n', 'aggressive': 'a'}

        parts = [
            market_map.get(self.market_type.value, 'us'),
            cycle_map.get(self.cycle_type.value, '15'),
            str(int(self.initial_capital / 10000)),
            data_map.get(self.data_level.value, 'f'),
            strategy_map.get(self.strategy_mode.value, 's'),
            attitude_map.get(self.attitude.value, 'a'),
        ]

        return '_'.join(parts)