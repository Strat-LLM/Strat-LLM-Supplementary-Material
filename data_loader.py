import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
import random
import re

warnings.filterwarnings("ignore", category=SyntaxWarning)

from config import ExperimentConfig, MarketType, DataPathConfig
from dateutil.relativedelta import relativedelta

class DataLoader:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.market_type = config.market_type
        self.cycle_days = config.trading_days_per_cycle
        self.test_cycles = config.test_cycles
        self.data_paths = config.data_paths

        self.stock_name_map = {}
        self.data_loaded = False
        self.data = None

        print(f"[*] Loading price data for {config.stock_code} ({config.market_type.value})...")
        self.price_data = self._load_price_data()

        self.fundamental_data = None
        if config.include_fundamental:
            print(f"[*] Loading fundamental data...")
            self.fundamental_data = self._load_fundamental_data()

        self.news_data = {}
        if config.include_news:
            print(f"[*] Loading news data...")
            self.news_data = self._load_news_data()

        self.trading_days = sorted(self.price_data['date'].unique())
        self.date_to_idx = {date: idx for idx, date in enumerate(self.trading_days)}
        self.idx_to_date = {idx: date for idx, date in enumerate(self.trading_days)}

        self.data_loaded = True
        self.data = self.price_data

        print(f"[+] Data loaded successfully: {len(self.trading_days)} trading days")
        if len(self.trading_days) > 0:
            print(f"    Range: {self.trading_days[0]} to {self.trading_days[-1]}")

    def _load_price_data(self) -> pd.DataFrame:
        start_date = self.config.start_date
        end_date = self.config.end_date

        print(f"    Date range: {start_date} to {end_date}")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if self.market_type == MarketType.US_STOCK:
            df = self._load_us_price_data(start_dt, end_dt)
        else: 
            # Default to A-share if not US (HK logic removed)
            df = self._load_a_price_data(start_dt, end_dt)

        if df.empty:
            raise ValueError(f"Price data not found for {self.config.stock_code}")

        df = df.sort_values('date')
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        # Exclude weekends
        df = df[df['date'].dt.dayofweek < 5]

        print(f"    Loaded {len(df)} daily records")
        return df.reset_index(drop=True)

    def _load_us_price_data(self, start_dt, end_dt) -> pd.DataFrame:
        price_dfs = []
        current_dt = start_dt

        while current_dt <= end_dt:
            year, month = current_dt.year, current_dt.month
            price_path = self.data_paths.us_price_dir / str(year) / f"{year}-{month:02d}" / f"{self.config.stock_code}_10min.csv"

            if price_path.exists():
                try:
                    df = pd.read_csv(price_path)
                    price_dfs.append(df)
                    print(f"    - Loaded: {price_path.name}")
                except Exception as e:
                    print(f"    ! Error loading {price_path.name}: {e}")

            current_dt = current_dt + relativedelta(months=1)
            
        if not price_dfs:
            return pd.DataFrame()

        df = pd.concat(price_dfs, ignore_index=True)
        df = self._standardize_columns(df)
        df = self._aggregate_to_daily(df)
        return df

    def _load_a_price_data(self, start_dt, end_dt) -> pd.DataFrame:
        stock_code = self.config.stock_code
        
        # Attempt to match A-share file patterns
        price_dir = self.data_paths.a_price_dir
        possible_patterns = [
            f"SH.{stock_code.replace('.SH', '').replace('.SZ', '')}*_10min.csv",
            f"SZ.{stock_code.replace('.SH', '').replace('.SZ', '')}*_10min.csv",
            f"*{stock_code.replace('.SH', '').replace('.SZ', '')}*_10min.csv",
        ]

        price_path = None
        for pattern in possible_patterns:
            matches = list(price_dir.glob(pattern))
            if matches:
                price_path = matches[0]
                # Extract stock name if available in filename
                match = re.search(r'_(.+)_10min\.csv$', price_path.name)
                if match:
                    self.stock_name_map[stock_code] = match.group(1)
                break

        if price_path is None or not price_path.exists():
            print(f"    ! A-share price file not found. Tried patterns: {possible_patterns}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(price_path)
            print(f"    - Loaded: {price_path.name}")

            df = self._standardize_columns(df)
            df = self._aggregate_to_daily(df)
            return df

        except Exception as e:
            print(f"    ! Error loading A-share data: {e}")
            return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = ['datetime', 'time', 'date', 'Date', 'Time', 'Datetime']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            # Fuzzy search for date column
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_col = col
                    break

        if date_col:
            df['date'] = pd.to_datetime(df[date_col])

        col_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
            'CLOSE': 'close', 'VOLUME': 'volume', 'Amount': 'amount'
        }
        df = df.rename(columns=col_mapping)

        return df

    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' not in df.columns:
            raise ValueError("Date column missing in dataset")

        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df['date_only'] = df['date'].dt.date

        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'

        daily_df = df.groupby('date_only').agg(agg_dict).reset_index()
        daily_df['date'] = pd.to_datetime(daily_df['date_only'])
        daily_df = daily_df.drop('date_only', axis=1)

        return daily_df

    def _load_fundamental_data(self) -> str:
        if self.market_type == MarketType.US_STOCK:
            return self._load_us_fundamental()
        else:
            return self._load_a_fundamental()

    def _load_us_fundamental(self) -> str:
        # Load recent annual summaries
        years = [2024, 2023, 2022] if self.config.stock_code in ["NVDA", "WMT"] else [2023, 2022, 2021]

        for year in years:
            path = self.data_paths.us_annual_dir / f"US_{self.config.stock_code}_{year}_summary.json"
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"    - Loaded Annual Report: {path.name}")
                    return self._format_fundamental(data, year)
                except Exception as e:
                    print(f"    ! Error loading {path.name}: {e}")

        return "No fundamental data available."

    def _load_a_fundamental(self) -> str:
        code = self.config.stock_code.replace('.SH', '').replace('.SZ', '')
        code = code.zfill(6)

        path = self.data_paths.a_annual_dir / f"CN_{code}_2024_summary.json"

        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"    - Loaded Annual Report: {path.name}")
                return self._format_fundamental(data, 2024)
            except Exception as e:
                print(f"    ! Error loading {path.name}: {e}")

        return "No fundamental data available."

    def _format_fundamental(self, data: Dict, year: int) -> str:
        # Formatted using English headers for consistency in paper presentation
        lines = [f"=== {self.config.stock_code} {year} Annual Report ==="]

        if 'business_performance' in data:
            lines.append(f"\n[Business Performance]\n{data['business_performance'][:500]}...")
        if 'future_guidance' in data:
            lines.append(f"\n[Future Guidance]\n{data['future_guidance'][:300]}...")
        if 'major_risks' in data:
            risks = data['major_risks']
            if isinstance(risks, list):
                lines.append(f"\n[Major Risks]\n" + "\n".join(f"- {r}" for r in risks[:3]))
        if 'sentiment' in data:
            lines.append(f"\n[Sentiment Analysis] {data['sentiment']}")

        return "\n".join(lines)

    def _load_news_data(self) -> Dict[str, Dict[str, Any]]:
        if self.market_type == MarketType.US_STOCK:
            return self._load_us_news()
        else:
            return self._load_a_news()

    def _load_us_news(self) -> Dict[str, Dict[str, Any]]:
        news_data = {}

        for month in range(1, 7): 
            for ext in ['.parquet', '.csv']:
                path = self.data_paths.us_news_dir / f"{self.config.stock_code}_2025_{month:02d}{ext}"
                if path.exists():
                    try:
                        df = pd.read_parquet(path) if ext == '.parquet' else pd.read_csv(path)
                        news_data.update(self._process_news_df(df))
                        print(f"    - Loaded News: {path.name}")
                        break
                    except Exception as e:
                        print(f"    ! Error loading {path.name}: {e}")

        print(f"    Total news loaded for {len(news_data)} days")
        return news_data

    def _load_a_news(self) -> Dict[str, Dict[str, Any]]:
        news_data = {}
        code = self.config.stock_code.replace('.SH', '').replace('.SZ', '')

        pattern = f"A_{code}_*_2025_*.csv"
        news_files = list(self.data_paths.a_news_dir.glob(pattern))

        for path in news_files:
            try:
                df = pd.read_csv(path, encoding='utf-8-sig')
                news_data.update(self._process_news_df(df))
                print(f"    - Loaded News: {path.name}")
            except Exception as e:
                print(f"    ! Error loading {path.name}: {e}")

        print(f"    Total news loaded for {len(news_data)} days")
        return news_data

    def _process_news_df(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        news_dict = {}

        date_col = None
        for col in ['date', 'Date', 'datetime', 'time']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return news_dict

        df['date'] = pd.to_datetime(df[date_col])

        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')

            major_events = row.get('major_events', [])
            if isinstance(major_events, str):
                try:
                    import ast
                    major_events = ast.literal_eval(major_events)
                except:
                    major_events = [major_events] if major_events else []

            news_dict[date_str] = {
                'summary': str(row.get('summary', '')),
                'sentiment': str(row.get('sentiment', 'Neutral')),
                'major_events': major_events,
                'source_type': str(row.get('source_type', 'Unknown')),
            }

        return news_dict

    def sample_cycles(self) -> List[Tuple[int, int]]:
        total_days = len(self.trading_days)
        cycle_length = self.cycle_days
        num_cycles = self.test_cycles

        max_possible = total_days // cycle_length
        if max_possible == 0:
            print(f"[!] Warning: Insufficient data for one full cycle. Using all data.")
            return [(0, total_days)]

        num_cycles = min(num_cycles, max_possible)

        cycles = []
        if num_cycles * cycle_length <= total_days:
            gap = (total_days - num_cycles * cycle_length) // max(1, num_cycles - 1) if num_cycles > 1 else 0
            start = 0
            for i in range(num_cycles):
                end = start + cycle_length
                cycles.append((start, end))
                start = end + gap
        else:
            step = max(1, (total_days - cycle_length) // num_cycles)
            for i in range(num_cycles):
                start = i * step
                cycles.append((start, min(start + cycle_length, total_days)))

        print(f"\n[+] Sampled {len(cycles)} testing cycles:")
        for i, (s, e) in enumerate(cycles):
            sd = self.idx_to_date.get(s, "N/A")
            ed = self.idx_to_date.get(e-1, "N/A")
            print(f"    Cycle {i+1}: {sd} ~ {ed} ({e-s} days)")

        return cycles

    def get_market_data(self, day_idx: int) -> Dict[str, Any]:
        if day_idx < 0 or day_idx >= len(self.trading_days):
            raise IndexError(f"Index out of bounds: {day_idx}")

        date = self.trading_days[day_idx]
        row = self.price_data[self.price_data['date'] == date].iloc[0]

        return {
            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row.get('volume', 0)) if 'volume' in row else 0
        }

    def get_context(self, day_idx: int) -> Tuple[str, int]:
        context_days = getattr(self.config, 'context_days', 3)
        start_idx = max(0, day_idx - context_days)

        lines = []
        for i in range(start_idx, day_idx):
            try:
                data = self.get_market_data(i)
                lines.append(f"T-{day_idx-i}: {data['date']} | O:{data['open']:.2f} H:{data['high']:.2f} L:{data['low']:.2f} C:{data['close']:.2f}")
            except:
                continue

        return "\n".join(lines) if lines else "No history", len(lines)

    def get_news_summary(self, date_str: str) -> Dict[str, Any]:
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d')
        return self.news_data.get(str(date_str), {})

    def get_fundamental_data(self) -> str:
        return self.fundamental_data or "No data"