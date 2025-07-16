import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pyarrow

def generate_tick_data(
    start_time: str = "2021-01-01 09:30:00",
    end_time: str   = "2021-01-01 16:00:00",
    avg_ticks_per_sec: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate tick-level order-book data for one trading day.

    Args:
        start_time: Trading session start (inclusive).
        end_time: Trading session end (exclusive).
        avg_ticks_per_sec: Average number of ticks generated per second.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
         - timestamp: pandas.Timestamp
         - bid: Best bid price
         - ask: Best ask price
         - bid_size: Size at bid
         - ask_size: Size at ask
         - mid_price: (bid + ask) / 2
         - size: Trade size at tick
    """
    np.random.seed(seed)

    # Parse times
    t0 = pd.to_datetime(start_time)
    t1 = pd.to_datetime(end_time)
    total_seconds = int((t1 - t0).total_seconds())

    # Estimate total ticks
    n_ticks = int(total_seconds * avg_ticks_per_sec)

    # Generate random timestamps (sorted)
    offsets = np.sort(np.random.uniform(0, total_seconds, size=n_ticks))
    timestamps = [t0 + timedelta(seconds=float(s)) for s in offsets]

    # Simulate mid-price as random walk
    mid = np.cumsum(np.random.normal(scale=0.02, size=n_ticks)) + 100.0

    # Simulate spread and derive bid/ask
    spread = np.abs(np.random.normal(loc=0.01, scale=0.005, size=n_ticks))
    bid = mid - spread / 2
    ask = mid + spread / 2

    # Simulate sizes
    bid_size = np.random.poisson(lam=50, size=n_ticks) + 1
    ask_size = np.random.poisson(lam=50, size=n_ticks) + 1
    size = np.random.poisson(lam=10, size=n_ticks) + 1

    df = pd.DataFrame({
        "timestamp": timestamps,
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid_price": mid,
        "size": size,
    })

    return df

if __name__ == "__main__":
    # Create output directory
    os.makedirs("data", exist_ok=True)

    # Generate and save
    df_ticks = generate_tick_data(
        start_time="2021-01-01 09:30:00",
        end_time="2021-01-01 16:00:00",
        avg_ticks_per_sec=2.0,
        seed=123
    )
    out_path = "data/market_ticks.parquet"
    df_ticks.to_parquet(out_path, index=False)
    print(f"Generated {len(df_ticks)} ticks → {out_path}")
