import pandas as pd
import numpy as np


def compute_features(
    bars: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Compute rolling features for time-series bars.

    Features include:
      - mid-price returns
      - VWAP
      - realized volatility
      - order-book imbalance

    Args:
        bars: DataFrame with ['open', 'high', 'low', 'close', 'volume'] indexed by timestamp.
        window: Rolling window size in periods.

    Returns:
        DataFrame with original columns plus feature columns.
    """
    df = bars.copy()
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['return'] = df['mid_price'].pct_change()

    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(window).sum() / \
                 df['volume'].rolling(window).sum()

    # Realized volatility
    df['rv'] = df['return'].rolling(window).std() * np.sqrt(window)

    # Imbalance (placeholder: (bid_size - ask_size) / (bid_size + ask_size))
    # Here, assume bars has bid_size and ask_size columns
    if 'bid_size' in df.columns and 'ask_size' in df.columns:
        df['imbalance'] = (
            df['bid_size'] - df['ask_size']
        ) / (df['bid_size'] + df['ask_size'])

    df = df.dropna()
    return df


if __name__ == "__main__":
    # Quick sanity check when run as script
    import sys
    path = sys.argv[1]
    bars = pd.read_parquet(path)
    feats = compute_features(bars)
    print(feats.tail())
