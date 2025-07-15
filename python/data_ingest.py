import pandas as pd


def load_tick_data(path: str) -> pd.DataFrame:
    """
    Load tick-level order-book data from Parquet or CSV.

    Args:
        path: Filesystem path to the tick data file.

    Returns:
        DataFrame with columns ['timestamp', 'bid', 'ask', 'bid_size', 'ask_size', ...].
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def resample_bars(
    df: pd.DataFrame,
    freq: str = "1S"
) -> pd.DataFrame:
    """
    Resample tick data into OHLC bars at given frequency.

    Args:
        df: Tick data DataFrame with a datetime index or 'timestamp' column.
        freq: Pandas offset alias (e.g., '1S', '1T').

    Returns:
        DataFrame indexed by timestamp with columns ['open', 'high', 'low', 'close', 'volume'].
    """
    df = df.set_index('timestamp')
    ohlc = df['mid_price'].resample(freq).ohlc()
    volume = df['size'].resample(freq).sum().rename('volume')
    bars = ohlc.join(volume).dropna()
    return bars
