import pandas as pd
import numpy as np
import pytest
from feature_engineering import compute_features


def make_dummy_bars(n=100):
    idx = pd.date_range("2021-01-01", periods=n, freq="1T")
    df = pd.DataFrame({
        'open': np.random.rand(n),
        'high': np.random.rand(n),
        'low': np.random.rand(n),
        'close': np.random.rand(n),
        'volume': np.random.rand(n),
        'bid_size': np.random.rand(n),
        'ask_size': np.random.rand(n)
    }, index=idx)
    return df


def test_compute_features():
    bars = make_dummy_bars()
    feats = compute_features(bars, window=5)
    # Expect no NaNs
    assert not feats.isnull().any().any()
    # Check feature columns present
    for col in ['return', 'vwap', 'rv', 'imbalance']:
        assert col in feats.columns
