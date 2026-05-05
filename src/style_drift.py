"""
style_drift.py
==============
Utilities for measuring and classifying ETF style drift from
rolling FF5 factor loadings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def compute_drift_magnitude(
    rolling: pd.DataFrame,
    factors: list[str] | None = None,
) -> pd.Series:
    """
    Euclidean norm of the change in factor loadings period-over-period.

    Parameters
    ----------
    rolling : pd.DataFrame
        Rolling coefficient DataFrame (output of regression.rolling_ff5).
    factors : list of str or None
        Factor columns to include; defaults to FACTORS.

    Returns
    -------
    pd.Series  (name='drift_magnitude')
    """
    if factors is None:
        factors = FACTORS
    deltas = rolling[factors].diff()
    drift = np.sqrt((deltas ** 2).sum(axis=1))
    drift.name = "drift_magnitude"
    return drift


def compute_drift_magnitude_all(
    rolling_dict: dict[str, pd.DataFrame],
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """Drift magnitude for every ETF. Returns wide DataFrame (cols=tickers)."""
    return pd.DataFrame(
        {ticker: compute_drift_magnitude(df, factors)
         for ticker, df in rolling_dict.items()}
    )


def classify_style(
    rolling: pd.DataFrame,
    smb_col: str = "SMB",
    hml_col: str = "HML",
) -> pd.Series:
    """
    Classify each period into a 3×3 Morningstar-style box.

    SMB > 0.2 → Small | SMB < -0.2 → Large | else Mid
    HML > 0.2 → Value | HML < -0.2 → Growth | else Blend

    Returns
    -------
    pd.Series of str  e.g. 'Large-Value', 'Small-Growth', 'Mid-Blend'
    """
    def _size(x):
        return "Small" if x > 0.2 else ("Large" if x < -0.2 else "Mid")

    def _value(x):
        return "Value" if x > 0.2 else ("Growth" if x < -0.2 else "Blend")

    style = rolling[smb_col].map(_size) + "-" + rolling[hml_col].map(_value)
    style.name = "style_box"
    return style


def classify_style_all(
    rolling_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Style classification for all ETFs. Returns wide DataFrame."""
    return pd.DataFrame(
        {ticker: classify_style(df) for ticker, df in rolling_dict.items()}
    )


def drift_zscore(drift_magnitude: pd.Series, window: int = 36) -> pd.Series:
    """Rolling z-score of drift magnitude to normalise across ETFs."""
    roll = drift_magnitude.rolling(window, min_periods=window // 2)
    z = (drift_magnitude - roll.mean()) / roll.std()
    z.name = "drift_zscore"
    return z


def summary_drift_stats(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics of drift magnitude per ETF.

    Returns
    -------
    pd.DataFrame  (rows=tickers, cols=mean/std/max/p75/p90)
    """
    stats = pd.DataFrame({
        "mean": drift_df.mean(),
        "std":  drift_df.std(),
        "max":  drift_df.max(),
        "p75":  drift_df.quantile(0.75),
        "p90":  drift_df.quantile(0.90),
    })
    stats.index.name = "ticker"
    return stats.sort_values("mean", ascending=False)
