"""
plotting.py
===========
Reusable matplotlib/seaborn chart helpers for FactorLens ETF Style Drift.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd


# ---------------------------------------------------------------------------
# Default style
# ---------------------------------------------------------------------------

PALETTE = sns.color_palette("tab10")

def set_style() -> None:
    """Apply a clean, publication-ready plot style."""
    sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Rolling factor loadings
# ---------------------------------------------------------------------------

def plot_rolling_loadings(
    rolling: pd.DataFrame,
    ticker: str,
    factors: list[str] | None = None,
    window_label: str = "36-month",
    save_path: Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Line chart of rolling FF5 factor loadings for a single ETF.

    Parameters
    ----------
    rolling : pd.DataFrame
        Rolling coefficient DataFrame (index=date, cols=factor names).
    ticker : str
        ETF ticker (used in the title).
    factors : list of str or None
        Which factor columns to plot; defaults to Mkt-RF, SMB, HML, RMW, CMA.
    window_label : str
        Human-readable window size for the title.
    save_path : Path or None
        If given, save the figure there.
    ax : Axes or None
        Existing axes to draw on; creates a new figure if None.

    Returns
    -------
    matplotlib Axes
    """
    if factors is None:
        factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))

    for factor in factors:
        if factor in rolling.columns:
            ax.plot(rolling.index, rolling[factor], label=factor, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{ticker} — {window_label} Rolling FF5 Loadings", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Factor loading (β)")
    ax.legend(ncol=3, fontsize=9)
    ax.xaxis.set_major_locator(mticker.AutoLocator())
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved → {save_path}")

    return ax


# ---------------------------------------------------------------------------
# Drift magnitude over time
# ---------------------------------------------------------------------------

def plot_drift_magnitude(
    drift_df: pd.DataFrame,
    tickers: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Axes:
    """
    Line chart of drift magnitude (L2-norm of Δbeta) for multiple ETFs.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Output of style_drift.compute_drift_magnitude_all().
    tickers : list of str or None
        Which tickers to plot; defaults to all columns.
    save_path : Path or None
        Optional output path.
    """
    if tickers is None:
        tickers = list(drift_df.columns)

    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        if ticker in drift_df.columns:
            ax.plot(drift_df.index, drift_df[ticker], label=ticker, linewidth=1.4)

    ax.set_title("ETF Style Drift Magnitude Over Time", fontsize=13)
    ax.set_ylabel("Drift magnitude (L2 norm of Δβ)")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved → {save_path}")

    return ax


# ---------------------------------------------------------------------------
# Full-period loading heatmap
# ---------------------------------------------------------------------------

def plot_loading_heatmap(
    summary: pd.DataFrame,
    factors: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Axes:
    """
    Heatmap of full-period FF5 loadings for all ETFs.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of regression.run_all_etfs() — rows=tickers, cols include factors.
    factors : list of str or None
        Which factor columns to show; defaults to alpha + FF5 factors.
    save_path : Path or None
    """
    if factors is None:
        factors = ["alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    data = summary[factors].copy()

    fig, ax = plt.subplots(figsize=(9, len(data) * 0.6 + 1.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="RdBu_r",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("FF5 Factor Loadings — Full Period", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved → {save_path}")

    return ax


# ---------------------------------------------------------------------------
# Style-box timeline
# ---------------------------------------------------------------------------

def plot_style_timeline(
    style_df: pd.DataFrame,
    tickers: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    One subplot per ETF showing how its style-box label changes over time.

    Parameters
    ----------
    style_df : pd.DataFrame
        Output of style_drift.classify_style_all() — cols=tickers, values=labels.
    tickers : list of str or None
        Subset of ETFs to plot.
    save_path : Path or None
    """
    if tickers is None:
        tickers = list(style_df.columns)

    style_order = [
        "Large-Value", "Large-Blend", "Large-Growth",
        "Mid-Value",   "Mid-Blend",   "Mid-Growth",
        "Small-Value", "Small-Blend", "Small-Growth",
    ]
    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(style_order)}

    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        series = style_df[ticker].dropna()
        for date, style in series.items():
            ax.bar(date, 1, width=25, color=color_map.get(style, "grey"), align="center")
        ax.set_yticks([])
        ax.set_ylabel(ticker, rotation=0, labelpad=40, va="center", fontsize=10)

    fig.suptitle("Rolling Style-Box Classification", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig
