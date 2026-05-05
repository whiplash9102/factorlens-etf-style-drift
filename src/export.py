"""
export.py
=========
Utilities to export results for reporting (CSV, Excel, Tableau-ready).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def export_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    print(f"Exported CSV → {path}")


def export_excel(
    sheets: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    """
    Write multiple DataFrames to separate sheets in one Excel workbook.

    Parameters
    ----------
    sheets : dict
        Keys = sheet names, values = DataFrames.
    path : Path
        Output .xlsx file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31])  # Excel max 31 chars
    print(f"Exported Excel → {path}")


def build_tableau_export(
    rolling_dict: dict[str, pd.DataFrame],
    drift_df: pd.DataFrame,
    style_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine rolling loadings, drift magnitude and style label into one
    long-format DataFrame ready for Tableau or similar BI tools.

    Returns
    -------
    pd.DataFrame with columns:
        date, ticker, Mkt-RF, SMB, HML, RMW, CMA, alpha,
        r_squared, drift_magnitude, style_box
    """
    frames = []
    for ticker, df in rolling_dict.items():
        tmp = df.copy()
        tmp["ticker"] = ticker
        if ticker in drift_df.columns:
            tmp["drift_magnitude"] = drift_df[ticker]
        if ticker in style_df.columns:
            tmp["style_box"] = style_df[ticker]
        frames.append(tmp.reset_index())

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"index": "date"})
    # Ensure date column is named correctly after reset_index
    if "date" not in combined.columns and combined.columns[0] != "date":
        combined = combined.rename(columns={combined.columns[0]: "date"})
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined
