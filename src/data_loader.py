"""
data_loader.py
==============
Functions to download and load:
  - ETF adjusted-close prices via yfinance
  - Fama-French 5-Factor monthly data from Ken French's website
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = [
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100 / growth-tech
    "IWM",  # Russell 2000 / small-cap
    "VTV",  # large-cap value
    "VUG",  # large-cap growth
    "QUAL",  # quality factor
    "MTUM",  # momentum factor
    "USMV",  # minimum volatility
    "VLUE",  # value factor
]

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

FF5_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)

# ---------------------------------------------------------------------------
# ETF price helpers
# ---------------------------------------------------------------------------


def download_etf_prices(
    tickers: list[str] = TICKERS,
    start: str = "2015-01-01",
    end: str | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for *tickers* from yfinance.

    Parameters
    ----------
    tickers : list of str
        ETF ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str or None
        End date in 'YYYY-MM-DD' format; None = today.
    save_path : Path or None
        If provided, save the raw daily prices as a CSV at this path.

    Returns
    -------
    pd.DataFrame
        Daily adjusted-close prices (columns = tickers, index = date).
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns MultiIndex columns when >1 ticker; keep only "Close"
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(save_path)
        print(f"Saved daily prices → {save_path}")

    return prices


def load_etf_prices(path: Path) -> pd.DataFrame:
    """Load previously saved daily ETF prices from a CSV file."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def to_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end prices (last trading day of each month).

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Month-end prices.
    """
    return prices.resample("ME").last()


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from month-end prices.

    Returns
    -------
    pd.DataFrame
        Monthly simple returns (NaN for the first period).
    """
    monthly = to_monthly_prices(prices)
    return monthly.pct_change()


# ---------------------------------------------------------------------------
# Fama-French 5-Factor data helpers
# ---------------------------------------------------------------------------


def download_ff5_monthly(
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Download Fama-French 5-Factor *monthly* data from Ken French's website.

    The returned DataFrame contains columns:
        Mkt-RF, SMB, HML, RMW, CMA, RF
    All values are in decimal form (divided by 100).

    Parameters
    ----------
    save_path : Path or None
        If provided, save the cleaned data as a CSV at this path.

    Returns
    -------
    pd.DataFrame
        Monthly FF5 factors with a DatetimeIndex (month-end).
    """
    ff = pd.read_csv(
        FF5_MONTHLY_URL,
        skiprows=3,
        index_col=0,
    )

    # Keep only rows whose index is a 6-digit YYYYMM number (drop annual block)
    valid = ff.index.astype(str).str.strip().str.match(r"^\d{6}$")
    ff = ff.loc[valid]

    ff.index = pd.to_datetime(ff.index.astype(str).str.strip(), format="%Y%m")
    ff.index = ff.index + pd.offsets.MonthEnd(0)  # snap to month-end
    ff.index.name = "date"

    ff.columns = ff.columns.str.strip()
    ff = ff.apply(pd.to_numeric, errors="coerce") / 100

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ff.to_csv(save_path)
        print(f"Saved FF5 monthly factors → {save_path}")

    return ff


def load_ff5_monthly(path: Path) -> pd.DataFrame:
    """Load previously saved FF5 monthly factors from a CSV file."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_etf_ff5(
    etf_returns: pd.DataFrame,
    ff5: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ETF monthly returns with FF5 factors and return long-format data.

    Input
    -----
    etf_returns:
        Wide monthly ETF returns.
        Index = date
        Columns = tickers
        Example columns: SPY, QQQ, IWM

    ff5:
        Fama-French 5 factors.
        Index = date
        Columns = Mkt-RF, SMB, HML, RMW, CMA, RF

    Output
    ------
    Long-format DataFrame:
        date
        ticker
        return
        mkt_rf
        smb
        hml
        rmw
        cma
        rf
        excess_return
    """

    # Copy to avoid modifying original data
    etf = etf_returns.copy()
    factors = ff5.copy()

    # Make sure index is datetime
    etf.index = pd.to_datetime(etf.index)
    factors.index = pd.to_datetime(factors.index)

    # Set index name for clean reset_index()
    etf.index.name = "date"
    factors.index.name = "date"

    # Convert ETF returns from wide to long
    etf_long = etf.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="return"
    )

    # Clean factor column names
    factors_clean = factors.reset_index().rename(
        columns={
            "Mkt-RF": "mkt_rf",
            "SMB": "smb",
            "HML": "hml",
            "RMW": "rmw",
            "CMA": "cma",
            "RF": "rf",
        }
    )

    # Merge ETF returns with factor data by date
    df = etf_long.merge(factors_clean, on="date", how="inner")

    # Calculate excess return
    df["excess_return"] = df["return"] - df["rf"]

    # Clean final order
    df = df[
        [
            "date",
            "ticker",
            "return",
            "mkt_rf",
            "smb",
            "hml",
            "rmw",
            "cma",
            "rf",
            "excess_return",
        ]
    ]

    return df.dropna().sort_values(["ticker", "date"]).reset_index(drop=True)
