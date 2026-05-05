# FactorLens: ETF Style Drift Analysis

FactorLens is a small research project for studying whether popular US ETFs keep the same factor profile over time.

The project uses monthly ETF returns and the Fama French five factor model to estimate exposure to market, size, value, profitability, and investment factors. It also calculates rolling factor betas so changes in style can be tracked over time and exported into Tableau.

## What This Project Does

1. Downloads ETF price data with `yfinance`.
2. Converts daily prices into monthly returns.
3. Loads monthly Fama French five factor data.
4. Merges ETF returns with factor data.
5. Runs full period factor regressions.
6. Runs rolling regressions to track style drift through time.
7. Exports clean, Tableau ready CSV files for dashboard work.

## ETFs Covered

`SPY`, `QQQ`, `IWM`, `VTV`, `VUG`, `QUAL`, `MTUM`, `USMV`, `VLUE`

## Project Layout

```text
factorlens_etf_style_drift/
  data/
    raw/
    processed/
    tableau/
  notebooks/
  outputs/
    charts/
    tables/
    tableau/
  src/
  .gitignore
  CLAUDE.md
  README.md
  requirements.txt
```

`data/raw/`
Stores the original ETF price files and Fama French factor downloads.

`data/processed/`
Stores cleaned monthly returns, merged factor datasets, regression summaries, rolling betas, and prediction outputs.

`data/tableau/`
Stores the main CSV file used by Tableau.

`notebooks/`
Contains the analysis workflow in the order it should be run.

`outputs/tableau/`
Stores the Tableau workbook for the dashboard.

`src/`
Contains reusable Python functions used by the notebooks.

## Source Modules

`src/data_loader.py`
Downloads ETF prices, loads factor data, computes monthly returns, and prepares merged datasets.

`src/regression.py`
Runs full period and rolling ordinary least squares regressions using the Fama French factors.

`src/style_drift.py`
Calculates drift magnitude, style box labels, and drift scores.

`src/plotting.py`
Creates charts for rolling loadings, drift, heatmaps, and style timelines.

`src/export.py`
Builds clean CSV outputs, including the long format file used by Tableau.

## Notebook Workflow

1. `01_download_cleaned_data.ipynb`
   Downloads raw data, creates monthly ETF returns, and merges ETF returns with Fama French factors.

2. `02_factor_regression.ipynb`
   Runs full period factor regressions for each ETF.

3. `03_rolling_style_drift.ipynb`
   Runs rolling regressions and measures how ETF factor exposure changes over time.

4. `04_export_tableau.ipynb`
   Creates the Tableau ready dataset.

## Main Model

```text
ETF excess return =
alpha
+ beta_mkt * mkt_rf
+ beta_smb * smb
+ beta_hml * hml
+ beta_rmw * rmw
+ beta_cma * cma
+ residual
```

## Output Standards

Processed files should be easy to use in Tableau and other reporting tools.

1. Use long format, with one row per ETF per date.
2. Use lowercase column names with underscores.
3. Name the date column `date`.
4. Store dates as `YYYY-MM-DD`.
5. Avoid multi level column headers.

## Main Outputs

```text
data/processed/etf_returns_monthly.csv
data/processed/etf_ff5_merged.csv
data/processed/factor_regression_summary.csv
data/processed/rolling_factor_betas.csv
data/processed/etf_ff5_with_predictions.csv
data/tableau/tableau_factorlens_main.csv
outputs/tableau/factorlens_tableau_dashboard.twb
```

## Running The Project

Create an environment, install the requirements, then open the notebooks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order from the `notebooks/` folder. The final Tableau dataset is written to `data/tableau/tableau_factorlens_main.csv`.

## Repository Name

`factorlens-etf-style-drift`

## Short GitHub Description

ETF style drift analysis using Fama French factor regressions, rolling betas, residual diagnostics, and Tableau ready outputs.
