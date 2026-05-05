import pandas as pd
import statsmodels.api as sm


def run_factor_regression(df: pd.DataFrame, factor_cols=None, y_col="excess_return"):
    if factor_cols is None:
        factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma"]

    summary_rows = []
    prediction_frames = []

    for ticker, group in df.groupby("ticker", sort=False):
        data = group.sort_values("date").dropna().copy()

        y = data[y_col]
        X = sm.add_constant(data[factor_cols])

        model = sm.OLS(y, X).fit()

        data["predicted_excess_return"] = model.predict(X)
        data["predicted_return"] = data["predicted_excess_return"] + data["rf"]
        data["residual"] = data[y_col] - data["predicted_excess_return"]

        summary_rows.append(
            {
                "ticker": ticker,
                "alpha": model.params["const"],
                "alpha_tstat": model.tvalues["const"],
                "alpha_pvalue": model.pvalues["const"],
                "beta_market": model.params["mkt_rf"],
                "beta_market_tstat": model.tvalues["mkt_rf"],
                "beta_market_pvalue": model.pvalues["mkt_rf"],
                "beta_smb": model.params["smb"],
                "beta_smb_tstat": model.tvalues["smb"],
                "beta_smb_pvalue": model.pvalues["smb"],
                "beta_hml": model.params["hml"],
                "beta_hml_tstat": model.tvalues["hml"],
                "beta_hml_pvalue": model.pvalues["hml"],
                "beta_rmw": model.params["rmw"],
                "beta_rmw_tstat": model.tvalues["rmw"],
                "beta_rmw_pvalue": model.pvalues["rmw"],
                "beta_cma": model.params["cma"],
                "beta_cma_tstat": model.tvalues["cma"],
                "beta_cma_pvalue": model.pvalues["cma"],
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "n_obs": int(model.nobs),
            }
        )

        prediction_frames.append(data)

    regression_summary = pd.DataFrame(summary_rows)
    df_predictions = pd.concat(prediction_frames, ignore_index=True)

    return regression_summary, df_predictions
