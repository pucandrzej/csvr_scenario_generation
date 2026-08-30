"""Print median-strategy optima and the implied bands calibration grid."""

from pathlib import Path

import pandas as pd


directory = (
    Path(__file__).resolve().parents[1]
    / "TRADING_STRATEGIES_RESULTS"
    / "CALIBRATION_MEASURES"
)
best = []
for path in sorted(
    directory.glob("calibration_trading_strategy_measures_*_median_*.csv")
):
    data = pd.read_csv(path).query("weights in ['kernel', 'mae']")
    rows = data.loc[
        data.groupby(["model_setting", "model", "weights"])["Sortino_ratio"].idxmax()
    ]
    best.append(rows.assign(one_sided="_True_" in path.name))

best = pd.concat(best).sort_values(["one_sided", "weights", "model_setting"])
print(
    best[
        [
            "one_sided",
            "model_setting",
            "model",
            "weights",
            "param2",
            "param3",
            "threshold",
            "Sortino_ratio",
        ]
    ].to_string(index=False)
)

kernel = best.query("weights == 'kernel'")
print("\nSuggested bands grid:")
print("p_list =", sorted(kernel["param2"].astype(float).unique()))
print("lambda_list =", sorted(kernel["param3"].astype(float).unique()))
print("trust_threshold_method =", sorted(best["threshold"].unique()))
