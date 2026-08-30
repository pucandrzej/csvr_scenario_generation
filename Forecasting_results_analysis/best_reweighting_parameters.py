"""Export paper-ready MAE- and CRPS-optimal reweighting parameters."""

import argparse
from pathlib import Path

import pandas as pd

from config.paths import GENERAL_STRATEGY_RESULTS, PAPER_TABLES_DIR


MODEL_LABELS = {
    "_____None____": "naive",
    "_hist_insample_None_False_None": "historical",
    "_weather_scenarios_None_False_None": "fundamental",
    "_hist_insample_None_True_dual_coeff": "SVS historical",
    "_weather_scenarios_None_True_dual_coeff": "SVS fundamental",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=Path(GENERAL_STRATEGY_RESULTS)
        / "REWEIGHTING_MAE_CRPS"
        / "calibration_dynamic_reweighting_forecast.csv",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path(PAPER_TABLES_DIR) / "reweighting_best_parameters.csv",
        type=Path,
    )
    return parser.parse_args()


def best_parameters(calibration):
    data = calibration.query("run_type == 'calibration' and weights == 'kernel'")
    missing = set(MODEL_LABELS) - set(data["model_setting"])
    if missing:
        raise ValueError(f"Missing calibration models: {sorted(missing)}")

    rows = []
    for setting, label in MODEL_LABELS.items():
        model = data[data["model_setting"] == setting]
        mae = model.loc[model["mae_weighted"].idxmin()]
        crps = model.loc[model["crps_weighted"].idxmin()]
        rows.append(
            {
                "scenario_selection": label,
                "model_setting": setting,
                "model": mae["model"],
                "mae_selected_p": mae["param2"],
                "mae_selected_lambda": mae["param3"],
                "mae_calibration": mae["mae_weighted"],
                "mae_improvement_pct": mae["mae_improvement_pct"],
                "crps_selected_p": crps["param2"],
                "crps_selected_lambda": crps["param3"],
                "crps_calibration": crps["crps_weighted"],
                "crps_improvement_pct": crps["crps_improvement_pct"],
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result = best_parameters(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
