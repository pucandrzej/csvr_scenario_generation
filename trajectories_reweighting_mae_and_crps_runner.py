"""Run MAE/CRPS dynamic-reweighting diagnostics for all calibrated configurations."""

import argparse
import os
import subprocess
import sys

import pandas as pd

from config.paths import CALIBRATION_STRATEGIES_MEASURES_DIR, LOGS_DIR


parser = argparse.ArgumentParser()
parser.add_argument("--processes", default="32")
args = parser.parse_args()

sys.stdout = open(os.path.join(LOGS_DIR, "MAE_CRPS_DIAGNOSTICS_RUNNER.txt"), "w")
sys.stderr = sys.stdout


def parse_file_flags(filename):
    """Extract strategy configuration encoded in a calibration filename."""
    return {
        "one_sided": "_True_" in filename,
        "model": "bands" if "_bands_" in filename else "median",
        "band_type": (
            "risk_seeking" if "risk_seeking" in filename
            else "risk_averse" if "risk_averse" in filename
            else None
        ),
    }


for file in sorted(os.listdir(CALIBRATION_STRATEGIES_MEASURES_DIR)):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(CALIBRATION_STRATEGIES_MEASURES_DIR, file)
    df = pd.read_csv(path)
    flags = parse_file_flags(file)

    print("\n" + "=" * 80)
    print(f"RUNNING DIAGNOSTICS FROM {file}")
    print("=" * 80, flush=True)

    # One diagnostic run handles raw + MAE + kernel simultaneously.
    # Run once for every calibrated forecast model/configuration.
    configs = df[["model_setting", "model"]].drop_duplicates()

    for _, row in configs.iterrows():
        cmd = [
            sys.executable,
            "-m",
            "Forecasting_results_analysis.dynamic_reweighting_forecast_diagnostics",
            "--model_setting",
            str(row["model_setting"]),
            "--underlying_model_column",
            str(row["model"]),
            "--strategy_model",
            flags["model"],
            "--run_type",
            "test",
            "--calibration_file",
            file,
            "--processes",
            str(args.processes),
        ]

        if flags["one_sided"]:
            cmd.append("--one_sided")

        if flags["model"] == "bands":
            cmd.extend(["--band_type", flags["band_type"]])

        print("\nCOMMAND:")
        print(" ".join(cmd), flush=True)
        print("-" * 80, flush=True)

        subprocess.run(cmd, check=True)


print("\nALL MAE/CRPS DIAGNOSTIC RUNS FINISHED", flush=True)