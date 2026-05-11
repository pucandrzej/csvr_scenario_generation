"""Script runs each configuration of trading strategies on best parameters from calibration run."""

import os
import sys
import pandas as pd
import subprocess
import numpy as np

from config.paths import LOGS_DIR, CALIBRATION_STRATEGIES_MEASURES_DIR

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--processes",
    default="32",
    help="No of parallel processes in underlying simulation.",
)
args = parser.parse_args()

# redirect all stdout/stderr to file
sys.stdout = open(os.path.join(LOGS_DIR, "STRATEGIES_VALIDATION_RUNNER.txt"), "w")
sys.stderr = sys.stdout


def parse_file_flags(filename):
    flags = {}

    if "False_0" in filename:
        flags["one_sided"] = False
        flags["direction"] = None

    elif "True_-1" in filename:
        flags["one_sided"] = True
        flags["direction"] = -1

    if "_median" in filename:
        flags["model"] = "median"

    elif "_bands_" in filename:
        flags["model"] = "bands"

    if "risk_seeking" in filename:
        flags["band_type"] = "risk_seeking"

    elif "risk_averse" in filename:
        flags["band_type"] = "risk_averse"

    else:
        flags["band_type"] = None

    return flags


for file in os.listdir(CALIBRATION_STRATEGIES_MEASURES_DIR):
    print("\n" + "=" * 80)
    print(f"RUNNING BEST MODELS BASED ON CALIBRATION RESULT FROM {file}")
    print("=" * 80)

    df = pd.read_csv(os.path.join(CALIBRATION_STRATEGIES_MEASURES_DIR, file))

    for weighting_type in ["_", "kernel", "mae"]:  # "_" is the static strategy
        weighting_type_df = df[df["weights"] == weighting_type]

        if (
            weighting_type == "_"
        ):  # to run the simulation for best static strategy parameters
            weighting_type_df[["param2", "param3"]] = weighting_type_df[
                ["param2", "param3"]
            ].replace("_", np.nan)
            weighting_type_df["threshold"] = "mae"
            weighting_type_df["weights"] = "mae"

        idx = weighting_type_df.groupby("model_setting")["Sortino_ratio"].idxmax()
        best_rows = weighting_type_df.loc[idx]

        flags = parse_file_flags(file)

        for _, row in best_rows.iterrows():
            cmd = [
                sys.executable,
                "-m",
                "Trading_strategies.strategies_simulation",
                "--model",
                flags["model"],
                "--run_type",
                "test",
                "--weights_method",
                str(row["weights"]),
                "--distribution_param",
                str(row["param2"]),
                "--lambda_parameter",
                str(row["param3"]),
                "--trust_threshold",
                str(row["threshold"]),
                "--underlying_model",
                str(row["model_setting"]),
                "--underlying_model_column",
                str(row["model"]),
                "--processes",
                str(args.processes),
                "--test_subdir",
                weighting_type.replace("_", "static"),
            ]

            if flags["one_sided"]:
                cmd.append("--one_sided")
                cmd.extend(["--direction", str(flags["direction"])])

            if flags["model"] == "bands" and flags["band_type"]:
                cmd.extend(["--band_type", flags["band_type"]])
                cmd.extend(["--scp", str(row["param1"])])
            print("\nCOMMAND:")
            print(" ".join(cmd))
            print("-" * 80)

            subprocess.run(cmd, check=True)


print("\nALL STRATEGIES VALIDATION RUNS FINISHED")
