"""
This script runs the simulation for all trading strategies configurations
"""

import os
import sys
import time
import subprocess

from config.paths import LOGS_DIR

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--processes",
    default="32",
    help="No of parallel processes in underlying simulation.",
)
parser.add_argument("--special_results_directory")
args = parser.parse_args()

sys.stderr = open(
    os.path.join(LOGS_DIR, "TRADING_CALIBRATION_ERR.txt"),
    "w",
)
sys.stdout = open(
    os.path.join(LOGS_DIR, "TRADING_CALIBRATION_OUT.txt"),
    "w",
)

joblist = []
for model in ["median", "bands"]:
    if model == "bands":
        bands_types = ["risk_seeking", "risk_averse"]
    else:
        bands_types = ["risk_seeking"]

    for band_type in bands_types:
        for one_sided in [True, False]:
            joblist.append(
                [
                    sys.executable,
                    "-m",
                    "Trading_strategies.strategies_simulation",
                ]
                + ["--one_sided"] * one_sided
                + ["--model", model]
                + ["--processes", args.processes]
                + ["--band_type", band_type]
                + (
                    ["--special_results_directory", args.special_results_directory]
                    if args.special_results_directory
                    else []
                )
            )

invoked = 0
stack = []
ts = time.time()
concurrent = 1
while invoked < len(joblist):
    while len(stack) == concurrent:
        for no, p in enumerate(stack):
            if p.poll() is not None:
                stack.pop(no)
                break
        time.sleep(1)
    line = joblist[invoked]
    print(
        f"running job {invoked + 1} of {len(joblist)}: {joblist[invoked]}", flush=True
    )
    stack.append(subprocess.Popen(line, stderr=sys.stderr, stdout=sys.stdout))
    return_code = stack[-1].wait()  # wait for the process to finish
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, line)
    invoked += 1
