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
        for direction, one_sided in zip([-1, 0], [True, False]):
            joblist.append(
                [
                    sys.executable,
                    "-m",
                    "Trading_strategies.strategies_simulation",
                ]
                + ["--direction", str(direction)] * (direction is not None)
                + ["--one_sided"] * one_sided
                + ["--model", model]
                + ["--processes", args.processes]
                + ["--band_type", band_type]
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
    stack[-1].wait()  # wait for the process to finish
    invoked += 1
