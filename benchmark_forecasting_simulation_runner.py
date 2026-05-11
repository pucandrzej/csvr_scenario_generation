"""Runner of the naive benchmark path forecasting simulation for all the deliveries"""

import sys
import time
import subprocess

from utils.forecasting_simulation_config import (
    limited_scenarios_number,
    deliveries_no,
    calib_window_days_no,
    last_trade_time_in_path_delta,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start_delivery", default=0, help="Start of the simulated deliveries"
)
parser.add_argument(
    "--end_delivery", default=deliveries_no, help="End of the simulated deliveries"
)
parser.add_argument(
    "--calibration_window_len",
    default=calib_window_days_no,
    help="For every date consider a historical results from a calibration window.",
)
parser.add_argument(
    "--processes", default=32, help="No of parallel processes in underlying simulation."
)

processes = 32

args = parser.parse_args()

start = args.start_delivery
sys.stderr = open(
    f"BENCHMARK_SIMU_ERR_{start}_{args.end_delivery}_benchmark_{args.calibration_window_len}.txt",
    "w",
)
sys.stdout = open(
    f"BENCHMARK_SIMU_LOG_{start}_{args.end_delivery}_benchmark_{args.calibration_window_len}.txt",
    "w",
)

joblist = []
for delivery_time in range(int(start), int(args.end_delivery)):
    trade_time = delivery_time * 3 + last_trade_time_in_path_delta

    for required_scenarios in limited_scenarios_number:
        joblist.append(
            [
                sys.executable,
                "-m",
                "naive_path_forecasting.py",
                "--trade_time",
                str(trade_time),
                "--delivery_time",
                str(delivery_time),
                "--calibration_window_len",
                str(args.calibration_window_len),
                "--processes",
                str(processes),
            ]
            + ["--required_scenarios", str(required_scenarios)]
            * (required_scenarios is not None)
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
    print(f"running job {invoked + 1} of {len(joblist)}: {joblist[invoked]}")
    stack.append(subprocess.Popen(line, stderr=sys.stderr, stdout=sys.stdout))
    stack[-1].wait()  # wait for the process to finish
    invoked += 1
