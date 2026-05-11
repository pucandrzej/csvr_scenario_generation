"""Runner of the cSVR path forecasting simulation for all the deliveries and configurations"""

import os
import time
import subprocess

import sys

import argparse

from config.paths import LOGS_DIR
from config.forecasting_simulation_config import (
    limited_scenarios_number,
    last_trade_time_in_path_delta,
    deliveries_no,
    calib_window_days_no,
)

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
    "--special_results_directory",
    default=None,
    help="Running on WCSS Wroclaw University of Science and Technology supercomputers requires us to save the results in dedicated path.",
)
parser.add_argument(
    "--processes", default=32, help="No of parallel processes in underlying simulation."
)

args = parser.parse_args()
start = args.start_delivery
end = args.end_delivery
processes = int(args.processes)

PROBAB_APPROACHES = ["hist_insample", "weather_scenarios"]

WASSERSTEIN_STOPPING_CRIT = [
    True,
    False,
]

SCENARIOS_STOPPING_METHODS = [
    "dual_coeff",
    None,
]

repeated_required_scenarios_list = [
    val
    for val in limited_scenarios_number
    for _ in range(len(WASSERSTEIN_STOPPING_CRIT))
]

print(  # show all of the configs to run
    WASSERSTEIN_STOPPING_CRIT * len(limited_scenarios_number),
    SCENARIOS_STOPPING_METHODS * len(limited_scenarios_number),
    repeated_required_scenarios_list,
)

# declare the location of logs
sys.stderr = open(
    os.path.join(
        LOGS_DIR,
        f"SIMU_ERR_{start}_{end}_{args.calibration_window_len}.txt",
    ),
    "w",
)
sys.stdout = open(
    os.path.join(
        LOGS_DIR,
        f"SIMU_LOG_{start}_{end}_{args.calibration_window_len}.txt",
    ),
    "w",
)

joblist = []  # list to gather all of the jobs to run later sequentially
for probab_approach in PROBAB_APPROACHES:
    for delivery_time in range(int(start), int(end)):
        trade_time = delivery_time * 3 + last_trade_time_in_path_delta

        for (
            wasserstein_stopping_crit,
            scenarios_sampling_method,
            required_scenarios,
        ) in zip(
            WASSERSTEIN_STOPPING_CRIT * len(limited_scenarios_number),
            SCENARIOS_STOPPING_METHODS * len(limited_scenarios_number),
            repeated_required_scenarios_list,
        ):
            joblist.append(
                [
                    sys.executable,
                    "-m",
                    "Forecasting.path_forecasting",
                    "--trade_time",
                    str(trade_time),
                    "--delivery_time",
                    str(delivery_time),
                    "--calibration_window_len",
                    str(args.calibration_window_len),
                    "--processes",
                    str(processes),
                    "--probab_approach",
                    probab_approach,
                ]
                + ["--scenarios_sampling_method", scenarios_sampling_method]
                * (scenarios_sampling_method is not None)
                + ["--required_scenarios", str(required_scenarios)]
                * (required_scenarios is not None)
                + ["--wasserstein_stopping_crit"] * wasserstein_stopping_crit
                + [
                    "--special_results_directory",
                    str(args.special_results_directory),
                ]
                * (args.special_results_directory is not None)
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
