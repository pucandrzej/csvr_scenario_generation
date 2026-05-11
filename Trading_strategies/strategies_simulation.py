import os
import pickle
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from multiprocessing import Pool
import plotly.graph_objects as go

from config.trading_strategies_calibration_config import (
    bands_grid_config,
    median_grid_config,
)
from config.paths import (
    CALIBRATION_PICKLES_DIR,
    CALIBRATION_STRATEGIES_MEASURES_DIR,
    TEST_STRATEGIES_MEASURES_DIR,
    MODEL_RESULTS_DIR,
    PAPER_FIGURES_DIR,
)
from .strategies_utils import (
    # STRATEGY QUALITY MEASURES UTILS
    rtp,
    hhi,
    gini,
    topk_contribution,
    profit,
    mdd,
    avg_dd,
    downside_std,
    win_rate,
    # NOVEL PROBABILISTIC FORECAST MEASURES UTILS
    weighted_classification_accuracy,
    probabilistic_weighted_classification_accuracy,
)

from .trading_agents import (
    # BANDS
    one_sided_bands_strategy,
    two_sided_bands_strategy,
    OneSidedBandsConfig,
    TwoSidedBandsConfig,
    # MEDIAN
    one_sided_median_trading_strategy,
    two_sided_median_trading_strategy,
    OneSidedMedianConfig,
    TwoSidedMedianConfig,
    # NAIVE
    naive_1,
    naive_30,
)

# script parameters
DEV_PLOTS = False

STRATEGY_REGISTRY = {
    ("bands", True): {
        "func": one_sided_bands_strategy,
        "config": OneSidedBandsConfig,
    },
    ("bands", False): {
        "func": two_sided_bands_strategy,
        "config": TwoSidedBandsConfig,
    },
    ("median", True): {
        "func": one_sided_median_trading_strategy,
        "config": OneSidedMedianConfig,
    },
    ("median", False): {
        "func": two_sided_median_trading_strategy,
        "config": TwoSidedMedianConfig,
    },
}


def run_configurable_strategy(
    strategy_name,
    one_sided,
    y_actual,
    y_forecast,
    strategy_kwargs,
):
    strategy = STRATEGY_REGISTRY[(strategy_name, one_sided)]

    config = strategy["config"](**strategy_kwargs)

    return strategy["func"](
        y_actual,
        y_forecast,
        config,
    )


# script arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="median",
    help="Select the strategies model: bands, median, naive_1, naive_30, crystal_ball, opposite_crystal_ball in wca and pwca, where wca and pwca are measures and not strategies.",
)
parser.add_argument(
    "--underlying_model",
    default=None,
    help="Select the model for price forecasts: _____None____, _hist_insample_None_True_dual_coeff or _weather_scenarios_None_True_dual_coeff.",
)
parser.add_argument(
    "--underlying_model_column",
    default=None,
    help="Select the MULTI_prediction or benchmark_prediction.",
)
parser.add_argument(
    "--run_type",
    default="calibration",
    help="Select the run type: calibration or test.",
)
parser.add_argument(
    "--scp",
    default=np.nan,
    type=float,
    help="If test run is selected and model is bands we must set the scp for bands.",
)
parser.add_argument(
    "--distribution_param",
    default=np.nan,
    type=float,
    help="If test run is selected we must set the distribution p parameter.",
)
parser.add_argument(
    "--lambda_parameter",
    default=np.nan,
    type=float,
    help="If test run is selected we must set the lambda exponential path history impact damping parameter.",
)
parser.add_argument(
    "--trust_threshold",
    default=None,
    help="If test run is selected we must set the method of certainty threshold selection.",
)
parser.add_argument(
    "--weights_method",
    default=None,
    help="Method of calculating the weights: kernel or mae.",
)
parser.add_argument(
    "--direction",
    type=int,
    default=0,
    help="For one sided strategy we need to specify the direction.",
)
parser.add_argument(
    "--one_sided",
    default=False,
    action="store_true",
    help="Whether to run a speculative two sided strategy or one sided strategies.",
)
parser.add_argument(
    "--band_type",
    default="risk_seeking",
    help="Risk seeking or risk averse band trading. In risk seeking type we enter at max.",
)
parser.add_argument(
    "--calibration_pickle_name",
    default=None,
    help="Path to a pickle containing calibration results - if passed, the calibration run won't be recalculated, only the csv report of calibration will be generated based on a given pickle.",
)
parser.add_argument(
    "--test_subdir",
    default=None,
    help="Path to subdirectory; we need this split to safely calculate test results for kernel, MAE and static strategies separately.",
)
parser.add_argument("--processes", default=1, help="No of processes")
args = parser.parse_args()

# define the underlying ensemble forecasting configurations
if args.underlying_model is None:
    models = [
        "_hist_insample_None_True_dual_coeff",
        "_weather_scenarios_None_True_dual_coeff",
        "_____None____",
        "_hist_insample_None_False_None",
        "_weather_scenarios_None_False_None",
    ]
else:
    models = [args.underlying_model]

if args.underlying_model_column is None:
    columns_names = [
        "MULTI_prediction",
        "MULTI_prediction",
        "benchmark_prediction",
        "MULTI_prediction",
    ]
else:
    columns_names = [args.underlying_model_column]


def iterate_over_probab_results_and_prepare_measure(inp):
    """Use the predefined parameters and trading agent function and iterate over given period."""
    measure_func = inp[0]
    dir_name = inp[1]
    column_name = inp[2]
    scp = inp[3]
    p = inp[4]
    lambda_ = inp[5]
    trust_threshold_method = inp[6]
    weights_method = inp[7]

    measure_values_delivery = []

    for counter, daily_file in enumerate(
        [
            f_name
            for f_name in os.listdir(os.path.join(MODEL_RESULTS_DIR, dir_name))
            if f_name.startswith(f"{args.run_type}_")
        ]
    ):
        measure_values_day = []

        df = pd.read_csv(
            os.path.join(MODEL_RESULTS_DIR, dir_name, daily_file), index_col=0
        )
        try:
            actual = df["actual"].values
        except:
            breakpoint()
            pass
        if args.model in ["wca", "pwca"]:
            naive = df["naive"].values
        fore = df[
            [
                c
                for c in df.columns
                if c.startswith(column_name) and "base_path" not in c
            ]
        ].values

        if args.model in ["bands", "median"]:
            strategy_kwargs = {
                "p": p,
                "lambda_": lambda_,
                "trust_threshold_method": trust_threshold_method,
                "weights_method": weights_method,
                "dev_plots": DEV_PLOTS,
            }

            if args.model == "bands":
                strategy_kwargs.update(
                    {
                        "band_type": args.band_type,
                        "scp": scp,
                    }
                )

            if args.one_sided:
                strategy_kwargs["direction"] = args.direction

            measure_values_day.append(
                run_configurable_strategy(
                    args.model,
                    args.one_sided,
                    actual,
                    fore,
                    strategy_kwargs,
                )
            )
        elif args.model in ["naive_1", "naive_30"]:
            measure_values_day.append(
                measure_func(actual, args.direction, args.one_sided)
            )
        elif args.model == "wca":
            measure_values_day.append(measure_func(actual, fore, naive))
        elif args.model == "pwca":
            measure_values_day.append(measure_func(actual, fore, naive))

        measure_values_delivery.append(measure_values_day)

    return measure_values_delivery


if __name__ == "__main__":
    results = {}
    if args.run_type == "calibration":  # calibration on calibration window data
        if args.model == "bands":
            grid_config = bands_grid_config
        elif args.model == "median":
            grid_config = median_grid_config
        else:
            raise ValueError(f"No calibration implemented for model type {args.model}")
        p_list = grid_config["p_list"]
        lambda_list = grid_config["lambda_list"]
        trust_threshold_method = grid_config["trust_threshold_method"]
        parameter_method_1 = grid_config["parameter_method_1"]
        parameter_method_2 = grid_config["parameter_method_2"]
        scp_list = grid_config["scp"]
        grid = list(  # list of configs for KERNEL weighting
            itertools.product(
                scp_list,
                p_list,
                lambda_list,
                trust_threshold_method,
                parameter_method_1,
            )
        ) + list(  # list of configs for MAE weighting
            itertools.product(
                scp_list,
                [np.nan],
                [np.nan],
                trust_threshold_method,
                parameter_method_2,
            )
        )
    elif args.run_type == "test":  # validation run on test window data
        scp_list = [float(args.scp)]
        p_list = [float(args.distribution_param)]
        lambda_list = [float(args.lambda_parameter)]
        trust_threshold_method = [args.trust_threshold]
        parameter_method = [args.weights_method]
        grid = list(
            itertools.product(
                scp_list, p_list, lambda_list, trust_threshold_method, parameter_method
            )
        )

    if args.one_sided:
        if args.model == "bands":
            func = one_sided_bands_strategy
        elif args.model == "median":
            func = one_sided_median_trading_strategy
    else:
        if args.model == "bands":
            func = two_sided_bands_strategy
        elif args.model == "median":
            func = two_sided_median_trading_strategy

    if args.model == "naive_1":
        func = naive_1
    elif args.model == "naive_30":
        func = naive_30
    elif args.model == "wca":
        func = weighted_classification_accuracy
    elif args.model == "pwca":
        func = probabilistic_weighted_classification_accuracy

    calibration_pickle_name = f"results_gridsearch_{args.one_sided}_{args.direction}_{args.model}_{args.band_type}_{datetime.now().strftime('%Y-%m-%d %H;%M;%S')}.pkl"

    if args.calibration_pickle_name is None:
        for model, column_name in zip(models, columns_names):
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing {model}, {column_name}",
                flush=True,
            )

            delivery_directories = [
                d for d in os.listdir(MODEL_RESULTS_DIR) if d.endswith(model)
            ]

            for parameter_tuple in grid:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing {parameter_tuple}",
                    flush=True,
                )

                inputlist = [
                    [
                        func,
                        delivery_dir,
                        column_name,
                        parameter_tuple[0],
                        parameter_tuple[1],
                        parameter_tuple[2],
                        parameter_tuple[3],
                        parameter_tuple[4],
                    ]
                    for delivery_dir in delivery_directories
                ]
                with Pool(processes=int(args.processes)) as p:
                    parallel_results = p.map(
                        iterate_over_probab_results_and_prepare_measure, inputlist
                    )
                # parallel_results = []
                # for inputs in inputlist:
                #     parallel_results.append(iterate_over_probab_results_and_prepare_measure(inputs))

                results[parameter_tuple + (model, column_name)] = np.array(
                    parallel_results
                )

        # save results to pickle file
        if args.run_type == "calibration":
            with open(
                os.path.join(CALIBRATION_PICKLES_DIR, calibration_pickle_name),
                "wb",
            ) as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(
            os.path.join(CALIBRATION_PICKLES_DIR, args.calibration_pickle_name),
            "rb",
        ) as f:
            results = pickle.load(f)

    if args.model in ["wca", "pwca"]:
        stability_measures = {}

        for (k, arr), params in zip(results.items(), results.keys()):
            stability_measures[params] = []
            # sum over axis=0, take last index along axis=-1, then cumsum
            y = np.cumsum(np.sum(arr, axis=0)[:, -1, :], axis=0) / 1000

            measure_per_day = arr[:, :, 0, 0].T.reshape(-1)
            ref_measure_per_day = arr[:, :, 0, 1].T.reshape(-1)

            stability_measures[params].append(np.mean(measure_per_day))
            ref_params = tuple(
                ["_" for p in range(np.shape(grid)[1])]
                + list(params[np.shape(grid)[1] :])
            )
            stability_measures[ref_params] = []
            stability_measures[ref_params].append(np.mean(ref_measure_per_day))

        df = pd.DataFrame(
            [
                (a, b, c, d, e, f, *vals)
                for (a, b, c, d, e, f), vals in stability_measures.items()
            ],
            columns=[
                "param1",
                "param2",
                "threshold",
                "weights",
                "model_setting",
                "model",
                "measure_value",
            ],
        )

        print(df.to_string())

    else:
        fig = go.Figure()

        stability_measures = {}
        stability_measures_reference = {}

        for (k, arr), params in zip(results.items(), results.keys()):
            stability_measures[params] = []
            # sum over axis=0, take last index along axis=-1, then cumsum
            y = np.cumsum(np.sum(arr, axis=0)[:, -1, :], axis=0) / 1000

            pnl_per_delivery_and_day = arr[:, :, 0, 0].T.reshape(-1)
            action_per_delivery_and_day = arr[:, :, 0, 2].T.reshape(-1)
            ref_pnl_per_delivery_and_day = arr[:, :, 0, 1].T.reshape(-1)
            best_pnl_per_delivery_and_day = arr[:, :, 0, 3].T.reshape(-1)
            worst_pnl_per_delivery_and_day = arr[:, :, 0, 4].T.reshape(-1)

            stability_measures[params].append(mdd(pnl_per_delivery_and_day))
            stability_measures[params].append(avg_dd(pnl_per_delivery_and_day))
            stability_measures[params].append(
                downside_std(pnl_per_delivery_and_day, one_sided=args.one_sided)
            )
            stability_measures[params].append(np.std(pnl_per_delivery_and_day))
            stability_measures[params].append(win_rate(pnl_per_delivery_and_day))
            stability_measures[params].append(profit(pnl_per_delivery_and_day))
            stability_measures[params].append(hhi(pnl_per_delivery_and_day))
            stability_measures[params].append(gini(pnl_per_delivery_and_day))
            stability_measures[params].append(
                topk_contribution(pnl_per_delivery_and_day)
            )
            stability_measures[params].append(
                rtp(
                    pnl_per_delivery_and_day,
                    best_pnl_per_delivery_and_day,
                    worst_pnl_per_delivery_and_day,
                    one_sided=args.one_sided,
                )
            )
            stability_measures[params].append(
                1
                - np.sum(action_per_delivery_and_day) / len(action_per_delivery_and_day)
            )
            stability_measures[params].append(np.mean(pnl_per_delivery_and_day))
            stability_measures[params].append(profit(best_pnl_per_delivery_and_day))
            stability_measures[params].append(profit(worst_pnl_per_delivery_and_day))
            stability_measures[params].append(
                downside_std(best_pnl_per_delivery_and_day, one_sided=args.one_sided)
            )
            stability_measures[params].append(
                downside_std(worst_pnl_per_delivery_and_day, one_sided=args.one_sided)
            )

            x = np.arange(len(y))  # forecast days

            if args.run_type == "test":
                fig.add_trace(
                    go.Scatter(x=x, y=y[:, 0], mode="lines", name=f"strategy {k}")
                )

            if args.model == "bands":  # for bands we want to save every basic SCP level
                ref_params = tuple(
                    [params[0]]
                    + ["_" for p in range(1, np.shape(grid)[1])]
                    + list(params[np.shape(grid)[1] :])
                )
            else:
                ref_params = tuple(
                    ["_" for p in range(np.shape(grid)[1])]
                    + list(params[np.shape(grid)[1] :])
                )

            if ref_params not in stability_measures_reference:
                stability_measures_reference[ref_params] = []

                stability_measures_reference[ref_params].append(
                    mdd(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    avg_dd(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    downside_std(ref_pnl_per_delivery_and_day, one_sided=args.one_sided)
                )
                stability_measures_reference[ref_params].append(
                    np.std(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    win_rate(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    profit(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    hhi(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    gini(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    topk_contribution(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    rtp(
                        ref_pnl_per_delivery_and_day,
                        best_pnl_per_delivery_and_day,
                        worst_pnl_per_delivery_and_day,
                        one_sided=args.one_sided,
                    )
                )
                stability_measures_reference[ref_params].append(0)
                stability_measures_reference[ref_params].append(
                    np.mean(ref_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    profit(best_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    profit(worst_pnl_per_delivery_and_day)
                )
                stability_measures_reference[ref_params].append(
                    downside_std(
                        best_pnl_per_delivery_and_day, one_sided=args.one_sided
                    )
                )
                stability_measures_reference[ref_params].append(
                    downside_std(
                        worst_pnl_per_delivery_and_day, one_sided=args.one_sided
                    )
                )

                if args.run_type == "test":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y[:, 1],
                            mode="lines",
                            name=f"baseline strategy {ref_params}",
                        )
                    )

        if args.run_type == "test":
            # Labels and style
            fig.update_layout(
                title="ENSEMBLE TRADING PnL",
                xaxis_title="days",
                yaxis_title="PnL [1000 EUR/MWh]",
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
                ),
                font=dict(size=10),
            )

            fig.write_html(
                os.path.join(
                    PAPER_FIGURES_DIR,
                    f"strategies_{args.run_type}_{args.one_sided}_{args.direction}_{args.model}.html",
                )
            )
            if DEV_PLOTS:
                fig.show()

        # save the table with the measures
        for k in stability_measures_reference.keys():
            stability_measures[k] = stability_measures_reference[k]
        df = pd.DataFrame(
            [
                (a, b, c, d, e, f, g, *vals)
                for (a, b, c, d, e, f, g), vals in stability_measures.items()
            ],
            columns=[
                "param1",
                "param2",
                "param3",
                "threshold",
                "weights",
                "model_setting",
                "model",
                "MDD",
                "avg D",
                "std_minus",
                "std",
                "win_rate",
                "profit",
                "hhi",
                "gini",
                "topk",
                "rtp",
                "no_action_perc",
                "profit_per_action",
                "crystal_profit",
                "noncrystal_profit",
                "crystal_std_minus",
                "noncrystal_std_minus",
            ],
        )

        if (
            float(args.direction) == 0 or float(args.direction) == -1
        ):  # RATIO if we maximize the profit and minimize risk
            df["Sortino_ratio"] = df["profit"] / df["std_minus"]
            df = df.sort_values("Sortino_ratio", ascending=False)
        elif (
            float(args.direction) == 1
        ):  # PRODUCT if we minimize the profit and minimize risk
            df["Sortino_product"] = df["profit"] * df["std_minus"]
            df = df.sort_values("Sortino_product", ascending=True)

        if args.run_type == "calibration":
            df.to_csv(
                os.path.join(
                    CALIBRATION_STRATEGIES_MEASURES_DIR,
                    f"calibration_trading_strategy_measures_{args.one_sided}_{args.direction}_{args.model}_{args.band_type}.csv",
                )
            )
        else:
            df.to_csv(
                os.path.join(
                    TEST_STRATEGIES_MEASURES_DIR,
                    args.test_subdir,
                    f"test_trading_strategy_measures_{args.underlying_model}_{args.underlying_model_column}_{args.one_sided}_{args.direction}_{args.model}_{args.band_type}.csv",
                )
            )
            print(df.to_string())
