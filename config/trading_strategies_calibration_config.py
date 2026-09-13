import numpy as np

bands_grid_config = {
    "scp": np.arange(0.05, 1.00, 0.05),
    "p_list": [0.25, 0.5, 1.0, 2.5, 3.5, 3.75, 4.0],
    "lambda_list": [0.2, 0.35, 0.45, 0.5, 1.0, 1.25, 2.0, 3.0],
    "trust_threshold_method": [
        "3sigma",
        "5_95_IPR",  # IPR: InterPercentile Range
        "iqr",
        "mae",
    ],
    "parameter_method_1": ["kernel"],
    "parameter_method_2": ["mae"],
}

# Frozen (p, lambda) pairs from reweighting calibration, matching
# PAPER_TABLES/reweighting_best_parameters.csv on 2026-09-07.
# Trading calibration never selects these parameters from result files at runtime.
reweighting_best_params = {
    ("_____None____", "benchmark_prediction"): {
        "mae": (0.5, 1.5), "crps": (0.5, 0.9),
    },
    ("_hist_insample_None_False_None", "MULTI_prediction"): {
        "mae": (0.5, 2.0), "crps": (0.5, 1.5),
    },
    ("_weather_scenarios_None_False_None", "MULTI_prediction"): {
        "mae": (2.5, 3.0), "crps": (1.0, 2.0),
    },
    ("_hist_insample_None_True_dual_coeff", "MULTI_prediction"): {
        "mae": (0.5, 1.25), "crps": (0.25, 2.0),
    },
    ("_weather_scenarios_None_True_dual_coeff", "MULTI_prediction"): {
        "mae": (2.75, 3.0), "crps": (1.0, 1.5),
    },
}


def trading_calibration_grid(strategy, model_setting, model):
    """Use frozen model-specific weights and calibrate thresholds/SCP only."""
    config = {"median": median_grid_config, "bands": bands_grid_config}[strategy]
    metric = "mae" if strategy == "median" else "crps"
    p, lambda_ = reweighting_best_params[(model_setting, model)][metric]
    return [
        (scp, power, decay, threshold, method)
        for method, power, decay in [("kernel", p, lambda_), ("mae", np.nan, np.nan)]
        for scp in config["scp"]
        for threshold in config["trust_threshold_method"]
    ]

median_grid_config = {
    "scp": [np.nan],
    "p_list": [
        0.1,
        0.25,
        0.5,
        0.75,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.25,
        3.5,
        3.75,
        4,
        4.25,
        4.5,
        4.75,
        5
    ],
    "lambda_list": [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,  # one every 0.05 up to 0.7
        0.8,
        0.9,
        1.0,  # then one every 0.1 up to 1
        1.25,
        1.5,  # two 0.25 apart
        2,
        3,
        4,
        5,
        6,  # five 1 apart
    ],
    "trust_threshold_method": [
        "3sigma",
        "iqr",
        "5_95_IPR",  # IPR: InterPercentile Range
        "mae",
    ],
    "parameter_method_1": ["kernel"],
    "parameter_method_2": ["mae"],
}
