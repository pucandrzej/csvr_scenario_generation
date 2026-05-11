import numpy as np

bands_grid_config = {
    "scp": [0.1, 0.5],
    "p_list": [0.5],
    "lambda_list": [0.05],
    "trust_threshold_method": ["iqr"],
    "parameter_method_1": ["kernel"],
    "parameter_method_2": ["mae"],
}

median_grid_config = {
    "scp": [np.nan],
    "p_list": [0.1, 0.2],
    "lambda_list": [0],
    "trust_threshold_method": ["3sigma"],
    "parameter_method_1": ["kernel"],
    "parameter_method_2": ["mae"],
}
