import numpy as np

bands_grid_config = {
    'scp': np.arange(0.05, 1.00, 0.05),
    'p_list': [0.25, 1.25, 1.75, 2.0, 3.0, 3.75],
    'lambda_list': [0.0, 0.05, 0.35, 0.7, 0.9, 2.0, 5.0],
    'trust_threshold_method': [
        '3sigma',
        '5_95_IPR', # IPR: InterPercentile Range
        'iqr'
    ],
    'parameter_method_1': ["kernel"],
    'parameter_method_2': ["mae"]
}

median_grid_config = {
    'scp': [np.nan],
    'p_list': [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4],
    'lambda_list': [
        0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,  # one every 0.05 up to 0.7
        0.8, 0.9, 1.0,  # then one every 0.1 up to 1
        1.25, 1.5,  # two 0.25 apart
        2, 3, 4, 5, 6  # five 1 apart
        ],
    'trust_threshold_method': [
        "3sigma",
        "iqr",
        "5_95_IPR", # IPR: InterPercentile Range
        "mae"
    ],
    'parameter_method_1': ["kernel"],
    'parameter_method_2': ["mae"]
}
