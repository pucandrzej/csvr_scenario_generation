quantile_diff_tolerance = 1e-2
min_scenarios = 10  # minimum no. of scenarios required when iterating over scenarios with Wasserstein stopping crit
wasserstein_moving_avg_window = 10  # note that this automatically gives default 12 scenarios minimum as we start from Wasserstein = inf so first 10 deltas contain inf
weather_scenarios_split_direction = True
