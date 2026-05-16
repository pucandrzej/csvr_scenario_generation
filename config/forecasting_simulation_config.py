forecasting_horizon = 31  # number of 5min intervals in the forecasting path
deliveries_no = 96  # number of 15min deliveries during one day
trading_start_hour = 16
first_trading_start_of_simulation = "2019-01-01 16:00:00"
needed_columns_of_continuous_preprocessed_data = (
    [str(i) for i in range(192)] + ["288"]
)  # first 96 columns are prices for each delivery, second 96 are respective volumes, column 288 contains weekday no. indicators
total_no_of_cont_market_columns = 193

limited_scenarios_number = [  # list of scenarios number cap to test the influence of hand picked history impact on forecasting
    28,
    182,
    None,
]
last_trade_time_in_path_delta = (
    8 * 12 - 6
)  # 8*12 is 8 hours each containing 12 5min periods, -6 as we are trading up to 30min before the delivery
calib_window_days_no = 182
