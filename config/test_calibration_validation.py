import pandas as pd
from datetime import datetime

required_start = datetime(2019, 1, 2)
required_start_pd_timestamp = pd.Timestamp(year=2019, month=1, day=2)
required_end = datetime(2021, 1, 1)

validation_window_start = "2020-01-01"
validation_window_end = "2020-12-31"
validation_window_length = 366

currency_change_date_PL = datetime(2019, 11, 20)

summer_time_2020_dst = datetime(day=29, month=3, year=2020, hour=3)
winter_time_2020_dst = datetime(day=25, month=10, year=2020, hour=3)

data_reporting_standardization_start = datetime(day=24, month=10, year=2020, hour=14)
data_reporting_standardization_end = datetime(day=24, month=10, year=2020, hour=15)
