import os
from pathlib import Path

# FORECASTING STUDY PATHS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(ROOT, "Data")
MARKET_DATA_DIR = os.path.join(
    ROOT, "Data", "preprocessed_continuous_intraday_prices_and_volume.db"
)
CONCATENATED_RAW_MARKET_DATA = os.path.join(
    DATA_DIR, "Transactions", "concatenated_table.csv"
)
INTERMEDIATE_MARKET_DATA = os.path.join(
    DATA_DIR, "Transactions", "quarterhourly_price_analysis_table_5min.csv"
)
INITIALLY_PREPROCESSED_MARKET_DATA = os.path.join(
    DATA_DIR, "quarterhourly_preprocessed_dataset_5min.csv"
)
LOGS_DIR = os.path.join(ROOT, "LOGS")
os.makedirs(
    LOGS_DIR, exist_ok=True
)  # create here to avoid calling it multiple times in other scripts

TIMING_RESULTS_DIR = os.path.join(ROOT, "TIMING_RESULTS")
os.makedirs(
    TIMING_RESULTS_DIR, exist_ok=True
)  # create here to avoid calling it multiple times in other scripts

BENCHMARK_RESULTS_DIR = os.path.join(ROOT, "BENCHMARK_FORECASTING_SIMULATION_RESULTS")
MODEL_RESULTS_DIR = os.path.join(ROOT, "FORECASTING_SIMULATION_RESULTS")
RAW_MODEL_RESULTS_DIR = "FORECASTING_SIMULATION_RESULTS"

MAE_CRPS_RESULTS_DIR = os.path.join(ROOT, "MAE_CRPS_RESULTS")
os.makedirs(MAE_CRPS_RESULTS_DIR, exist_ok=True)

PAPER_FIGURES_DIR = os.path.join(ROOT, "PAPER_FIGURES")
PAPER_TABLES_DIR = os.path.join(ROOT, "PAPER_TABLES")

# TRADING STRATEGIES PATHS
GENERAL_STRATEGY_RESULTS = os.path.join(ROOT, "TRADING_STRATEGIES_RESULTS")
CALIBRATION_PICKLES_DIR = os.path.join(
    ROOT, "TRADING_STRATEGIES_RESULTS", "CALIBRATION_PICKLES"
)
os.makedirs(CALIBRATION_PICKLES_DIR, exist_ok=True)
CALIBRATION_STRATEGIES_MEASURES_DIR = os.path.join(
    ROOT, "TRADING_STRATEGIES_RESULTS", "CALIBRATION_MEASURES"
)
os.makedirs(CALIBRATION_STRATEGIES_MEASURES_DIR, exist_ok=True)
TEST_STRATEGIES_MEASURES_DIR = os.path.join(
    ROOT, "TRADING_STRATEGIES_RESULTS", "TEST_MEASURES"
)
os.makedirs(os.path.join(TEST_STRATEGIES_MEASURES_DIR, "kernel"), exist_ok=True)
os.makedirs(os.path.join(TEST_STRATEGIES_MEASURES_DIR, "mae"), exist_ok=True)
os.makedirs(os.path.join(TEST_STRATEGIES_MEASURES_DIR, "static"), exist_ok=True)
