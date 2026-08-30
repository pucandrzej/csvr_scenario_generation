# Replication package for "Scenario generation of intraday electricity price paths for optimal trading in continuous markets".
## Authors
Andrzej Puć

Wrocław University of Science and Technology, Faculty of Pure and Applied Mathematics, Hugo Steinhaus Center, Wyb. Wyspiańskiego 27, Wrocław, 50-370, Poland

### Contact information
andrzej.puc@pwr.edu.pl

## Date of replication package creation
2026.05.11

## Last modification date
2026.08.30

## Overview & contents
The code in this replication material allows for recalculating the forecasting and trading strategy simulation which served as an illustration of forecasting methodology proposed in the paper "Scenario generation of intraday electricity price paths for optimal trading in continuous markets". 

When the simulation is recalculated, each figure presented in the paper can be generated using 

1. `forecasting_study_tables_and_figures.ipynb`, which contains the main forecasting task results and methodology illustrations: the MAE and CRPS tables, CRPS plot, scenarios number table, an illustration of fundamental scenarios construction and an example of trajectories reweighting,
2. `weather_scenarios_analysis.ipynb`, containing the additional analysis of fundamental scenarios and their distribution,
3. `Forecasting_results_analysis/diebold_mariano_mae_crps.py`, resulting in two standard Diebold-Mariano p-value plots: one for MAE and one for CRPS,
4. `Forecasting_results_analysis/plot_mae_crps_validation_by_t0.py` and `Forecasting_results_analysis\best_reweighting_parameters.py`, producing a plot showing the impact of reweighting on MAE and CRPS and the kernel weighting parameters resulting from calibration with MAE and CRPS objectives,
5. `strategies_results_parser.py`, which aggregates the trading strategies results into tables of format similar to the one we use in the paper.

Each notebook and script saves the generated figures in the `PAPER_FIGURES` directory and the generated tables in `PAPER_TABLES` directory.

Alternatively, one can generate **most** of the figures from the paper using the precomputed intermediate files by running the notebooks and scripts right after downloading the repository, downloading the intermediate files from [link](https://drive.google.com/drive/folders/1W0-t2OIbTrZoHCUhn81qdaKbFenzyZIE?usp=drive_link) and saving these additional files in the `MAE_CRPS_RESULTS` directory.

## Software requirements
The simulation uses Python 3.11.
A full list of packages needed for recalculating the simulation can be found in the requirements.txt file.
To generate figures, a full LaTeX installation is required. Requirements for text rendering with LaTeX in Matplotlib can be found here: [link](https://matplotlib.org/stable/users/explain/text/usetex.html).

## Data availability and provenance
The raw data is stored in the `Data` directory. In this repository it contains the exogenous variables used in the forecasting study: crossborder physical and commercial flows, day-ahead quarter-hourly German market electricity prices, day-ahead hourly prices of German borders, load, solar generation and wind generation actual values and forecasts.
All of the aforementioned data was sourced from ENTSO-E.

The non-public directories are **not** attached in `Data`: `Intraday_Auction` and `Transactions/`, containing respectively the German intraday auction market prices and continuous intraday transaction prices and volumes. Data stored in these directories is a part of a package "DE Trades on the continuous market - Histo (up to Y-1)":
https://webshop.eex-group.com/data-type/de-trades-continuous-market-histo-y-1. The data has been purchased from the EXPEX Spot under University License, under which the Contracting Party is entitled to a limited Internal Usage in unchanged format according to Section 3 of the General Conditions, specifically for educational and academic research purposes and publication of results of analysis and research. **The Agreement with the EPEX Spot do not allow to transfer the data to third Parties.** It can be accessed through EPEX Spot sFTP server. The yearly cost of this access is equal to 480EUR.

## Hardware requirements and expected runtime
The simulation relies on heavy usage of parallel computing.

It was performed using the resources of Wrocław Centre for Networking and Supercomputing (WCSS).
Specifically, CPU: 2 x Intel Xeon Platinum 8268 (24 cores, 2,9 GHz), RAM: 192 GB 2933 MHz ECC DDR4.

The runtime for all of the cSVR models simulation on four such nodes, split by deliveries, using effectively 192 parallel workers, was around one day.
Median trading strategies calibration done by two nodes, one for the seller and one for spread trader took one day.
The calibration of bands strategies, done by four nodes, split by agent type and band type took 1.5 days.
Finally, the kernel weighting parameters calibration based on MAE and CRPS objectives was performed on one node and took 10 hours.

## Running the forecasting simulation
**If you only want to regenerate figures and tables from the paper, it is enough to run the scripts and notebooks specified in **Overview & contents** section.**
If your goal is to run the complete simulation, please follow the steps below.

### Preprocess the data
Store the downloaded EPEX Spot continuous market transactions in yearly directories in `Data/Transactions/` folder and the intraday auction prices in `Data/Intraday_Auction/Aggregated curves`. In this study, these are `2018`, `2019` and `2020` directories containing daily `.csv` files with transactions corresponding to this delivery date.

First, use `exogenous_data_preprocessing.py` to concatenate the yearly ENTSO-E CSV files and handle daylight-saving-time transitions.

Next, run `continuous_market_data_preprocessing.py` to preprocess the continuous-market data as described in the paper.

Finally, run `elasticities_computation.py` to calculate the elasticities.

### Run the simulation
Run the `forecasting_simulation_runner.py` script to schedule all of the cSVR simulations.
Run the `benchmark_forecasting_simulation_runner.py` script to schedule all of the naive benchmark simulations.

### Calculate MAE/CRPS aggregations and Diebold-Mariano test p-values
The forecasts can be analyzed using accuracy measures.

For that, run the `Forecasting_results_analysis/csvr_crps_and_mae_calc.py` script.
By default it will save the results in `MAE_CRPS_RESULTS`, which we use as a main source of the forecasting simulation intermediate files.
After completing these steps, you can also run the `forecasting_study_tables_and_figures.ipynb` on your own results.

The pairwise Diebold-Mariano test workflow is a separate logic implemented in `Forecasting_results_analysis/diebold_mariano_mae_crps.py`.

## Running the trading strategy simulation

### Running the trading strategies calibration

The median and bands calibrations are run sequentially. First calibrate the
median strategies using `strategies_calibration_runner.py`.

Next, make the resulting median calibration CSVs available in
`TRADING_STRATEGIES_RESULTS/CALIBRATION_MEASURES` and run `Trading_strategies/best_median_parameters.py`.

The script prints the best median parameters for every model and the resulting
`p_list`, `lambda_list`, and `trust_threshold_method`. Use these suggested
values in `bands_grid_config` in
`config/trading_strategies_calibration_config.py`, then calibrate the bands
strategies with the regular runner `strategies_calibration_runner.py`.

### Running the strategies evaluation

After both calibration stages, run `strategies_validation_runner.py`.

After completing these steps, you can run the `strategies_results_parser.py` on your own results.

### Evaluating forecast reweighting with MAE and CRPS

Run the calibration and subsequent validation of the dynamic scenario weights
with `trajectories_reweighting_mae_and_crps_calibration_runner.py`.
