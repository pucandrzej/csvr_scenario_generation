# Replication package for "Scenario generation of intraday electricity price paths for optimal trading in continuous markets".
## Authors
Andrzej Puć, Joanna Janczura

Wrocław University of Science and Technology, Faculty of Pure and Applied Mathematics, Hugo Steinhaus Center, Wyb. Wyspiańskiego 27, Wrocław, 50-370, Poland

### Contact information
andrzej.puc@pwr.edu.pl

## Date of replication package creation
2026.05.11

## Overview & contents
The code in this replication material allows for recalculating the forecasting simulation which served as an illustration of forecasting methodology proposed in the paper "Scenario generation of intraday electricity price paths for optimal trading in continuous markets". 
When the simulation is recalculated, each figure presented in the paper can be generated using `forecasting_study_tables_and_figures.ipynb` and `weather_scenarios_analysis.ipynb` notebooks. Each notebook saves the generated figures in the `PAPER_FIGURES` directory and the generated tables in `PAPER_TABLES` directory. Trading strategies results are aggregated to the format coherent with the one visible in the publication by `strategies_results_parser.py` which saves the resulting tables in `PAPER_TABLES`.

Alternatively, one can generate figures from the paper using the precomputed intermediate files by running the notebooks and `strategies_results_parser.py` right after downloading the repository, downloading the intermediate files from [link](https://drive.google.com/drive/folders/1W0-t2OIbTrZoHCUhn81qdaKbFenzyZIE?usp=drive_link) and saving these additional files in the `MAE_CRPS_RESULTS` directory.

## Software requirements
The simulation used Python 3.11.
A full list of packages needed for recalculating the simulation can be found in the requirements.txt file.
To generate figures a full LaTeX installation is required. Requirements for text rendering with LaTeX in Matplotlib can be found here: [link](https://matplotlib.org/stable/users/explain/text/usetex.html).

## Data availability and provenance
The raw data is stored in the `Data` directory. In this repository it contains the exogenous variables used in the forecasting study: crossborder physical flows, day-ahead quarter-hourly German market electricity prices, SPV and wind generation actual values and forecasts and load actual values and forecasts.
All of the aforementioned data was sourced from ENTSO-E.

The non-public directories are not attached in `Data`: `Intraday_auction` and `Transactions/`. Data stored in these directories is a part of a package "DE Trades on the continuous market - Histo (up to Y-1)":
https://webshop.eex-group.com/data-type/de-trades-continuous-market-histo-y-1. The data has been purchased from the EXPEX Spot under University License, under which the Contracting Party is entitled to a limited Internal Usage in unchanged format according to Section 3 of the General Conditions, specifically for educational and academic research purposes and publication of results of analysis and research. The Agreement with the EPEX Spot do not allow to transfer the data to third Parties. It can be accessed through EPEX Spot sFTP server. The yearly cost of this access is equal to 480EUR.

## Hardware requirements and expected runtime
The simulation relies on heavy usage of parallel computing.
It was performed using the resources of Wrocław Centre for Networking and Supercomputing (WCSS).
Specifically, CPU: 2 x Intel Xeon Platinum 8268 (24 cores, 2,9 GHz), RAM: 192 GB 2933 MHz ECC DDR4.
Runtime on such config, using 48 parallel workers, is around two days for cSVR models simulation.

## Running the forecasting simulation
**If you only want to regenerate figures from the paper, it is enough to run the `forecasting_study_tables_and_figures.ipynb` and `weather_scenarios_analysis.ipynb` notebooks.**
If your goal is to run the complete simulation, please follow the steps below.

### Preprocess the data
Store the downloaded EPEX Spot continuous market transactions in yearly directories in `Data/Transactions/` folder/ In this study, these are `2018`, `2019` and `2020` directories containing daily `.csv` files with transactions corresponding to this delivery date.

Run the `continuous_market_data_preprocessing.py` to preprocess the data in line with preprocessing approach described in the paper.

Run the `elasticities_computation.py` to calculate the elasticities.

Use the `exogenous_data_preprocessing.py` script for concatenation of ENTSO-E yearly csvs and dst handling.

### Run the simulation
Run the `forecasting_simulation_runner.py` script to schedule all of the cSVR simulations.
Run the `benchmark_forecasting_simulation_runner.py` script to schedule all of the naive benchmark simulations.

### Calculate MAE/CRPS aggregations
Finally, the forecasts can be analyzed using accuracy measures.

For that, run the `Forecasting_results_analysis/csvr_crps_and_mae_calc.py` script.
By default it will save the results in `MAE_CRPS_RESULTS`, which we use as a main source of the intermediate files.

After completing these steps, you can run the `forecasting_study_tables_and_figures.ipynb` on your own results.

## Running the trading strategy simulation

### Running the trading strategies calibration
Run the `strategies_calibration_runner.py`.

### Running the strategies evaluation
Run the `strategies_validation_runner.py`.

After completing these steps, you can run the `strategies_results_parser.py` on your own results.
