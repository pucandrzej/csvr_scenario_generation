#!/bin/bash
#SBATCH -N1
#SBATCH -c48
#SBATCH --mem=192gb
#SBATCH --time=168:00:00
#SBATCH --mail-user=andrzej.puc@pwr.edu.pl
#SBATCH --job-name=median_seller_calibration

source /usr/local/sbin/modules.sh
module load Python/3.11.5-GCCcore-13.2.0
module load virtualenv/20.24.6-GCCcore-13.2.0

virtualenv contmarket311

python -m Trading_strategies.strategies_simulation \
    --model median \
    --run_type calibration \
    --one_sided \
    --band_type risk_seeking \
    --processes 48 \
    --special_results_directory '/lustre/pd01/hpc-andpuc2524-1756857346'
