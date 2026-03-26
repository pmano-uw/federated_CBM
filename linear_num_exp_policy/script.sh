#!/bin/sh
#SBATCH --time=0-20:00:00
#SBATCH --mail-user=pojtanut@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=8g
#SBATCH --account=stats_dept1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "Starting experiments"

RUN_ID=$(date +"%Y-%m-%d-%H-%M-%S")
RUN_DIR="experiment_results/run_${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "Saving all experiment outputs to: $RUN_DIR"

for noise in 1 3; do
    echo "Running algorithm: collaborative (dataset=nasa1, lap_noise=$noise)" && \
    python3 main.py --savelog --experiment="collaborative" --lap-noise=$noise --dataset "nasa1" --output-dir "$RUN_DIR"
done

for experiment in collaborative isolated EP; do
    echo "Running algorithm: $experiment (dataset=nasa1, lap_noise=0)" && \
    python3 main.py --savelog --experiment=$experiment --dataset "nasa1" --output-dir "$RUN_DIR"
done
