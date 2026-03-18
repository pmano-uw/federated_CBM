#!/bin/sh
#SBATCH --time=0-10:00:00
#SBATCH --mail-user=pojtanut@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=1g
#SBATCH --account=stats_dept1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8

echo "dtsrting"

for experiment in collaborative isolated EP; do \
    python3 main.py  --savelog  --sim-num=100  --parallel  --experiment=$experiment
done