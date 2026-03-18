#!/bin/sh
#SBATCH --time=0-10:00:00
#SBATCH --mail-user=pojtanut@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=8g
#SBATCH --account=stats_dept1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "dtsrting"

for noise in 1 3; do \
    python3 main_battery.py  --savelog  --experiment="collaborative" --lap-noise=$noise --dataset "battery1"
done

for experiment in collaborative isolated EP; do \
    python3 main_battery.py  --savelog  --experiment=$experiment --dataset "battery1" 
done