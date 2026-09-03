#!/bin/sh
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=pojtanut@umich.edu
#SBATCH --mail-type=END
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=500m
#SBATCH --account=alkontar1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

echo "dtsrting"

# Array of lap noise values
lap_noise_values=(1 3)

# Iterate over each lap noise value and execute the command
for noise in "${lap_noise_values[@]}"; do
  echo "Running with lap noise: $noise"
  python3 main.py --experiment collaborative --sim-num 30 --window 3 --parallel --savelog --lap-noise "$noise"
done