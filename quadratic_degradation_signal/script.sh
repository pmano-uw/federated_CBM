#!/bin/sh
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=pojtanut@umich.edu
#SBATCH --mail-type=END
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=1g
#SBATCH --account=alkontar1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

echo "dtsrting"

# Array of experiments
experiments=("isolated" "collaborative")

# Loop over each experiment and run the Python script
for experiment in "${experiments[@]}"
do
    python3 main.py --experiment "${experiment}" --sim-num 30 --window 3 --parallel --savelog
done