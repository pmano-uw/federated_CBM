#!/bin/sh
#SBATCH --time=0-5:00:00
#SBATCH --mail-user=pojtanut@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=1g
#SBATCH --account=alkontar1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

echo "dtsrting"

for dataset in nasa1 nasa2 nasa3 nasa4; do
    for num_site in 20 50 100; do
        echo "Running experiment with number of sites: $num_site"
        python3 main.py --num-site=$num_site --dataset=$dataset
    done
done
