#!/bin/bash
#SBATCH --job-name=main_parallel
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=pmano@uw.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --export=all

cd "$SLURM_SUBMIT_DIR"

echo "Starting job on $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

python3 -u main.py --parallel

echo "Job finished"