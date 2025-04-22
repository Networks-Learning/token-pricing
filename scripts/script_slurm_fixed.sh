#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=a100
#SBATCH --gres gpu:1            # set 1 GPUs per job
#SBATCH --mem=40G

#SBATCH -o /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.out      
#SBATCH -e /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.err      





source /NL/token-pricing/work/env/bin/activate

# run python script
cd /NL/token-pricing/work/src
python tokenizations_fixed.py 
deactivate