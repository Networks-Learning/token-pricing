#!/bin/bash

source env/bin/activate

# set fixed parameters
exp_name="test"
# N=10
iterations=10
# prob=0.5
# seed=42
output_dir="outputs"

# define parameter sets
declare -a N_seq=(10 20)
declare -a prob_seq=(0.1 0.5)
declare -a seed_seq=(1 2 3 4 5 6 7 8 9 10)

# Loop over parameter sets and submit jobs
for N in "${N_seq[@]}"; do
    for prob in "${prob_seq[@]}"; do
        for seed in "${seed_seq[@]}"; do
            sbatch --export=ALL,exp_name=$exp_name,N=$N,iterations=$iterations,prob=$prob,seed=$seed,output_dir=$output_dir scripts/script_slurm.sh
        done
    done
done

deactivate