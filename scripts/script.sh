#!/bin/bash

source env/bin/activate

exp_name="test"
# N=10
iterations=10
# prob=0.5
# seed=42
output_dir="outputs"

# individual run (python)
# python src/experiment.py --exp_name=$exp_name --N=$N --iterations=$iterations --prob=$prob --seed=$seed --output_dir=$output_dir

# loop over configurations (make sure that these are the same ones that appear in the output filenames specified in py/jl)
N_seq=(10 20)
prob_seq=(0.1 0.5)
seed_seq=(1 2 3 4 5 6 7 8 9 10)
for N in "${N_seq[@]}"; do
    for prob in "${prob_seq[@]}"; do
        for seed in "${seed_seq[@]}"; do
            python src/experiment.py --exp_name=$exp_name --N=$N --iterations=$iterations --prob=$prob --seed=$seed --output_dir=$output_dir
        done
    done
done

deactivate