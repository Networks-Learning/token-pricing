#!/bin/bash

echo "Starting the script..."

# File containing the list of string arguments
TEXT_FILE="/NL/token-pricing/work/data/LMSYS.txt"

# Number of lines per job (adjustable)
LINES_PER_JOB=50

# Total number of jobs to submit (adjustable)
TOTAL_JOBS=1

START_OFFSET=1  # Start reading from 

# Activate the virtual environment once for efficiency
source /NL/token-pricing/work/env/bin/activate

# Loop through the text file in chunks based on the specified parameters
for ((job_index=0; job_index<TOTAL_JOBS; job_index++)); do
    # Calculate the starting and ending lines for this job
    start_line=$((START_OFFSET + job_index * LINES_PER_JOB))
    end_line=$((start_line + LINES_PER_JOB - 1))
    
    # Extract the lines for this job and format them as space-separated quoted strings
    prompts=$(sed -n "${start_line},${end_line}p" "$TEXT_FILE" | sed 's/"/\\"/g' | awk '{printf("\"%s\" ", $0)}')
    
    # Submit the job to SLURM
    sbatch <<EOF
#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=h100
#SBATCH --gres gpu:1            # set 1 GPUs per job
#SBATCH --mem=40G
#SBATCH -o /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.out
#SBATCH -e /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.err

source /NL/token-pricing/work/env/bin/activate
cd /NL/token-pricing/work/src

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u heuristic.py --prompts ${prompts} --p 0.99 --temperature 1.3 --num_seq 1 --model meta-llama/Llama-3.2-1B-Instruct 
EOF

done



# Deactivate the virtual environment
deactivate