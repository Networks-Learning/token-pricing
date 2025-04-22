#!/bin/bash

echo "Starting the script..."

# File containing the list of string arguments
TEXT_FILE="/NL/token-pricing/work/data/LMSYS.txt"

# Number of lines per job (adjustable)
LINES_PER_JOB=30

# Total number of jobs to submit (adjustable)
TOTAL_JOBS=20

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
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=a100
#SBATCH --gres gpu:1            # set 1 GPUs per job
#SBATCH --mem=40G
#SBATCH -o /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.out
#SBATCH -e /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.err

source /NL/token-pricing/work/env/bin/activate
cd /NL/token-pricing/work/src

python -u LMSYS_dp.py --num_seq 10 --p 1 --prompts ${prompts} --model mistralai/Ministral-8B-Instruct-2410
deactivate
EOF

done

# Deactivate the virtual environment
deactivate