#!/bin/bash

echo "Starting the script..."

# Base directory for the project (adjust this to the project's root directory)
BASE_DIR="$(pwd)/../.."  # Assuming the script is in `work/scripts`, this goes two levels up

# File containing the list of string arguments (relative to the base directory)
TEXT_FILE="${BASE_DIR}/data/LMSYS.txt"

# Output directory for logs (relative to the base directory)
LOG_DIR="${BASE_DIR}/outputs/slurm_logs"

# Path to the virtual environment (relative to the base directory)
VENV_PATH="${BASE_DIR}/env"

# Path to the source directory (relative to the base directory)
SRC_DIR="${BASE_DIR}/src"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

# Number of lines per job (adjustable)
LINES_PER_JOB=100

# Total number of jobs to submit (adjustable)
TOTAL_JOBS=1

START_OFFSET=1  # Start reading from 

# Activate the virtual environment once for efficiency
source "${VENV_PATH}/bin/activate"

# Loop through the text file in chunks based on the specified parameters
for ((job_index=0; job_index<TOTAL_JOBS; job_index++)); do
    # Calculate the starting and ending lines for this job
    start_line=$((START_OFFSET + job_index * LINES_PER_JOB))
    end_line=$((start_line + LINES_PER_JOB - 1))
    
    # Extract the lines for this job and format them as space-separated quoted strings
    prompts=$(sed -n "${start_line},${end_line}p" "${TEXT_FILE}" | sed 's/"/\\"/g' | awk '{printf("\"%s\" ", $0)}')
    
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
#SBATCH -o ${LOG_DIR}/simulation_%j.out
#SBATCH -e ${LOG_DIR}/simulation_%j.err

source ${VENV_PATH}/bin/activate
cd ${SRC_DIR}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u LMSYS_dp.py --num_seq 3 --p 1 --temperature 1.3 --prompts ${prompts} --model google/gemma-3-1b-it
deactivate
EOF

done

# Deactivate the virtual environment
deactivate