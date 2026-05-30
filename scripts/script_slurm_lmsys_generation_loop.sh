#!/bin/bash

echo "Starting the script..."

# Base directory for the project (adjust this to the project's root directory)
BASE_DIR="$(pwd)/../.."  # Assuming the script is in `work/scripts`, this goes two levels up

# Output directory for logs (relative to the base directory)
LOG_DIR="${BASE_DIR}/outputs/slurm_logs"

# Path to the virtual environment (relative to the base directory)
VENV_PATH="${BASE_DIR}/env"

# Path to the source directory (relative to the base directory)
SRC_DIR="${BASE_DIR}/src"

# Path to the data directory (relative to the base directory)
DATA_DIR="${BASE_DIR}/data"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

# Number of lines per job (adjustable)
LINES_PER_JOB=500
TOTAL_JOBS=1
START_OFFSET=1  # Start reading from

# Parameters to sweep
LANGUAGES=("english" "spanish" "russian" "chinese")
TEMPS=(1.0 1.15 1.30)
TOPPS=(0.99 0.95 0.90)
MODELS=(
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Ministral-8B-Instruct-2410
  google/gemma-3-4b-it
  google/gemma-3-1b-it
)

# Activate the virtual environment once for efficiency
source "${VENV_PATH}/bin/activate"

for lang in "${LANGUAGES[@]}"; do
    # Select text file depending on language
    case $lang in
        english) TEXT_FILE="${DATA_DIR}/LMSYS.txt" ;;
        spanish) TEXT_FILE="${DATA_DIR}/LMSYS_esp.txt" ;;
        russian) TEXT_FILE="${DATA_DIR}/LMSYS_ru.txt" ;;
        chinese) TEXT_FILE="${DATA_DIR}/LMSYS_ch.txt" ;;
    esac

    for temp in "${TEMPS[@]}"; do
        for topp in "${TOPPS[@]}"; do
            for model in "${MODELS[@]}"; do
                for ((job_index=0; job_index<TOTAL_JOBS; job_index++)); do
                    start_line=$((START_OFFSET + job_index * LINES_PER_JOB))
                    end_line=$((start_line + LINES_PER_JOB - 1))
                    prompts=$(sed -n "${start_line},${end_line}p" "$TEXT_FILE" | sed 's/"/\\"/g' | awk '{printf("\"%s\" ", $0)}')

                    # Submit the job to SLURM
                    sbatch <<EOF
#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -o ${LOG_DIR}/simulation_%j.out
#SBATCH -e ${LOG_DIR}/simulation_%j.err

source ${VENV_PATH}/bin/activate
cd ${SRC_DIR}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running with: language=${lang}, model=${model}, temp=${temp}, top-p=${topp}"

python -u LMSYS_generation.py \\
    --num_seq 1 \\
    --p ${topp} \\
    --temperature ${temp} \\
    --prompts ${prompts} \\
    --model ${model} \\
    --language ${lang}

deactivate
EOF
                done
            done
        done
    done
done

# Deactivate the virtual environment
deactivate
