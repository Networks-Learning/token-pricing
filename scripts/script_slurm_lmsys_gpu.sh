#!/bin/bash

# File containing the list of string arguments
TEXT_FILE="/NL/token-pricing/work/data/LMSYS.txt"

# Read only the first 20 lines of the text file
head -n 1 "$TEXT_FILE" | while IFS= read -r line || [ -n "$line" ]; do
    # Escape quotes and ensure each string is properly quoted in the command
    prompt_arg=$(echo "$line" | sed 's/\([^ ]*\)/"\1"/g') 
    # Submit a new SLURM job for each line of the text file
    sbatch <<EOF
#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --gres gpu:1            # set 1 GPUs per job
#SBATCH --constraint=h100-nvl   # request "h100-nvl" feature
#SBATCH --mem=25G
#SBATCH -o /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.out
#SBATCH -e /NL/token-pricing/work/outputs/slurm_logs/simulation_%j.err
source /NL/token-pricing/work/env/bin/activate
cd /NL/token-pricing/work/src
echo "Submitting job for prompt: ${line}"
python LMSYS_dp.py --num_seq 10 --p 0.95 --prompts ${prompt_arg}
deactivate
EOF
done