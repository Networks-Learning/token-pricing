TEXT_FILE="/NL/token-pricing/work/data/LMSYS.txt"

# Number of lines per job (adjustable)
LINES_PER_JOB=2

# Total number of jobs to submit (adjustable)
TOTAL_JOBS=1

# Activate the virtual environment once for efficiency
source /NL/token-pricing/work/env/bin/activate

for ((job_index=0; job_index<TOTAL_JOBS; job_index++)); do
    # Calculate the starting and ending lines for this job
    start_line=$((job_index * LINES_PER_JOB + 1))
    end_line=$((start_line + LINES_PER_JOB - 1))
    
    # Extract the lines for this job and format them as space-separated quoted strings
    prompts=$(sed -n "${start_line},${end_line}p" "$TEXT_FILE" | sed 's/"/\\"/g' | awk '{printf("\"%s\" ", $0)}')
    echo $prompts

done

