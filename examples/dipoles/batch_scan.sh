#!/bin/bash
#SBATCH -A m1266
#SBATCH --job-name=single_stage
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128   # one process per core; each core runs one optimization
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com

# Purge all modules to remove the "Application linked against multiple cray-libsci libraries" warning
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh

# Load Python module
module load python/3.11

# Activate virtual environment
conda activate simsopt

# Number of runs - default to 100, but can be overridden by the first positional argument
N=${1:-100}

# Redirect stdout/stderr to a dated, run-count, job-specific log file:
#   YYMMDD_<N>_<JOBID>.out
OUT_DIR="../single_stage_scans_no_sparsity_cp_fix/slurm_outputs"
mkdir -p "$OUT_DIR"
DATE_STR=$(date +%y%m%d)
OUT_FILE="${OUT_DIR}/${DATE_STR}_${N}_${SLURM_JOB_ID:-local}.out"
exec >"$OUT_FILE" 2>&1

# Stage 2 directory to initialize from
INIT_DIR="../outputs/stage_2_LHS_3D_scan_no_sparsity/59_20260312_diprad_0.050_VV_a_0.263_VV_b_0.282_VV_R0_1.028"

# How many runs to execute in parallel on this node and how many threads to use for each.
export OMP_NUM_THREADS=8        # or 8, 16 – you can experiment
MAX_JOBS=$(( SLURM_CPUS_PER_TASK / OMP_NUM_THREADS ))
echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS per job, MAX_JOBS=$MAX_JOBS"

running=0
completed=0

for i in $(seq 0 $((N-1))); do
    # Generate random input parameters for the optimization
    PARAMS=$(python3 generate_inputs.py)
    echo "[$((i+1))/$N] Starting optimization $i with parameters $PARAMS"

    # Launch this optimization in the background.
    python3 single_stage_dipole_example.py --init-dir "$INIT_DIR" $PARAMS &
    ((running++))

    # If we've reached the concurrency limit, wait for at least one job to finish.
    if (( running >= MAX_JOBS )); then
        wait -n
        ((running--))
        ((completed++))
        echo "[$(date +%H:%M:%S)] One iteration finished — completed: $completed/$N, still running: $running"
    fi
done

# Wait for any remaining jobs to finish and print completion status.
while (( running > 0 )); do
    wait -n
    ((running--))
    ((completed++))
    echo "[$(date +%H:%M:%S)] One iteration finished — completed: $completed/$N, still running: $running"
done
echo "All $N iterations completed."