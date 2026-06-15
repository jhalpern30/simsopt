#!/bin/bash
#SBATCH -A m1266
#SBATCH --job-name=200kA_highres
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

# ======================== User Configuration ================================
INIT_DIR="../outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_000281/90_npol_10_ntor_8_VV_a_0.250_VV_b_0.282_VV_R0_1.042"

# Scan grid
IOTA_MIN=0.07
IOTA_MAX=0.27
IOTA_COUNT=6
FCP_THRESHOLDS="200000"

# Defaults passed through to single_stage_true_epsilon.py
FB_THRESHOLD="5e-5"
IOTA_TOLERANCE="0.0025"
MPOL=9
NTOR=9
MAXITER=300
VOLUME_TARGET="0.3"
# ============================================================================

# Purge all modules to remove the "Application linked against multiple cray-libsci libraries" warning
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh

# Load Python module
module load python/3.11

# Activate virtual environment
conda activate simsopt

# How many runs to execute in parallel on this node
export OMP_NUM_THREADS=16
MAX_JOBS=$(( SLURM_CPUS_PER_TASK / OMP_NUM_THREADS ))

N=$(python3 generate_inputs_true_epsilon.py \
    --iota-min "${IOTA_MIN}" --iota-max "${IOTA_MAX}" --iota-count "${IOTA_COUNT}" \
    --f-cp-thresholds "${FCP_THRESHOLDS}" --total)

echo "============================================"
echo "  True epsilon-constraint scan"
echo "============================================"
echo "  INIT_DIR:       ${INIT_DIR}"
echo "  iota:           [${IOTA_MIN}, ${IOTA_MAX}] × ${IOTA_COUNT}"
echo "  f_CP [A]:       ${FCP_THRESHOLDS}"
echo "  volume target:  ${VOLUME_TARGET}"
echo "  Total runs:     ${N}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS},  MAX_JOBS=${MAX_JOBS}"
echo "============================================"

running=0
completed=0

for i in $(seq 0 $((N-1))); do
    PARAMS=$(python3 generate_inputs_true_epsilon.py \
        --iota-min "${IOTA_MIN}" --iota-max "${IOTA_MAX}" --iota-count "${IOTA_COUNT}" \
        --f-cp-thresholds "${FCP_THRESHOLDS}" \
        --f-b-threshold "${FB_THRESHOLD}" \
        --iota-threshold "${IOTA_TOLERANCE}" \
        --volume-target "${VOLUME_TARGET}" \
        --mpol "${MPOL}" --ntor "${NTOR}" --maxiter "${MAXITER}" \
        --index "$i")

    echo "[$((i+1))/${N}] Starting run $i with parameters ${PARAMS}"

    python3 single_stage_true_epsilon.py --init-dir "${INIT_DIR}" ${PARAMS} &
    ((running++))

    # Throttle: wait for one to finish before launching the next batch
    if (( running >= MAX_JOBS )); then
        wait -n
        ((running--))
        ((completed++))
        echo "[$(date +%H:%M:%S)] Finished — completed: ${completed}/${N}, running: ${running}"
    fi
done

# Drain remaining
while (( running > 0 )); do
    wait -n
    ((running--))
    ((completed++))
    echo "[$(date +%H:%M:%S)] Finished — completed: ${completed}/${N}, running: ${running}"
done

echo "All ${N} runs completed."
