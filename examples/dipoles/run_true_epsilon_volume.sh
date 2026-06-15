#!/bin/bash
#SBATCH -A m1266
#SBATCH --job-name=vol0.45_hr
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -C cpu
#SBATCH --qos=shared
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

# ======================== User Configuration ================================
INIT_DIR="../single_stage_true_epsilon_sequential_bigger_step/wout_nfp22ginsburg_000_000281/iota0.15_fcp200kA_vt0.45/mpol6_ntor6"

# Iota walk grid (low -> high; sequential warm-starting goes in this direction)
IOTA_MIN=0.15
IOTA_MAX=0.15
IOTA_COUNT=1

# f_CP value for this walk [A]. Run one f_CP per submission.
FCP_THRESHOLD="200000"

# Defaults passed through to single_stage_true_epsilon_sequential.py.
FB_THRESHOLD="5e-5"
IOTA_TOLERANCE="0.0025"
MPOL=9
NTOR=9
MAXITER=400
VOLUME_TARGET="0.45"
OUTPUT_ROOT="../single_stage_true_epsilon_sequential_bigger_step"

# Pass --new to force the walk to redo every iota point (instead of skipping
# already-completed ones in OUTPUT_ROOT).
EXTRA_FLAGS=""
# EXTRA_FLAGS="--new"
# ============================================================================

# Purge all modules to remove the "Application linked against multiple cray-libsci
# libraries" warning (matches run_true_epsilon_scan.sh).
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
conda activate simsopt

# Single walk -> hand the entire node's CPUs to BLAS/OpenMP.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p ../slurm_outputs

echo "============================================"
echo "  True epsilon-constraint SEQUENTIAL scan"
echo "============================================"
echo "  INIT_DIR:        ${INIT_DIR}"
echo "  iota walk:       [${IOTA_MIN}, ${IOTA_MAX}] x ${IOTA_COUNT} (sequential warm-start)"
echo "  f_CP [A]:        ${FCP_THRESHOLD}"
echo "  volume target:   ${VOLUME_TARGET}"
echo "  mpol/ntor:       ${MPOL}/${NTOR}"
echo "  maxiter/step:    ${MAXITER}"
echo "  threads:         ${OMP_NUM_THREADS}"
echo "  output root:     ${OUTPUT_ROOT}"
echo "  extra flags:     ${EXTRA_FLAGS}"
echo "============================================"

python3 single_stage_true_epsilon_sequential.py \
    --init-dir "${INIT_DIR}" \
    --iota-min "${IOTA_MIN}" --iota-max "${IOTA_MAX}" --iota-count "${IOTA_COUNT}" \
    --f-cp-threshold "${FCP_THRESHOLD}" \
    --f-b-threshold "${FB_THRESHOLD}" \
    --iota-threshold "${IOTA_TOLERANCE}" \
    --mpol "${MPOL}" --ntor "${NTOR}" \
    --maxiter "${MAXITER}" \
    --volume-target "${VOLUME_TARGET}" \
    --output-root "${OUTPUT_ROOT}" \
    ${EXTRA_FLAGS}

rc=$?
if (( rc != 0 )); then
    echo "Sequential walk FAILED with exit ${rc}." >&2
    exit "${rc}"
fi
echo "Sequential walk complete."
