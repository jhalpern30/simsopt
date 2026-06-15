#!/bin/bash
#SBATCH -A m1266
#SBATCH --job-name=200kA_v0.35
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -C cpu
#SBATCH --qos=shared
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

# ============================================================================
# Adaptive-resolution sequential warm-started true epsilon-constraint scan.
#
# For each iota target the script climbs a ladder of (mpol=ntor, fb_threshold)
# pairs before advancing to the next iota.  The warm-start chain is:
#
#   (iota_0, res[0]) <- INIT_DIR
#   (iota_i, res[j+1]) <- (iota_i, res[j])       ← within iota
#   (iota_{i+1}, res[0]) <- (iota_i, res[-1])     ← highest res of prev iota
#
# Each (iota, resolution) pair is written under
#   <OUTPUT_ROOT>/<eq_name>/iota<X>_fcp<Y>kA_vt<Z>/mpol<M>_ntor<N>/
# (same layout as the single-resolution scripts).
#
# Parallelism: one f_CP per submission; give the whole node to BLAS/OpenMP.
# ============================================================================

# ======================== User Configuration ================================
# Initial dir for this walk.  Either a stage-2 output directory or the
# highest-resolution iota_min subdirectory of a previous walk at a different
# f_CP.
INIT_DIR="../single_stage_true_epsilon_adaptive_res/wout_nfp22ginsburg_000_000281/iota0.15_fcp200kA_vt0.3/mpol12_ntor12"

# Volume only
IOTA_MIN=0.15
IOTA_MAX=0.15
IOTA_COUNT=1

# f_CP value for this walk [A].  Run one f_CP per submission.
FCP_THRESHOLD="200000"

# Resolution ladder: comma-separated list of mpol=ntor values.
# Each entry must have a matching entry in FB_THRESHOLDS.
RESOLUTIONS="6,9,12"
# RESOLUTIONS="9,12"


# Boozer residual thresholds, one per resolution level.
# The script emits a RuntimeWarning (not an error) if a step finishes with
# residual > threshold * 1.05.
FB_THRESHOLDS="1e-4,5e-5,2.5e-5"
# FB_THRESHOLDS="5e-5,2.5e-5"

# Other optimisation parameters (passed through unchanged).
MAXITER=400
VOLUME_TARGET="0.35"
OUTPUT_ROOT="../single_stage_true_epsilon_adaptive_res"

# Pass --new to force a full re-run (skip logic disabled).
EXTRA_FLAGS=""
# EXTRA_FLAGS="--new"
# ============================================================================

# Purge all modules to remove the "Application linked against multiple
# cray-libsci libraries" warning (matches run_true_epsilon_scan.sh).
module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
conda activate simsopt

# Single walk -> hand the entire node's CPUs to BLAS/OpenMP.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p ../slurm_outputs

echo "============================================================"
echo "  True epsilon-constraint ADAPTIVE-RESOLUTION SEQUENTIAL scan"
echo "============================================================"
echo "  INIT_DIR:        ${INIT_DIR}"
echo "  iota walk:       [${IOTA_MIN}, ${IOTA_MAX}] x ${IOTA_COUNT} (sequential warm-start)"
echo "  f_CP [A]:        ${FCP_THRESHOLD}"
echo "  resolutions:     ${RESOLUTIONS}"
echo "  fb_thresholds:   ${FB_THRESHOLDS}"
echo "  volume target:   ${VOLUME_TARGET}"
echo "  maxiter/step:    ${MAXITER}"
echo "  threads:         ${OMP_NUM_THREADS}"
echo "  output root:     ${OUTPUT_ROOT}"
echo "  extra flags:     ${EXTRA_FLAGS}"
echo "============================================================"

python3 single_stage_true_epsilon_adaptive_res.py \
    --init-dir "${INIT_DIR}" \
    --iota-min "${IOTA_MIN}" --iota-max "${IOTA_MAX}" --iota-count "${IOTA_COUNT}" \
    --f-cp-threshold "${FCP_THRESHOLD}" \
    --resolutions "${RESOLUTIONS}" \
    --fb-thresholds "${FB_THRESHOLDS}" \
    --maxiter "${MAXITER}" \
    --volume-target "${VOLUME_TARGET}" \
    --output-root "${OUTPUT_ROOT}" \
    ${EXTRA_FLAGS}

rc=$?
if (( rc != 0 )); then
    echo "Adaptive-resolution sequential walk FAILED with exit ${rc}." >&2
    exit "${rc}"
fi
echo "Adaptive-resolution sequential walk complete."
