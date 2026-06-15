#!/bin/bash
#SBATCH -A m4680
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -C cpu
#SBATCH --qos=shared
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacobhalpern667@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

# ======================== User Configuration ================================
# Initial dir for this walk.  Either a stage-2 output directory or the
# highest-resolution iota_min subdirectory of a previous walk at a different
# f_CP.
INIT_DIR="../outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_000281/90_npol_10_ntor_8_VV_a_0.250_VV_b_0.282_VV_R0_1.042"
# INIT_DIR="../single_stage_true_epsilon_adaptive_res/wout_nfp22ginsburg_000_000281/iota0.25_fcp200kA_vt0.3/mpol6_ntor6"

# Iota walk grid (low -> high; sequential warm-starting runs in this direction)
IOTA_MIN=0.075
IOTA_MAX=0.075
IOTA_COUNT=1

# f_CP value for this walk [A].  Run one f_CP per submission.
FCP_THRESHOLD="100000"

# Resolution ladder: comma-separated list of mpol=ntor values.
# Each entry must have a matching entry in FB_THRESHOLDS.
# RESOLUTIONS="6,9,12"
RESOLUTIONS="12"

# Boozer residual thresholds, one per resolution level.
# FB_THRESHOLDS="1e-5,7.5e-6,5e-6"
FB_THRESHOLDS="5e-6"


# Other optimisation parameters (passed through unchanged).
MAXITER=500
VOLUME_TARGET="0.3"
OUTPUT_ROOT="../single_stage_true_epsilon_adaptive_res_lower_residual"

# Pass --new to force a full re-run (skip logic disabled).
EXTRA_FLAGS=""
# EXTRA_FLAGS="--new"
# ============================================================================

# Auto-submit with a job name derived from iota / f_CP / volume target.
# Run as ./run_true_epsilon_adaptive_res_scan.sh (not bare sbatch) for naming.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    fcp_ka=$((FCP_THRESHOLD / 1000))
    job_name="iota${IOTA_MIN}_fcp${fcp_ka}kA_vt${VOLUME_TARGET}"
    if (( IOTA_COUNT > 1 )) || ! awk -v a="${IOTA_MIN}" -v b="${IOTA_MAX}" 'BEGIN{exit (a==b)?0:1}'; then
        job_name="${job_name}_scan"
    fi
    exec sbatch --job-name="${job_name}" "$0"
fi

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
