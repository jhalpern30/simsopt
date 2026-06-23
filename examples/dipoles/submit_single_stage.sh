#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -C cpu
#SBATCH --qos=shared
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@gmail.com
#SBATCH --output=../slurm_outputs/%x_%j.out
#SBATCH --error=../slurm_outputs/%x_%j.out

# ======================== User Configuration ================================
# Initialization directory (stage-2 directory or prior single-stage output).
INIT_DIR="../outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_000281/90_npol_10_ntor_8_VV_a_0.250_VV_b_0.282_VV_R0_1.042"

# Single target values (always one iota and one fcp for this script).
IOTA_TARGET="0.075"

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

# Set to 1 to remove outboard-midplane dipoles (--sparse).
SPARSE=0

# Pass --new to force a full re-run (skip logic disabled).
EXTRA_FLAGS=""
# EXTRA_FLAGS="--new"
# ============================================================================

# Auto-submit with a job name derived from iota / f_CP / volume target.
# Run as ./run_true_epsilon_adaptive_res_scan.sh (not bare sbatch) for naming.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    fcp_ka=$((FCP_THRESHOLD / 1000))
    job_name="iota${IOTA_TARGET}_fcp${fcp_ka}kA_vt${VOLUME_TARGET}"
    if (( SPARSE )); then
        job_name="${job_name}_sparse"
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
echo "  True epsilon-constraint ADAPTIVE-RESOLUTION run"
echo "============================================================"
echo "  INIT_DIR:        ${INIT_DIR}"
echo "  iota target:     ${IOTA_TARGET}"
echo "  f_CP [A]:        ${FCP_THRESHOLD}"
echo "  resolutions:     ${RESOLUTIONS}"
echo "  fb_thresholds:   ${FB_THRESHOLDS}"
echo "  volume target:   ${VOLUME_TARGET}"
echo "  sparse:          ${SPARSE}"
echo "  maxiter/step:    ${MAXITER}"
echo "  threads:         ${OMP_NUM_THREADS}"
echo "  output root:     ${OUTPUT_ROOT}"
echo "  extra flags:     ${EXTRA_FLAGS}"
echo "============================================================"

SPARSE_FLAGS=()
if (( SPARSE )); then
    SPARSE_FLAGS=(--sparse)
fi

python3 single_stage_true_epsilon_adaptive_res.py \
    --init-dir "${INIT_DIR}" \
    --iota-target "${IOTA_TARGET}" \
    --f-cp-threshold "${FCP_THRESHOLD}" \
    --resolutions "${RESOLUTIONS}" \
    --fb-thresholds "${FB_THRESHOLDS}" \
    --maxiter "${MAXITER}" \
    --volume-target "${VOLUME_TARGET}" \
    --output-root "${OUTPUT_ROOT}" \
    "${SPARSE_FLAGS[@]}" \
    ${EXTRA_FLAGS}

rc=$?
if (( rc != 0 )); then
    echo "Adaptive-resolution run FAILED with exit ${rc}." >&2
    exit "${rc}"
fi
echo "Adaptive-resolution run complete."
