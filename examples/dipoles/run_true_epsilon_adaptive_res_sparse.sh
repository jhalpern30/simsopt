#!/bin/bash
#SBATCH -A m4680
#SBATCH --time=48:00:00
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
# Single-iota adaptive-resolution sparse true epsilon run.
#
# Same adaptive-resolution workflow as run_true_epsilon_adaptive_res_scan.sh,
# but with dipole sparsity enabled (outboard coils removed by theta tolerance).
# ============================================================================

# ======================== User Configuration ================================
# Initialization directory (stage-2 directory or prior single-stage output).
# INIT_DIR="../single_stage_true_epsilon_adaptive_res/wout_nfp22ginsburg_000_000281/iota0.1_fcp100kA_vt0.3/mpol12_ntor12"
# INIT_DIR="../single_stage_true_epsilon_adaptive_res/wout_nfp22ginsburg_000_000281/iota0.2_fcp150kA_vt0.3/mpol12_ntor12"
INIT_DIR="../single_stage_true_epsilon_adaptive_res_lower_residual/wout_nfp22ginsburg_000_000281/iota0.1_fcp250kA_vt0.3/mpol12_ntor12"
# INIT_DIR="../single_stage_true_epsilon_adaptive_res/wout_nfp22ginsburg_000_000281/iota0.15_fcp150kA_vt0.3/mpol12_ntor12"

# Single target values (always one iota and one fcp for this script).
IOTA_TARGET="0.05"
FCP_THRESHOLD="250000"

# Adaptive resolution ladder.
RESOLUTIONS="12"
FB_THRESHOLDS="5e-6"

# Other optimization parameters.
MAXITER=500
VOLUME_TARGET="0.3"
THETA_TOL="0.01"
OUTPUT_ROOT="../single_stage_true_epsilon_adaptive_res_lower_residual_sparse"

# Pass --new to force a full rerun.
EXTRA_FLAGS=""
# EXTRA_FLAGS="--new"
# ============================================================================

# Auto-submit with a job name derived from iota / f_CP / volume target.
# Run as ./run_true_epsilon_adaptive_res_sparse.sh (not bare sbatch) for naming.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    fcp_ka=$((FCP_THRESHOLD / 1000))
    job_name="iota${IOTA_TARGET}_fcp${fcp_ka}kA_vt${VOLUME_TARGET}_sparse"
    exec sbatch --job-name="${job_name}" "$0"
fi

module purge
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
module load python/3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate simsopt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p ../slurm_outputs

echo "============================================================"
echo "  True epsilon adaptive-resolution SPARSE run"
echo "============================================================"
echo "  INIT_DIR:        ${INIT_DIR}"
echo "  iota target:     ${IOTA_TARGET}"
echo "  f_CP [A]:        ${FCP_THRESHOLD}"
echo "  resolutions:     ${RESOLUTIONS}"
echo "  fb_thresholds:   ${FB_THRESHOLDS}"
echo "  volume target:   ${VOLUME_TARGET}"
echo "  theta_tol [rad]: ${THETA_TOL}"
echo "  maxiter/step:    ${MAXITER}"
echo "  threads:         ${OMP_NUM_THREADS}"
echo "  output root:     ${OUTPUT_ROOT}"
echo "  extra flags:     ${EXTRA_FLAGS}"
echo "============================================================"

python3 single_stage_true_epsilon_adaptive_res_sparse.py \
    --init-dir "${INIT_DIR}" \
    --iota-target "${IOTA_TARGET}" \
    --f-cp-threshold "${FCP_THRESHOLD}" \
    --resolutions "${RESOLUTIONS}" \
    --fb-thresholds "${FB_THRESHOLDS}" \
    --maxiter "${MAXITER}" \
    --volume-target "${VOLUME_TARGET}" \
    --theta-tol "${THETA_TOL}" \
    --output-root "${OUTPUT_ROOT}" \
    ${EXTRA_FLAGS}

rc=$?
if (( rc != 0 )); then
    echo "Adaptive-resolution sparse run FAILED with exit ${rc}." >&2
    exit "${rc}"
fi
echo "Adaptive-resolution sparse run complete."
