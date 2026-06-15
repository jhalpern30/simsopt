#!/bin/bash
#SBATCH -A m1266
#SBATCH --job-name=fcp100kA_small_scan
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
# Sequential warm-started true epsilon-constraint scan, ONE node, ONE f_CP.
#
# Iota is walked sequentially (low -> high) inside the python script: each
# iota step is initialized from the previous step's optimized coils + Boozer
# surface, so adjacent iota points stay in the same optimization basin.
#
# To chain across f_CP values, just edit INIT_DIR below to point at the
# iota_min subdirectory of the previously-completed walk, e.g.
#   ${OUTPUT_ROOT}/<eq_name>/iota<IOTA_MIN>_fcp<prev_fcp_kA>kA_vt<VT>/mpolM_ntorN
# and resubmit. The python loader handles either a stage-2 dir or a previous
# single-stage run dir transparently (both save the required bs_opt.json,
# surf_opt.json, and metadata-bearing results.json).
#
# Parallelism: only one walk runs, so it gets the full 128-CPU node via
# OMP_NUM_THREADS=128.
# ============================================================================

# ======================== User Configuration ================================
# Initial dir for this walk. Either a stage-2 output directory or the
# iota_min subdirectory of a previous sequential walk at a different f_CP.
# INIT_DIR="../single_stage_true_epsilon_sequential_bigger_step/wout_nfp22ginsburg_000_000281/iota0.07_fcp100kA_vt0.3/mpol6_ntor6"
INIT_DIR="../outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_000281/90_npol_10_ntor_8_VV_a_0.250_VV_b_0.282_VV_R0_1.042"

# Iota walk grid (low -> high; sequential warm-starting goes in this direction)
IOTA_MIN=0.07
IOTA_MAX=0.27
IOTA_COUNT=6

# f_CP value for this walk [A]. Run one f_CP per submission.
FCP_THRESHOLD="200000"

# Defaults passed through to single_stage_true_epsilon_sequential.py.
FB_THRESHOLD="1e-4"
IOTA_TOLERANCE="0.0025"
MPOL=6
NTOR=6
MAXITER=400
VOLUME_TARGET="0.30"
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
