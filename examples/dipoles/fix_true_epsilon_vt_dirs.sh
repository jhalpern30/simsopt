#!/usr/bin/env bash
set -euo pipefail

# Rename legacy single_stage_true_epsilon run directories that do not include
# volume target in their name:
#   iota0.12_fcp50kA -> iota0.12_fcp50kA_vt0.3
#
# Default behavior is a dry run. Use --apply to perform renames.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/../single_stage_true_epsilon"
VOLUME_TARGET="0.3"
APPLY=false

usage() {
    cat <<'EOF'
Usage:
  ./fix_true_epsilon_vt_dirs.sh [--base-dir DIR] [--vt VALUE] [--apply]

Options:
  --base-dir DIR   Root containing eq directories (default: ../single_stage_true_epsilon)
  --vt VALUE       Volume target suffix value (default: 0.3)
  --apply          Perform renames (default is dry-run)
  -h, --help       Show this help text
EOF
}

while (($#)); do
    case "$1" in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --vt)
            VOLUME_TARGET="$2"
            shift 2
            ;;
        --apply)
            APPLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "${BASE_DIR}" ]]; then
    echo "Base directory does not exist: ${BASE_DIR}" >&2
    exit 1
fi

echo "Base directory: ${BASE_DIR}"
echo "Volume target:  ${VOLUME_TARGET}"
if [[ "${APPLY}" == true ]]; then
    echo "Mode:           APPLY (renaming directories)"
else
    echo "Mode:           DRY RUN (no changes)"
fi
echo

shopt -s nullglob

seen=0
renamed=0
skipped=0

for eq_dir in "${BASE_DIR}"/*; do
    [[ -d "${eq_dir}" ]] || continue

    for run_dir in "${eq_dir}"/iota*_fcp*kA; do
        [[ -d "${run_dir}" ]] || continue

        run_name="$(basename "${run_dir}")"

        # Skip anything that already has a vt tag.
        if [[ "${run_name}" == *_vt* ]]; then
            ((skipped+=1))
            continue
        fi

        ((seen+=1))
        new_dir="${run_dir}_vt${VOLUME_TARGET}"

        if [[ -e "${new_dir}" ]]; then
            echo "SKIP (target exists): ${run_dir} -> ${new_dir}"
            ((skipped+=1))
            continue
        fi

        if [[ "${APPLY}" == true ]]; then
            mv "${run_dir}" "${new_dir}"
            echo "RENAMED: ${run_dir} -> ${new_dir}"
            ((renamed+=1))
        else
            echo "WOULD RENAME: ${run_dir} -> ${new_dir}"
        fi
    done
done

echo
echo "Legacy run dirs found: ${seen}"
if [[ "${APPLY}" == true ]]; then
    echo "Renamed:             ${renamed}"
fi
echo "Skipped:             ${skipped}"
