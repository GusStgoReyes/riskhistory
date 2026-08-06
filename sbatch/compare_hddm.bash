#!/bin/bash
set -euo pipefail

# Usage: bash sbatch/compare_hddm.bash [dataset_name] [diagnostics_job_id]
#
# The model list (ALL_MODELS) and comparison groups (MODEL_GROUPS) are
# defined directly in compare_models.py, not read from a CSV here.

REPO_PTH=/scratch/users/gustxsr/riskhistory
DATASET_NAME=${1:-equalIndifference}
DIAG_JOB_ID=${2:-}
PYTHONPATH="$REPO_PTH/src"
MODULE="riskhistory.analysis_ddm.compare_models"

source "$REPO_PTH/sbatch/configs/general.sh"

echo "-------------------------------------------------"
echo " Dataset:     $DATASET_NAME"
echo " Dependency:  ${DIAG_JOB_ID:-none}"
echo "-------------------------------------------------"

TEMP_SBATCH=$(mktemp)

cat > "$TEMP_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=compare_hddm
#SBATCH --time=00:30:00
#SBATCH --partition=normal,russpold
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PYTHONPATH="$PYTHONPATH" 

singularity exec \\
  --env "PYTHONPATH=$REPO_PTH/src" \\
  --env "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1" \\
  "$HDDM_SIF" \\
  python -m "$MODULE" \\
  --dataset_name "$DATASET_NAME" \\
  --results_pth "$RESULTS_DIR"
EOF

export REPO_PTH DATASET_NAME RESULTS_DIR PYTHONPATH MODULE HDDM_SIF

DEPENDENCY_FLAG=""
if [[ -n "$DIAG_JOB_ID" ]]; then
  DEPENDENCY_FLAG="--dependency=afterok:${DIAG_JOB_ID}"
fi

echo "Submitting comparison job..."
sbatch --export=ALL $DEPENDENCY_FLAG "$TEMP_SBATCH"
