#!/bin/bash
set -euo pipefail

# ======================
# User settings
# ======================
REPO_PTH=/scratch/users/gustxsr/riskhistory
DATASET_NAME=${1:-equalIndifference}
PYTHONPATH="$REPO_PTH/src"
MODULE="riskhistory.analysis_ddm.compare_models"

source "$REPO_PTH/sbatch/configs/general.sh"

mapfile -t models < <(tail -n +2 "$HDDM_MODEL_CSV")

echo "-------------------------------------------------"
echo " Dataset:     $DATASET_NAME"
echo " N_MODELS:    ${#models[@]}"
echo " Models:      ${models[*]}"
echo "-------------------------------------------------"

TEMP_SBATCH=$(mktemp)

cat > "$TEMP_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=compare_hddm
#SBATCH --time=02:00:00
#SBATCH --partition=normal,russpold
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
EOF

cat >> "$TEMP_SBATCH" <<EOF

PYTHONPATH="$PYTHONPATH" singularity exec "$REPO_PTH/hddm.sif" \\
  python -m "$MODULE" \\
  --models ${models[*]} \\
  --dataset_name "$DATASET_NAME" \\
  --results_pth "$RESULTS_DIR"
EOF

export REPO_PTH DATASET_NAME RESULTS_DIR HDDM_MODEL_CSV PYTHONPATH MODULE
echo "Submitting comparison job..."
sbatch --export=ALL "$TEMP_SBATCH"
