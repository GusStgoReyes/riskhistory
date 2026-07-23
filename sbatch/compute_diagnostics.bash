#!/bin/bash
set -euo pipefail

# ======================
# User settings
# ======================
REPO_PTH=/scratch/users/gustxsr/riskhistory
DATASET_NAME=${1:-equalIndifference}
PYTHONPATH="$REPO_PTH/src"
MODULE="riskhistory.analysis_ddm.compute_diagnostics"

source "$REPO_PTH/sbatch/configs/general.sh"

mapfile -t models < <(tail -n +2 "$HDDM_MODEL_CSV")

N_MODELS=${#models[@]}
ARRAY_RANGE="1-${N_MODELS}"

echo "-------------------------------------------------"
echo " Dataset:     $DATASET_NAME"
echo " N_MODELS:    $N_MODELS"
echo " Array range: $ARRAY_RANGE"
echo "-------------------------------------------------"

TEMP_SBATCH=$(mktemp)

# --- header (outer vars expanded now) ---
cat > "$TEMP_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=diag_hddm
#SBATCH --time=01:00:00
#SBATCH --partition=normal,russpold
#SBATCH --array=${ARRAY_RANGE}
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G

set -euo pipefail
EOF

# --- body (inner vars expanded at runtime) ---
cat >> "$TEMP_SBATCH" <<'EOF'

mapfile -t models < <(tail -n +2 "$HDDM_MODEL_CSV")
IDX=$((SLURM_ARRAY_TASK_ID - 1))
IFS=',' read -r MODEL_ID PUPIL_COL <<< "${models[$IDX]}"

echo "Computing diagnostics for: $MODEL_ID (pupil_col=$PUPIL_COL)"

cd $REPO_PTH

PYTHONPATH="$PYTHONPATH"

singularity exec --env "PYTHONPATH=$REPO_PTH/src" "$HDDM_SIF" \
  python -m riskhistory.analysis_ddm.compute_diagnostics \
  --model_id "$MODEL_ID" \
  --dataset_name "$DATASET_NAME" \
  --data_pth "$BEHAV_CSV" \
  --pupil_pth "$PUPIL_CSV" \
  --pupil_col "$PUPIL_COL" \
  --results_pth "$RESULTS_DIR"
EOF

export REPO_PTH DATASET_NAME RESULTS_DIR HDDM_MODEL_CSV PYTHONPATH MODULE BEHAV_CSV PUPIL_CSV HDDM_SIF
echo "Submitting diagnostics array job..."
ARRAY_JOB_ID=$(sbatch --export=ALL --parsable "$TEMP_SBATCH")
echo "Array job ID: $ARRAY_JOB_ID"
echo "$ARRAY_JOB_ID"
