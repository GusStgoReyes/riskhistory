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
#SBATCH --cpus-per-task=10
# Peak is ~3x the log_lik array: az.loo allocates its own full-shape
# log_weights and log_weights + log_lik on top of it, plus a transient
# second copy when the degenerate trials are dropped. Measured MaxRSS was
# 26G for equalIndifference (8.3 GB log_lik, 3:57 elapsed); sheng2020 is
# ~1.2x larger, so ~31G. 64G leaves 2x headroom.
#SBATCH --mem=64G

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
  --pupil_col "$PUPIL_COL" \
  --results_pth "$RESULTS_DIR" \
  --p_outlier "$P_OUTLIER"
EOF

export REPO_PTH DATASET_NAME RESULTS_DIR HDDM_MODEL_CSV PYTHONPATH MODULE HDDM_SIF P_OUTLIER
echo "Submitting diagnostics array job..."
ARRAY_JOB_ID=$(sbatch --export=ALL --parsable "$TEMP_SBATCH")
echo "Array job ID: $ARRAY_JOB_ID"
echo "$ARRAY_JOB_ID"
