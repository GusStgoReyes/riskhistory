#!/bin/bash
set -euo pipefail

# ======================
# User settings
# ======================
REPO_PTH=/scratch/users/gustxsr/riskhistory
DATASET_NAME=${1:-equalIndifference}
TEST_MODE=${2:-0}
PYTHONPATH="$REPO_PTH/src"
MODULE="riskhistory.analysis_ddm.model_hddm"

source "$REPO_PTH/sbatch/configs/general.sh"

# Each row is "model_id,pupil_col" -- pupil_col is only meaningful for models
# whose formula references "pupil" (Pupil_stp/_drift/_full); it's ignored
# otherwise.
mapfile -t models   < <(tail -n +2 "$HDDM_MODEL_CSV")

N_MODELS=${#models[@]}
ARRAY_RANGE="1-${N_MODELS}"

if [[ "$TEST_MODE" -eq 1 ]]; then
  TIME_HDDM="00:30:00"
  MEM_HDDM="10G"
  N_CPUS_HDDM=4
  TEST_FLAG="--test"
else
  TEST_FLAG=""
fi

echo "-------------------------------------------------"
echo " Dataset:     $DATASET_NAME"
echo " N_MODELS:    $N_MODELS"
echo " Time:        $TIME_HDDM"
echo " CPUs:        $N_CPUS_HDDM"
echo " Array range: $ARRAY_RANGE"
echo " Memory:      $MEM_HDDM"
echo "-------------------------------------------------"

# ======================
# Create temporary sbatch file
# ======================
TEMP_SBATCH=$(mktemp)
# --- 1. SBATCH header (expand outer vars now) ---
cat > "$TEMP_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=fit_hddm
#SBATCH --time=${TIME_HDDM}
#SBATCH --partition=normal,russpold
#SBATCH --array=${ARRAY_RANGE}
#SBATCH --cpus-per-task=${N_CPUS_HDDM}
#SBATCH --mem=${MEM_HDDM}

set -euo pipefail
EOF

# --- 2. Job body (expand inner vars only at runtime) ---
cat >> "$TEMP_SBATCH" <<'EOF'

# Recompute models
mapfile -t models  < <(tail -n +2 "$HDDM_MODEL_CSV")

N_MODELS=${#models[@]}
TOTAL=$((N_MODELS))

IDX=$((SLURM_ARRAY_TASK_ID - 1))

IFS=',' read -r MODEL_ID PUPIL_COL <<< "${models[$IDX]}"

echo "Running model: $MODEL_ID (pupil_col=$PUPIL_COL)"

cd $REPO_PTH

singularity exec \
  --env "PYTHONPATH=$REPO_PTH/src" \
  --env "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1" \
  "$HDDM_SIF" \
  python -m riskhistory.analysis_ddm.model_hddm \
  --model_id "$MODEL_ID" \
  --dataset_name "$DATASET_NAME" \
  --data_pth "$BEHAV_CSV" \
  --pupil_pth "$PUPIL_CSV" \
  --pupil_col "$PUPIL_COL" \
  --results_pth "$RESULTS_DIR" \
  --p_outlier "$P_OUTLIER" \
  $TEST_FLAG
EOF

# ======================
# Submit and clean up
# ======================
export REPO_PTH DATASET_NAME TEST_FLAG BEHAV_CSV PUPIL_CSV RESULTS_DIR HDDM_MODEL_CSV PYTHONPATH MODULE HDDM_SIF P_OUTLIER
echo "Submitting array job..."
sbatch --export=ALL "$TEMP_SBATCH"
