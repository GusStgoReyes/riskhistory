#!/bin/bash
# =============================================================================
# Posterior-predictive PPC-DDM job array — one job per (dataset, sub, model)
# triple
# =============================================================================

set -euo pipefail

MODELS="HDDM_Base,HDDM_Base_Lag1-stpdrift_Pupil,HDDM_Base_Lag1-stpdrift_Pupil_stpmain,HDDM_Base_Lag1-stpdrift_Pupil_stp,HDDM_Base_Lag1-stpdrift_Pupil_driftmain,HDDM_Base_Lag1-stpdrift_Pupil_drift,HDDM_Base_Lag1-stpdrift_Pupil_stpdrift"
NUM_DRAWS=1
MAX_CONCURRENT=250

REPO_PTH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_PTH/sbatch/configs/general.sh"

LOG_DIR="$REPO_PTH/logs/ppcddm_sampled"
mkdir -p "$LOG_DIR"

MODELS_FLAG=""
if [ -n "$MODELS" ]; then
  MODELS_FLAG="--models $MODELS"
fi

# Median runs (NUM_DRAWS=1) take ~25 s per model, so run every model in one
# job per (dataset, sub); full posterior runs get one job per model too.
if [ "$NUM_DRAWS" -eq 1 ]; then
  MODELS_FLAG="$MODELS_FLAG --per_subject"
fi

if [ ! -d "$RESULTS_DIR/hddm_posteriors" ]; then
  echo "ERROR: $RESULTS_DIR/hddm_posteriors not found -- check RESULTS_DIR in sbatch/configs/general.sh" >&2
  exit 1
fi

# Resolve number of (dataset, sub, model) jobs from the manifest -- must
# pass the same --models filter as the array itself, since job count scales
# with how many models are selected.
N_JOBS=$(cd "$REPO_PTH" && uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled --print_njobs $MODELS_FLAG)
echo "Submitting $N_JOBS jobs (max $MAX_CONCURRENT concurrent per array), models: ${MODELS:-all}, num_draws: $NUM_DRAWS..."
echo "Posteriors from: $RESULTS_DIR/hddm_posteriors"

# Array indices must be < MaxArraySize (1000 on Sherlock), so submit in
# chunks; each task runs manifest entry OFFSET + SLURM_ARRAY_TASK_ID.
MAX_ARRAY_SIZE=$(scontrol show config | awk '/^MaxArraySize/ {print $3}')
CHUNK=${MAX_ARRAY_SIZE:-1000}

for (( OFFSET = 0; OFFSET < N_JOBS; OFFSET += CHUNK )); do
  CHUNK_N=$(( N_JOBS - OFFSET < CHUNK ? N_JOBS - OFFSET : CHUNK ))
  echo "  jobs $OFFSET-$(( OFFSET + CHUNK_N - 1 ))"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ppcddm_sampled
#SBATCH --time=00:30:00
#SBATCH --partition=normal,russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=${LOG_DIR}/ppcddm_sampled_%A_%a.out
#SBATCH --error=${LOG_DIR}/ppcddm_sampled_%A_%a.err
#SBATCH --array=0-$(( CHUNK_N - 1 ))%${MAX_CONCURRENT}
#SBATCH --export=ALL

ml openblas/0.3.28
cd "$REPO_PTH"

uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled \\
    --job_index \$(( $OFFSET + SLURM_ARRAY_TASK_ID )) --num_draws $NUM_DRAWS $MODELS_FLAG \\
    --results_pth "$RESULTS_DIR"
EOF
done
