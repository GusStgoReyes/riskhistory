#!/bin/bash
# =============================================================================
# Posterior-predictive PPC-DDM job array — one job per (dataset, sub, model)
# triple
#
# Usage (from repo root):
#   bash sbatch/run_ppcddm_sampled.sh
#
# Unlike run_ppcddm.sh (point-estimate simulation, one job per (dataset,
# sub) looping over every model), this runs run_analysis_ppcddm_sampled.py
# with the array split on model too: at NUM_DRAWS=1000, one (dataset, sub,
# model) posterior-predictive simulation measured ~6.5h on its own, so
# looping over multiple models serially in one job (as run_ppcddm.sh does)
# would mean a single array task taking >30h. --time below is set from that
# measured 6.5h plus margin -- re-measure with a pilot run if NUM_DRAWS or
# the model set changes substantially:
#   uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled \
#       --job_index 0 --num_draws 20 --models HDDM_Base
#
# MODELS below controls which model_id(s) get simulated (comma-separated,
# passed through to --models). Leave empty ("") to simulate every model
# currently in run_analysis_ppcddm.py's MODELS dict.
#
# CONCURRENCY: the array is now (n_dataset_sub_pairs * n_models) jobs --
# e.g. 190 subs * 5 models = 950 -- specifically so more of them can run at
# once. The %<N> below is a placeholder; raise it to whatever your
# partition/QOS actually allows concurrently, or this refactor doesn't buy
# you anything.
# =============================================================================

set -euo pipefail

MODELS=""
NUM_DRAWS=1000
MAX_CONCURRENT=250

REPO_PTH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_PTH/sbatch/configs/general.sh"

LOG_DIR="$REPO_PTH/logs/ppcddm_sampled"
mkdir -p "$LOG_DIR"

MODELS_FLAG=""
if [ -n "$MODELS" ]; then
  MODELS_FLAG="--models $MODELS"
fi

# Resolve number of (dataset, sub, model) jobs from the manifest -- must
# pass the same --models filter as the array itself, since job count scales
# with how many models are selected.
N_JOBS=$(cd "$REPO_PTH" && uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled --print_njobs $MODELS_FLAG)
echo "Submitting array of $N_JOBS jobs (max $MAX_CONCURRENT concurrent), models: ${MODELS:-all}, num_draws: $NUM_DRAWS..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ppcddm_sampled
#SBATCH --time=10:00:00
#SBATCH --partition=normal,russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=${LOG_DIR}/ppcddm_sampled_%A_%a.out
#SBATCH --error=${LOG_DIR}/ppcddm_sampled_%A_%a.err
#SBATCH --array=0-$((N_JOBS - 1))%${MAX_CONCURRENT}
#SBATCH --export=ALL

ml openblas/0.3.28
cd /scratch/users/gustxsr/riskhistory

uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled \\
    --job_index \$SLURM_ARRAY_TASK_ID --num_draws $NUM_DRAWS $MODELS_FLAG
EOF
