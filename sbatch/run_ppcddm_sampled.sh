#!/bin/bash
# =============================================================================
# Posterior-predictive PPC-DDM job array — one job per (dataset, sub) pair
#
# Usage (from repo root):
#   bash sbatch/run_ppcddm_sampled.sh
#
# Unlike run_ppcddm.sh (point-estimate simulation), this runs
# run_analysis_ppcddm_sampled.py: NUM_DRAWS posterior parameter draws per
# (dataset, sub, model), each a full simulation, averaged into the same
# per-trial statistics. That means ~NUM_DRAWS times more model.solve() calls
# per job than run_ppcddm.sh -- --time below is a starting guess, not a
# measured budget. Do a pilot run first:
#   uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled \
#       --job_index 0 --num_draws 20 --models HDDM_Base
# and scale the observed wall time up to NUM_DRAWS before trusting --time
# for the real array.
#
# MODELS below controls which model_id(s) get simulated (comma-separated,
# passed through to --models). Leave empty ("") to simulate every model
# currently in run_analysis_ppcddm.py's MODELS dict.
# =============================================================================

set -euo pipefail

MODELS=""
NUM_DRAWS=1000

REPO_PTH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_PTH/sbatch/configs/general.sh"

LOG_DIR="$REPO_PTH/logs/ppcddm_sampled"
mkdir -p "$LOG_DIR"

MODELS_FLAG=""
if [ -n "$MODELS" ]; then
  MODELS_FLAG="--models $MODELS"
fi

# Resolve number of jobs from the manifest (same manifest as run_ppcddm.sh)
N_JOBS=$(cd "$REPO_PTH" && uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled --print_njobs)
echo "Submitting array of $N_JOBS jobs (max 20 concurrent), models: ${MODELS:-all}, num_draws: $NUM_DRAWS..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ppcddm_sampled
#SBATCH --time=12:00:00
#SBATCH --partition=normal,russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=${LOG_DIR}/ppcddm_sampled_%A_%a.out
#SBATCH --error=${LOG_DIR}/ppcddm_sampled_%A_%a.err
#SBATCH --array=0-$((N_JOBS - 1))%20
#SBATCH --export=ALL

ml openblas/0.3.28
cd /scratch/users/gustxsr/riskhistory

uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm_sampled \\
    --job_index \$SLURM_ARRAY_TASK_ID --num_draws $NUM_DRAWS $MODELS_FLAG
EOF
