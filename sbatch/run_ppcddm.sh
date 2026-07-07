#!/bin/bash
# =============================================================================
# PPC-DDM job array — one job per (dataset, sub) pair
#
# Usage (from repo root):
#   bash sbatch/run_ppcddm.sh
#
# This wrapper:
#   1. Queries the Python script for the exact number of jobs.
#   2. Submits a SLURM array of that size (max 20 concurrent).
# =============================================================================

set -euo pipefail

REPO_PTH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_PTH/sbatch/configs/general.sh"

LOG_DIR="$REPO_PTH/logs/ppcddm"
mkdir -p "$LOG_DIR"

# Resolve number of jobs from the manifest
N_JOBS=$(cd "$REPO_PTH" && uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm --print_njobs)
echo "Submitting array of $N_JOBS jobs (max 20 concurrent)..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ppcddm
#SBATCH --time=00:15:00
#SBATCH --partition=normal,russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=${LOG_DIR}/ppcddm_%A_%a.out
#SBATCH --error=${LOG_DIR}/ppcddm_%A_%a.err
#SBATCH --array=0-$((N_JOBS - 1))%20
#SBATCH --export=ALL

ml openblas/0.3.28
cd /scratch/users/gustxsr/riskhistory

uv run python -m riskhistory.analysis_ppcddm.run_analysis_ppcddm \\
    --job_index \$SLURM_ARRAY_TASK_ID
EOF
