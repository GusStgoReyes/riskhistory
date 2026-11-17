#!/bin/bash
# =============================================================================
# Split-half DDM fits — one job per (dataset, sub, half_id)
#
# Usage (from repo root):
#   bash sbatch/run_ddm_splithalf.sh
#
# JOB_INDICES below reruns only those array indices (comma-separated); leave
# empty ("") to submit every job in the manifest.
# =============================================================================

set -euo pipefail

JOB_INDICES="7,33,40,58,63,92,98,212,215"
TIME_LIMIT="06:00:00"

REPO_PTH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_PTH/sbatch/configs/general.sh"

LOG_DIR="$REPO_PTH/logs/ddm_splithalf"
mkdir -p "$LOG_DIR"

if [ -n "$JOB_INDICES" ]; then
  ARRAY="$JOB_INDICES"
else
  N_JOBS=$(cd "$REPO_PTH" && uv run python -m riskhistory.analysis_splithalf.2_fitddm_splithalf --print_njobs)
  ARRAY="0-$((N_JOBS - 1))"
fi
echo "Submitting array $ARRAY (max 100 concurrent, time limit $TIME_LIMIT)..."

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=ddm_splithalf
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=normal,russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=${LOG_DIR}/ddm_splithalf_%A_%a.out
#SBATCH --error=${LOG_DIR}/ddm_splithalf_%A_%a.err
#SBATCH --array=${ARRAY}%100
#SBATCH --export=ALL

ml openblas/0.3.28
cd /scratch/users/gustxsr/riskhistory

uv run python -m riskhistory.analysis_splithalf.2_fitddm_splithalf \\
    --job_index \$SLURM_ARRAY_TASK_ID
EOT
