#!/bin/bash
# =============================================================================
# Split-half DDM fits — one job per (dataset, sub, half_id)
#
# Usage (from repo root):
#   bash sbatch/run_ddm_splithalf.sh
# =============================================================================

set -euo pipefail

REPO_PTH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_PTH/sbatch/configs/general.sh"

LOG_DIR="$REPO_PTH/logs/ddm_splithalf"
mkdir -p "$LOG_DIR"

N_JOBS=$(cd "$REPO_PTH" && uv run python -m riskhistory.analysis_splithalf.2_fitddm_splithalf --print_njobs)
echo "Submitting array of $N_JOBS jobs (max 100 concurrent)..."

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=ddm_splithalf
#SBATCH --time=02:00:00
#SBATCH --partition=normal,russpold
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=${LOG_DIR}/ddm_splithalf_%A_%a.out
#SBATCH --error=${LOG_DIR}/ddm_splithalf_%A_%a.err
#SBATCH --array=0-$((N_JOBS - 1))%100
#SBATCH --export=ALL

ml openblas/0.3.28
cd /scratch/users/gustxsr/riskhistory

uv run python -m riskhistory.analysis_splithalf.2_fitddm_splithalf \\
    --job_index \$SLURM_ARRAY_TASK_ID
EOT
