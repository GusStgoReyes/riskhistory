#!/bin/bash
#SBATCH --job-name=run_lsa
#SBATCH --time=6:00:00
#SBATCH --partition=normal,russpold
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB

set -euo pipefail

source /scratch/users/gustxsr/cptpyddm/.venv/bin/activate

python /scratch/users/gustxsr/cptpyddm/src/cptpyddm/fmri/lsa_trial_beta_extraction.py "$1"
