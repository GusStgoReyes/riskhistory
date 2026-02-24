# Common paths (the REPO_PTH variable should be set in the sbatch script)
DATASET_DIR="$REPO_PTH/data"
BEHAV_CSV="$DATASET_DIR/preprocessed/behavioral_data.csv"
RESULTS_DIR="$REPO_PTH/results/analysis_ddm"

# PyDDM Models
MODEL_CSV="$DATASET_DIR/model_ids/pyddm.csv"
# HDDM Models
HDDM_MODEL_CSV="$DATASET_DIR/model_ids/hddm.csv"

# Sbatch settings pyddm
TIME="02:00:00"
MEM="4G"
N_CPUS=8

# Sbatch setting hddm
TIME_HDDM="20:00:00"
MEM_HDDM="16G"
N_CPUS_HDDM=1
