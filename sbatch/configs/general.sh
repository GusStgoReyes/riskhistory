# Common paths (the REPO_PTH variable should be set in the sbatch script)
DATASET_DIR="$REPO_PTH/data"
BEHAV_CSV="$DATASET_DIR/preprocessed/behavioral_data.csv"
PUPIL_CSV="$DATASET_DIR/preprocessed/pupilbetas_glm_preprocessed.csv"
RESULTS_DIR="$REPO_PTH/results/analysis_ddm_final_pupil"

# PyDDM Models
MODEL_CSV="$DATASET_DIR/model_ids/pyddm.csv"
# HDDM Models
HDDM_MODEL_CSV="$DATASET_DIR/model_ids/hddm.csv"

# Singularity image for HDDM jobs
HDDM_SIF="/home/groups/russpold/singularity_images/hddm_latest.sif"

# Sbatch settings pyddm
TIME="02:00:00"
MEM="4G"
N_CPUS=8

# Sbatch setting hddm
# MEM_HDDM sized for the worst-case model (HDDM_Base_Lag1-full_sheng2020)
TIME_HDDM="36:00:00"
MEM_HDDM="190G"
N_CPUS_HDDM=25

# Must match the p_outlier passed to HDDMnnRegressor/HDDMRegressor
P_OUTLIER=0.05
