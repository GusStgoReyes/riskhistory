# =============================================================================
# Choice History Under Risk: Run the Split-Half Analysis
# =============================================================================
# Need to run 1_fitprospecttheory_splithalf.py first
# (2_fitddm_splithalf.py runs on Sherlock via sbatch/run_ddm_splithalf.sh)
source("0_config.R")
source("3_rtswitch_splithalf.R")

message("\nSplit-half analysis complete.")
