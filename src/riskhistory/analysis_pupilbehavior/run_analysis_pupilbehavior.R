# =============================================================================
# Choice History Under Risk: Run Pupil-Behavior Analysis
# =============================================================================
# Run from this directory:
#   cd src/riskhistory/analysis_pupilbehavior
#   Rscript run_analysis_pupilbehavior.R
# =============================================================================

source("0_config.R")
source("0_choicehistoryreplication.R")
source("2_pupilbins_models.R")
source("2b_pupilbins_simpleeffects.R")
source("3_pupilxchoicehistory_model.R")