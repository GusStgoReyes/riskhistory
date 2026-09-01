# =============================================================================
# Shared analysis settings for R pipelines in src/riskhistory/analysis_*
# =============================================================================
# Assumes library(lme4) has already been loaded by the caller (glmerControl()/
# lmerControl() come from lme4).

set.seed(42)

datasets <- c("equalIndifference", "equalRange", "sheng2020")

p_method <- "BH"  # Benjamini-Hochberg

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
LMER_CONTROL  <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
