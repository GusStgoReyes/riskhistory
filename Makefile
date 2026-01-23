#####################################
# Configuration
#####################################

PYTHON := python
RSCRIPT := Rscript

SRC := src/riskhistory
RESULTS := results
FIGURES := figures


#####################################
# Meta targets
#####################################

.PHONY: all clean help analyses

all: help

help:
	@echo "Available analyses:"
	@echo "  make setup_behavioral    Setup behavioral data (Python)"
	@echo "  make analysis_acceptglme Run accept/reject GLME analysis (R)"
	@echo "  make all_analyses        Run all analyses"
	@echo ""
	@echo "Utility:"
	@echo "  make clean               Remove generated results"

analyses:
	@ls $(SRC)


#####################################
# Individual analyses
#####################################

setup_behavioral: data/preprocessed
	uv run $(PYTHON) $(SRC)/setup_behavioraldata/run_setup_behavioraldata.py

setup_pupildata: data/preprocessed
	cd $(SRC)/setup_pupildata_v2 && $(RSCRIPT) run_setup_pupildata.R

analysis_acceptglme: $(RESULTS)/analysis_acceptglme
	cd $(SRC)/analysis_acceptglme && $(RSCRIPT) run_analysis_acceptglme.R


#####################################
# Grouped runs
#####################################

all_analyses: setup_behavioral analysis_acceptglme


#####################################
# Directories
#####################################

data/preprocessed:
	mkdir -p $@

$(RESULTS)/analysis_acceptglme:
	mkdir -p $@


#####################################
# Cleanup
#####################################

clean:
	rm -rf $(RESULTS)/*
	rm -rf $(FIGURES)/*
