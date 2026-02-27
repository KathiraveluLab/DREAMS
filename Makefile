
# One-command shortcuts for common workflows.

DATASET ?= clinical_depression_study/dataset.csv

.PHONY: install pipeline pipeline-d1 pipeline-resume verify manifest clean-db help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

pipeline: install ## Run full pipeline from CSV dataset
	python pipeline/run_pipeline.py $(DATASET)

pipeline-d1: ## Pull from Cloudflare D1 and run full pipeline
	python pipeline/run_pipeline.py --source d1

pipeline-resume: ## Resume an interrupted pipeline run
	python pipeline/run_pipeline.py $(DATASET) --resume

verify: ## Run only the verification step
	python pipeline/run_pipeline.py $(DATASET) --only verify

manifest: ## Generate manifest report (add EXPORT=1 to export parquet)
	python pipeline/run_pipeline.py $(DATASET) --only manifest $(if $(EXPORT),--export,)

clean-db: ## Remove dreams.db and chroma_db (fresh start)
	rm -rf data/processed/dreams.db data/processed/chroma_db
	@echo "Cleaned. Run 'make pipeline' to rebuild."
