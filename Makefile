.PHONY: setup
setup: ## Install npm + Python dependencies (run once)
	npm install
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

.PHONY: jupyterlite
jupyterlite: ## Local JupyterLite dev server with the stanwasm bridge notebook
	bash scripts/copy-stanwasm.sh
	.venv/bin/jupyter lite serve

.PHONY: jupyterlite-build
jupyterlite-build: ## Production build, into _output/
	bash scripts/copy-stanwasm.sh
	.venv/bin/jupyter lite build
