VENV := $(CURDIR)/.venv

.PHONY: setup
setup: ## Install npm + Python dependencies (run once)
	cd examples/jupyterlite && npm install
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[examples]"

.PHONY: build-wheel
build-wheel: ## Build the pystanwasm wheel into examples/jupyterlite/files/
	rm -f examples/jupyterlite/files/pystanwasm-*.whl
	$(VENV)/bin/python -m build --wheel --outdir examples/jupyterlite/files .

.PHONY: jupyterlite
jupyterlite: build-wheel ## Local JupyterLite dev server with the pystanwasm demo notebook
	cd examples/jupyterlite && bash scripts/copy-stanwasm.sh
	cd examples/jupyterlite && $(VENV)/bin/jupyter lite serve

.PHONY: jupyterlite-build
jupyterlite-build: build-wheel ## Production build, into examples/jupyterlite/_output
	cd examples/jupyterlite && bash scripts/copy-stanwasm.sh
	cd examples/jupyterlite && $(VENV)/bin/jupyter lite build
