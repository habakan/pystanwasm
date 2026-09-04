VENV := $(CURDIR)/.venv
MARIMO_NB := examples/marimo/pystanwasm_demo.py
MARIMO_OUT := examples/marimo/_output

.PHONY: setup
setup: ## Install npm + Python dependencies (run once)
	cd examples/jupyterlite && npm install
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[examples]"

.PHONY: build-wheel
build-wheel: ## Build the pystanwasm wheel into examples/jupyterlite/files/
	rm -f examples/jupyterlite/files/pystanwasm-*.whl
	$(VENV)/bin/python -m build --wheel --outdir examples/jupyterlite/files .

.PHONY: copy-stanwasm
copy-stanwasm: ## Copy stanwasm's npm-built assets into examples/jupyterlite/files/stanwasm
	cd examples/jupyterlite && bash scripts/copy-stanwasm.sh

.PHONY: jupyterlite
jupyterlite: build-wheel copy-stanwasm ## Local JupyterLite dev server with the pystanwasm demo notebooks
	cd examples/jupyterlite && $(VENV)/bin/jupyter lite serve

.PHONY: jupyterlite-build
jupyterlite-build: build-wheel copy-stanwasm ## Production build, into examples/jupyterlite/_output
	cd examples/jupyterlite && $(VENV)/bin/jupyter lite build

.PHONY: marimo-build
marimo-build: build-wheel copy-stanwasm ## Production build of the marimo demo, into examples/marimo/_output
	rm -rf $(MARIMO_OUT)
	$(VENV)/bin/marimo export html-wasm $(MARIMO_NB) -o $(MARIMO_OUT) --mode run -f
	cp -R examples/jupyterlite/files/stanwasm $(MARIMO_OUT)/stanwasm
	cp examples/jupyterlite/files/pystanwasm-*.whl $(MARIMO_OUT)/

.PHONY: pages-build
pages-build: jupyterlite-build marimo-build ## Combined static site for GitHub Pages: JupyterLite at /, marimo at /marimo/
	rm -rf dist
	mkdir -p dist/marimo
	cp -R examples/jupyterlite/_output/. dist/
	cp -R $(MARIMO_OUT)/. dist/marimo/
