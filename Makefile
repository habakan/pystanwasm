VENV := $(CURDIR)/.venv
MARIMO_DIR := examples/marimo
MARIMO_OUT := $(MARIMO_DIR)/_output
MARIMO_NOTEBOOKS := linear_regression logistic_regression eight_schools parallel_chains

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

# Each of the 4 demo notebooks is its own marimo WASM export (one HTML
# bundle per notebook, marimo's model), landing at /marimo/<slug>/; stanwasm
# assets + the wheel are duplicated into each one since every export is a
# self-contained static bundle. index.html ties them together at /marimo/.
.PHONY: marimo-build
marimo-build: build-wheel copy-stanwasm ## Production build of the marimo demos, into examples/marimo/_output
	rm -rf $(MARIMO_OUT)
	mkdir -p $(MARIMO_OUT)
	for nb in $(MARIMO_NOTEBOOKS); do \
	  slug=$$(echo $$nb | tr '_' '-'); \
	  $(VENV)/bin/marimo export html-wasm $(MARIMO_DIR)/$$nb.py -o $(MARIMO_OUT)/$$slug --mode run -f; \
	  cp -R examples/jupyterlite/files/stanwasm $(MARIMO_OUT)/$$slug/stanwasm; \
	  cp examples/jupyterlite/files/pystanwasm-*.whl $(MARIMO_OUT)/$$slug/; \
	done
	cp $(MARIMO_DIR)/index.html $(MARIMO_OUT)/index.html

# PyScript demos are plain static HTML, no export/build step -- just need
# stanwasm's assets + the wheel sitting next to them.
.PHONY: pyscript-build
pyscript-build: build-wheel copy-stanwasm ## Stage stanwasm + the wheel next to examples/pyscript/'s HTML pages
	rm -rf examples/pyscript/stanwasm
	cp -R examples/jupyterlite/files/stanwasm examples/pyscript/stanwasm
	cp examples/jupyterlite/files/pystanwasm-*.whl examples/pyscript/

.PHONY: pages-build
pages-build: jupyterlite-build marimo-build pyscript-build ## Combined static site for GitHub Pages: JupyterLite at /, marimo at /marimo/, PyScript at /pyscript/
	rm -rf dist
	mkdir -p dist/marimo dist/pyscript
	cp -R examples/jupyterlite/_output/. dist/
	cp -R $(MARIMO_OUT)/. dist/marimo/
	cp -R examples/pyscript/. dist/pyscript/
