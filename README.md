# stanwasm-jupyterlite

Run [Stan](https://mc-stan.org/) models from Python cells in a
[JupyterLite](https://jupyterlite.readthedocs.io/) notebook — a fully
client-side Jupyter distribution whose kernel is
[Pyodide](https://pyodide.org/) (Python compiled to WebAssembly, running in a
browser Web Worker). No server, nothing installed, nothing to run.

Compilation and sampling happen in [`stanwasm`](https://github.com/habakan/stanwasm),
a Rust→WebAssembly Stan implementation published on npm. Two independent wasm
modules run side by side in the browser tab — Pyodide's own runtime and
stanwasm's — bridged through Pyodide's JavaScript interop: dynamically
importing stanwasm's ES module (the same mechanism JupyterLite's own kernel
uses to load its own runtime) and registering it as an importable Python
module via `pyodide_js.registerJsModule`.

[`files/stanwasm_lite.py`](files/stanwasm_lite.py) hides that bridge behind
a small, [PyStan](https://pystan.readthedocs.io/)-flavored API —
`StanModel(code).sampling(data=..., iter=..., warmup=..., seed=...)`
returning a `StanFit` — for readability in the notebook. It is not affiliated
with PyStan and is not a clone of it: no multi-chain support, no diagnostics
beyond what stanwasm itself returns, and every `sampling()` call recompiles
the model, since stanwasm binds Stan code and data together at construction
time rather than compiling once and rebinding data per call. See the
module's docstring for the full list of simplifications, and
[`files/Stan-in-Pyodide.ipynb`](files/Stan-in-Pyodide.ipynb) for the
notebook that uses it end to end.

This is an early, experimental proof of concept.

## Running locally

```bash
make setup   # npm install + a Python venv with jupyterlite-core
make jupyterlite   # build + serve at http://127.0.0.1:8000
```

`make jupyterlite-build` produces a static site under `_output/` without
serving it — the same tree a static host (e.g. GitHub Pages) would need.

`scripts/copy-stanwasm.sh` copies `node_modules/stanwasm/*` into
`files/stanwasm/` before each build; that copy is generated, not checked in.
Bumping the `stanwasm` version in `package.json` and re-running `npm install`
picks up a new stanwasm release with no other changes needed here.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
