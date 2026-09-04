# pystanwasm

Run [Stan](https://mc-stan.org/) models from Python — inside
[Pyodide](https://pyodide.org/), the WebAssembly build of Python that runs
entirely in a browser. `pystanwasm` bridges Pyodide to
[`stanwasm`](https://github.com/habakan/stanwasm), a Rust→WebAssembly Stan
implementation published on npm, via Pyodide's JavaScript interop: two
independent wasm modules running side by side in the same browser tab.

**This only works inside Pyodide.** `import pystanwasm` is safe anywhere
(numpy/pandas are its only real dependencies), but calling `StanModel(...).sampling(...)`
needs the `js`/`pyodide_js` runtime modules a plain CPython interpreter
doesn't have.

```python
import pystanwasm as stan

model = stan.StanModel(stan_code)
fit = await model.sampling(data=data, iter=2000, warmup=1000, seed=42)
fit.summary()      # pandas DataFrame.describe() over the post-warmup draws
fit["beta"]        # one parameter's draws as a numpy array
```

`StanModel(code).sampling(...)` is styled after
[PyStan](https://pystan.readthedocs.io/)'s API for readability, but
`pystanwasm` is not affiliated with PyStan and is not a clone of it: no
diagnostics beyond what `stanwasm` itself returns, and every `sampling()`
call recompiles the model, since `stanwasm` binds Stan code and data
together at construction time rather than compiling once and rebinding data
per call. `sampling_parallel(n_chains=...)` runs multiple chains
concurrently, one per Web Worker, when nested-Worker support is available
(falls back to sequential otherwise) — see
[`src/pystanwasm/_bridge.py`](src/pystanwasm/_bridge.py) for that and the
full list of simplifications.

This is an early, experimental proof of concept — not on PyPI yet.

## Demos

**[habakan.github.io/pystanwasm](https://habakan.github.io/pystanwasm/)** —
a [JupyterLite](https://jupyterlite.readthedocs.io/) notebook (no server,
nothing installed) that installs `pystanwasm` via `micropip` from a locally
built wheel: linear regression, logistic regression, a hierarchical model
(eight schools), and `sampling_parallel` timed against sequential sampling.

**[habakan.github.io/pystanwasm/marimo](https://habakan.github.io/pystanwasm/marimo/)**
— the same bridge inside a [marimo](https://marimo.io/) WASM notebook.
marimo runs Pyodide in a dedicated Web Worker the same way JupyterLite does,
so nothing in `pystanwasm` itself is JupyterLite-specific; the one
host-dependent piece is finding the site's own base URL from inside that
worker (`self.location.origin` alone breaks under a project-site subpath —
see `_bridge.py`'s `_KNOWN_WORKER_MARKERS`), which is why both demos are
deployed at a real subpath (`/pystanwasm/` and `/pystanwasm/marimo/`)
rather than each getting its own site.

```bash
make setup           # npm install + a venv with jupyterlite-core, marimo, build, etc.
make jupyterlite      # build the pystanwasm wheel, then serve JupyterLite at http://127.0.0.1:8000
make marimo-build     # build the marimo demo into examples/marimo/_output
make pages-build      # both, combined into dist/ (JupyterLite at /, marimo at /marimo/) -- what CI ships
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
