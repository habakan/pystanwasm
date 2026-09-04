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
— the same four demos, one [marimo](https://marimo.io/) WASM notebook each
(marimo exports one static bundle per notebook, so each gets its own
`/marimo/<slug>/`). marimo runs Pyodide in a dedicated Web Worker the same
way JupyterLite does, so nothing in `pystanwasm` itself is JupyterLite-
specific; the one host-dependent piece is finding the site's own base URL
from inside that worker (`self.location.origin` alone breaks under a
project-site subpath — see `_bridge.py`'s `_KNOWN_WORKER_MARKERS`), which is
why all demos below are deployed at a real subpath rather than each getting
its own site.

**[habakan.github.io/pystanwasm/pyscript](https://habakan.github.io/pystanwasm/pyscript/)**
— the same four demos again, this time as plain
[PyScript](https://pyscript.net/) pages: no notebook kernel at all, just
`<script type="py">` in HTML. PyScript's Worker mode needs
cross-origin-isolation headers a static Pages site doesn't set, so these run
on the main thread — `self.location` is just the real page location there,
no site-root detection needed, and the Workers `sampling_parallel` spawns
aren't nested inside another Worker (an easier case than JupyterLite/
marimo). Building this demo out surfaced two real portability bugs in
`_bridge.py`, both fixed there rather than worked around in the page:
`pyodide_js.registerJsModule()` (leftover from an earlier design, unused,
and outright broken under PyScript's Pyodide build) and `js.BigInt(<python
int>)` (silently version-fragile — fixed by building the whole
compile-and-sample call as one pure-JS `run_js` snippet instead, the same
pattern `sampling_parallel` already used).

**[habakan.github.io/pystanwasm/quarto](https://habakan.github.io/pystanwasm/quarto/)**
— the same four demos as [Quarto Live](https://r-wasm.github.io/quarto-live/)
`{pyodide}` documents: a rendered page with an interactive Python block
embedded in it, Pyodide running in its own Web Worker again (same
subpath-detection story as marimo, though the worker URL shape here needs its
own regex — the extension nests a per-document `_files/` directory in the
path). Each demo is a single `{pyodide}` block rather than several
cooperating ones; splitting the state that `pystanwasm` builds up (a
`StanModel`/`StanFit` holding live JS proxies) across Quarto Live's
otherwise-shared-namespace blocks breaks with `TypeError: unhashable type:
'pyodide.ffi.JsProxy'`.

`make marimo-build`/`make pages-build` need [`uv`](https://github.com/astral-sh/uv)
on `PATH` — `marimo export html-wasm` shells out to it to resolve the
notebook's imports. `make quarto-build`/`make pages-build` also need the
[Quarto](https://quarto.org/) CLI on `PATH`.

```bash
make setup           # npm install + a venv with jupyterlite-core, marimo, build, etc.
make jupyterlite      # build the pystanwasm wheel, then serve JupyterLite at http://127.0.0.1:8000
make marimo-build     # build the 4 marimo demos into examples/marimo/_output
make pyscript-build   # stage stanwasm + the wheel next to examples/pyscript/'s HTML pages
make quarto-build     # render the 4 Quarto Live demos in place in examples/quarto/
make pages-build      # all four, combined into dist/ (JupyterLite /, marimo /marimo/, PyScript /pyscript/, Quarto Live /quarto/) -- what CI ships
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
