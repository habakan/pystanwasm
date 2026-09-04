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
multi-chain support, no diagnostics beyond what `stanwasm` itself returns,
and every `sampling()` call recompiles the model, since `stanwasm` binds
Stan code and data together at construction time rather than compiling once
and rebinding data per call. See [`src/pystanwasm/_bridge.py`](src/pystanwasm/_bridge.py)
for the full list of simplifications.

This is an early, experimental proof of concept — not on PyPI yet.

## Demo

[`examples/jupyterlite`](examples/jupyterlite) is a
[JupyterLite](https://jupyterlite.readthedocs.io/) notebook — no server,
nothing installed — that installs `pystanwasm` via `micropip` from a locally
built wheel and runs a small linear regression end to end.

```bash
make setup            # npm install + a venv with jupyterlite-core, build, etc.
make jupyterlite       # build the pystanwasm wheel, then serve at http://127.0.0.1:8000
```

`make jupyterlite-build` produces the same static site under
`examples/jupyterlite/_output/` without serving it.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
