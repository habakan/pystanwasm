"""JS interop bridge between Pyodide and the stanwasm wasm module.

Everything here is a convenience layer over Pyodide's JavaScript interop;
actual parsing, autodiff and sampling still happen entirely inside
stanwasm's own wasm module (see stanwasm's README for its JS API). This
does not replicate PyStan's feature set: one chain, no diagnostics beyond
what stanwasm itself returns, and every `sampling()` call recompiles the
model, because stanwasm binds Stan code and data together at construction
time rather than compiling code once and rebinding data per call.

`js`, `pyodide_js` and `pyodide.code.run_js` are synthetic modules a
Pyodide runtime provides, not real pip packages — they're only importable
inside Pyodide, so they're imported lazily inside the functions that need
them rather than at module import time. That keeps a plain `import
pystanwasm` from failing outside Pyodide; only calling into stanwasm does.
"""

import json

import numpy as np
import pandas as pd

# Where stanwasm's wasm-bindgen output is expected to be served from,
# relative to the page origin. Override via `StanModel(..., stanwasm_path=...)`
# if your deployment lays out static files differently.
DEFAULT_STANWASM_PATH = "/files/stanwasm/pkg/stanwasm.js"

_stanwasm_js_cache = {}


async def _load_stanwasm_js(path):
    if path in _stanwasm_js_cache:
        return _stanwasm_js_cache[path]

    import pyodide_js
    from pyodide.code import run_js

    base = run_js("self.location.origin") + path
    js_src = (
        "(async () => {"
        "  const mod = await import(" + json.dumps(base) + ");"
        "  await mod.default();"
        "  return mod;"
        "})()"
    )
    mod = await run_js(js_src)
    pyodide_js.registerJsModule("stanwasm_js", mod)
    _stanwasm_js_cache[path] = mod
    return mod


class StanFit:
    """Draws from one `StanModel.sampling()` call.

    `draws` is constrained-space, shape (warmup + n_draws, len(param_names))
    — `param_names` covers `parameters` *and* `transformed parameters`
    (stanwasm's `paramNames()` order), so a `<lower=0>` parameter is already
    on its natural scale and any `transformed parameters` quantity is
    included, not just the raw sampled parameters. Built by running each
    raw unconstrained draw stanwasm's `sample()` returns through
    `StanModel.constrainDraw()`.
    """

    def __init__(self, param_names, draws, warmup):
        self.param_names = list(param_names)
        self._draws = draws
        self.warmup = warmup

    def to_dataframe(self, include_warmup=False):
        draws = self._draws if include_warmup else self._draws[self.warmup :]
        return pd.DataFrame(draws, columns=self.param_names)

    def __getitem__(self, name):
        return self._draws[self.warmup :, self.param_names.index(name)]

    def summary(self):
        return self.to_dataframe().describe()


class StanModel:
    """A Stan program, compiled against a dataset on each `sampling()` call."""

    def __init__(self, model_code, stanwasm_path=DEFAULT_STANWASM_PATH):
        self.model_code = model_code
        self.stanwasm_path = stanwasm_path

    async def sampling(self, data, iter=2000, warmup=None, seed=42, init=None):
        if warmup is None:
            warmup = iter // 2
        n_draws = iter - warmup

        stanwasm_js = await _load_stanwasm_js(self.stanwasm_path)
        model = stanwasm_js.StanModel.new(self.model_code, json.dumps(data))

        from js import BigInt

        if init is None:
            init = [0.0] * model.n_params
        raw = model.sample(init, warmup, n_draws, BigInt(seed))

        names = list(model.paramNames())
        total = warmup + n_draws
        raw_np = np.asarray(raw.to_py()).reshape((total, model.n_params))
        # `sample()` is unconstrained-parameters-only; `paramNames()` covers
        # parameters + transformed parameters, so each row needs constraining
        # (and expanding) via `constrainDraw()` before it lines up with `names`.
        draws = np.empty((total, len(names)))
        for i in range(total):
            # constrainDraw expects a real JS Float64Array-compatible
            # array-like; a bare Pyodide-wrapped numpy row isn't one, so
            # convert to a plain Python list first (same as `init` above).
            draws[i] = model.constrainDraw(raw_np[i].tolist()).to_py()

        return StanFit(names, draws, warmup)
