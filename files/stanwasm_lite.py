"""A thin, PyStan-flavored wrapper around the stanwasm JS/wasm bridge.

Everything here is a convenience layer over `pyodide.code.run_js` and
`pyodide_js.registerJsModule` — actual parsing, autodiff and sampling still
happen entirely inside stanwasm's own wasm module (see the top-level
README's JS API). This does not replicate PyStan's feature set: one chain,
no diagnostics beyond what stanwasm itself returns, and every `sampling()`
call recompiles the model, because stanwasm binds Stan code and data
together at construction time rather than compiling code once and rebinding
data per call.
"""

import json

import numpy as np
import pandas as pd
from js import BigInt
from pyodide.code import run_js

_stanwasm_js = None


async def _load_stanwasm_js():
    global _stanwasm_js
    if _stanwasm_js is not None:
        return _stanwasm_js

    import pyodide_js

    base = run_js("self.location.origin") + "/files/stanwasm/pkg/stanwasm.js"
    js_src = (
        "(async () => {"
        "  const mod = await import(" + json.dumps(base) + ");"
        "  await mod.default();"
        "  return mod;"
        "})()"
    )
    mod = await run_js(js_src)
    pyodide_js.registerJsModule("stanwasm_js", mod)
    _stanwasm_js = mod
    return mod


class StanFit:
    """Draws from one `StanModel.sampling()` call.

    `draws` is unconstrained-space, shape (warmup + n_draws, n_params) — the
    same layout `StanModel.sample()` returns in JS. Values are on the scale
    `stanwasm` samples in (e.g. a `<lower=0>` parameter is log-scale here),
    not the constrained scale a Stan user normally expects.
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

    def __init__(self, model_code):
        self.model_code = model_code

    async def sampling(self, data, iter=2000, warmup=None, seed=42, init=None):
        if warmup is None:
            warmup = iter // 2
        n_draws = iter - warmup

        stanwasm_js = await _load_stanwasm_js()
        model = stanwasm_js.StanModel.new(self.model_code, json.dumps(data))

        if init is None:
            init = [0.0] * model.n_params
        samples = model.sample(init, warmup, n_draws, BigInt(seed))

        draws = np.asarray(samples.to_py()).reshape((-1, model.n_params))
        return StanFit(list(model.paramNames()), draws, warmup)
