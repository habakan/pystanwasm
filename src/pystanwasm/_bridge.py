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
import warnings

import numpy as np
import pandas as pd

# Where stanwasm's wasm-bindgen output is expected to be served from,
# relative to the page origin. Override via `StanModel(..., stanwasm_path=...)`
# if your deployment lays out static files differently.
DEFAULT_STANWASM_PATH = "/files/stanwasm/pkg/stanwasm.js"

_stanwasm_js_cache = {}
_nested_worker_support = None  # None = not probed yet; cached per session after that.

# A plain-JS worker body (no Pyodide) for `sampling_parallel`: one chain per
# instance. Sent as a Blob URL rather than a static file so there's no extra
# path to keep in sync with `stanwasm_path`. Each job carries its own
# `stanwasmUrl` (computed once by the caller — a nested worker shouldn't
# re-derive it via `self.location`, since that resolves to the *worker's*
# own URL, not the page's, one level up from where the caller already had
# to learn that the hard way) and constrains every draw itself, the same
# way `StanModel.sampling()` does, so results line up with `paramNames()`.
_WORKER_SRC = """
let modPromise;
self.onmessage = async (e) => {
  try {
    if (!modPromise) {
      modPromise = import(e.data.stanwasmUrl).then(async (m) => { await m.default(); return m; });
    }
    const mod = await modPromise;
    const model = new mod.StanModel(e.data.modelCode, e.data.dataJson);
    const names = model.paramNames();
    const nParams = model.n_params;
    const initArr = e.data.init ? Float64Array.from(e.data.init) : new Float64Array(nParams);
    const raw = model.sample(initArr, e.data.warmup, e.data.nDraws, BigInt(e.data.seed));
    const total = e.data.warmup + e.data.nDraws;
    const out = new Float64Array(total * names.length);
    for (let i = 0; i < total; i++) {
      const row = raw.slice(i * nParams, (i + 1) * nParams);
      out.set(model.constrainDraw(row), i * names.length);
    }
    self.postMessage({ ok: true, names, draws: out }, [out.buffer]);
  } catch (err) {
    self.postMessage({ ok: false, error: String((err && err.stack) || err) });
  }
};
"""


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


async def _nested_workers_supported():
    """Probe once per session whether this browser can run a Worker spawned
    from inside the Pyodide kernel's own Worker.

    A bare try/except isn't enough: some engines have historically let
    `new Worker(...)` construct successfully while the worker never
    actually starts, so this waits for a real round-trip message with a
    timeout rather than trusting construction alone.
    """
    global _nested_worker_support
    if _nested_worker_support is not None:
        return _nested_worker_support

    from pyodide.code import run_js

    probe_js = (
        "(async () => {"
        "  try {"
        "    const src = \"self.onmessage = () => self.postMessage('pong');\";"
        "    const blobUrl = URL.createObjectURL(new Blob([src], {type: 'application/javascript'}));"
        "    const ok = await new Promise((resolve) => {"
        "      let w;"
        "      try { w = new Worker(blobUrl, {type: 'module'}); }"
        "      catch { resolve(false); return; }"
        "      const t = setTimeout(() => { w.terminate(); resolve(false); }, 3000);"
        "      w.onmessage = () => { clearTimeout(t); w.terminate(); resolve(true); };"
        "      w.onerror = () => { clearTimeout(t); w.terminate(); resolve(false); };"
        "      w.postMessage('ping');"
        "    });"
        "    URL.revokeObjectURL(blobUrl);"
        "    return ok;"
        "  } catch { return false; }"
        "})()"
    )
    _nested_worker_support = bool(await run_js(probe_js))
    return _nested_worker_support


async def _sample_chains_via_workers(stanwasm_url, model_code, data, jobs):
    from pyodide.code import run_js

    data_json = json.dumps(data)
    job_payloads = [
        {
            "stanwasmUrl": stanwasm_url,
            "modelCode": model_code,
            "dataJson": data_json,
            "warmup": job["warmup"],
            "nDraws": job["nDraws"],
            "seed": job["seed"],
            "init": job["init"],
        }
        for job in jobs
    ]

    orchestrator_js = (
        "(async () => {"
        "  const jobs = " + json.dumps(job_payloads) + ";"
        "  const workerSrc = " + json.dumps(_WORKER_SRC) + ";"
        "  const blobUrl = URL.createObjectURL(new Blob([workerSrc], {type: 'application/javascript'}));"
        "  const workers = [];"
        "  try {"
        "    return await Promise.all(jobs.map((job) => new Promise((resolve, reject) => {"
        "      const w = new Worker(blobUrl, {type: 'module'});"
        "      workers.push(w);"
        "      w.onmessage = (e) => e.data.ok ? resolve(e.data) : reject(new Error(e.data.error));"
        "      w.onerror = (e) => reject(new Error(e.message || String(e)));"
        "      w.postMessage(job);"
        "    })));"
        "  } finally {"
        "    for (const w of workers) w.terminate();"
        "    URL.revokeObjectURL(blobUrl);"
        "  }"
        "})()"
    )
    return await run_js(orchestrator_js)


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

    async def sampling_parallel(self, data, n_chains=4, iter=2000, warmup=None, seeds=None, init=None):
        """Sample `n_chains` chains concurrently, one per Web Worker.

        Each worker independently loads stanwasm's wasm module and compiles
        the model — no Pyodide in the loop, no shared state between chains.
        Needs nested-Worker support (Safari 16.4+, current Chrome/Edge/
        Firefox); where that's unavailable, falls back to running the
        chains sequentially through `sampling()` and prints a one-time
        warning, so this never hard-fails just because of an old browser.
        Returns a plain `list[StanFit]` — no combined-array type, no R-hat
        or other convergence diagnostics; feed the per-chain draws to
        something like arviz if you want those.
        """
        if warmup is None:
            warmup = iter // 2
        n_draws = iter - warmup
        if seeds is None:
            seeds = [42 + i for i in range(n_chains)]

        if not await _nested_workers_supported():
            warnings.warn(
                "nested Web Workers unavailable in this browser; running "
                f"{len(seeds)} chains sequentially (no wall-clock speedup)",
                stacklevel=2,
            )
            return [
                await self.sampling(data, iter=iter, warmup=warmup, seed=s, init=init) for s in seeds
            ]

        from pyodide.code import run_js

        stanwasm_url = run_js("self.location.origin") + self.stanwasm_path
        jobs = [{"warmup": warmup, "nDraws": n_draws, "seed": s, "init": init} for s in seeds]
        results = await _sample_chains_via_workers(stanwasm_url, self.model_code, data, jobs)

        total = warmup + n_draws
        fits = []
        for r in results:
            names = list(r.names)
            draws = np.asarray(r.draws.to_py()).reshape((total, len(names)))
            fits.append(StanFit(names, draws, warmup))
        return fits
