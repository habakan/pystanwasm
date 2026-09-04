import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # pystanwasm in marimo

        [`pystanwasm`](https://github.com/habakan/pystanwasm) bridges Pyodide to
        [`stanwasm`](https://github.com/habakan/stanwasm)'s WebAssembly module the same way in
        marimo's WASM notebooks as it does in JupyterLite — both run Pyodide inside a dedicated
        Web Worker, and the bridge (`pyodide.code.run_js` + `pyodide_js.registerJsModule`) isn't
        specific to either host. The one thing that *is* host-specific is finding the site's own
        base URL from inside that worker (`self.location.origin` alone breaks under a subpath
        deployment like this one, at `/marimo/` off the repo's Pages site) — `pystanwasm` tries a
        few known Pyodide-host worker-script patterns and falls back cleanly, or takes an explicit
        `site_base` if you'd rather not rely on detection at all.
        """
    )
    return (mo,)


@app.cell
async def _(mo):
    import json

    import micropip
    from pyodide.code import run_js

    # Same site-root detection pystanwasm uses internally once it's
    # installed -- needed here too, one time, just to find the wheel itself.
    site_base = run_js(
        """
        (() => {
          const href = self.location.href;
          const markers = ["/extensions/", "/assets/"];
          for (const marker of markers) {
            const idx = href.indexOf(marker);
            if (idx !== -1) return href.slice(0, idx);
          }
          return self.location.origin;
        })()
        """
    )
    wheel_url = site_base + "/pystanwasm-0.1.0-py3-none-any.whl"
    await micropip.install(wheel_url)

    import pystanwasm as stan

    mo.md(f"pystanwasm `{stan.__version__}` loaded.")
    return json, stan


@app.cell
def _(mo):
    mo.md(r"""## A small linear regression""")
    return


@app.cell
def _():
    stan_code = """
    data {
      int<lower=0> N;
      vector[N] x;
      vector[N] y;
    }
    parameters {
      real alpha;
      real beta;
      real<lower=0> sigma;
    }
    model {
      y ~ normal(alpha + beta * x, sigma);
    }
    """
    data = {"N": 5, "x": [1, 2, 3, 4, 5], "y": [2.1, 3.9, 6.2, 7.8, 10.1]}
    return data, stan_code


@app.cell
async def _(data, mo, stan, stan_code):
    # stanwasm's own assets live at /stanwasm/pkg/stanwasm.js here (see
    # scripts/copy-stanwasm.sh and the Makefile's marimo-build target) rather
    # than JupyterLite's /files/stanwasm/pkg/... convention.
    model = stan.StanModel(stan_code, stanwasm_path="/stanwasm/pkg/stanwasm.js")
    fit = await model.sampling(data=data, iter=2000, warmup=1000, seed=42)

    mo.md(f"parameters: `{fit.param_names}`")
    return fit, model


@app.cell
def _(fit):
    fit.summary()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Parallel chains via Web Workers

        `sampling_parallel(n_chains=...)` spawns one plain-JS Web Worker per chain — no Pyodide in
        them, just stanwasm loaded fresh in each — so multiple chains run concurrently instead of
        one at a time. Confirmed working here the same way it does in JupyterLite: nested-Worker
        support (a Worker spawning another Worker) is a browser feature, not a notebook-host one.
        """
    )
    return


@app.cell
async def _(data, mo, model, stan_code):
    import time

    from pystanwasm import StanModel

    parallel_model = StanModel(stan_code, stanwasm_path=model.stanwasm_path)
    t0 = time.perf_counter()
    fits = await parallel_model.sampling_parallel(data=data, n_chains=4, iter=2000, warmup=1000)
    elapsed = time.perf_counter() - t0

    mo.md(
        f"{len(fits)} chains in {elapsed:.2f}s — "
        f"beta means: {[round(float(f['beta'].mean()), 3) for f in fits]}"
    )
    return


if __name__ == "__main__":
    app.run()
