import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Parallel chains via Web Workers

        `sampling()` runs one chain inside the notebook's own Pyodide kernel
        worker -- a single thread. `StanModel.sampling_parallel(n_chains=...)`
        instead spawns one plain-JS Web Worker per chain (no Pyodide in them,
        just stanwasm's wasm module loaded fresh in each), so multiple chains
        can run at once across whatever cores the browser gives them --
        something a single-threaded Pyodide session can't do on its own.
        Mirrors `Parallel-Chains.ipynb` in the JupyterLite demo; nested-Worker
        support is a browser feature, not a notebook-host one, so this works
        the same way here.

        This needs nested-Worker support (a Worker spawning another Worker):
        current Chrome/Edge/Firefox, and Safari 16.4+ (March 2023). Where
        that's unavailable it falls back to running the chains one at a time
        through `sampling()` and prints a warning -- it never just fails.
        """
    )
    return (mo,)


@app.cell
async def _(mo):
    import micropip
    from pyodide.code import run_js

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
    return (stan,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## A model with real work per draw

        10 predictors, 500 observations -- small enough to run comfortably
        in a browser tab, large enough (matrix products every gradient
        evaluation) that a single chain takes a fraction of a second rather
        than a few milliseconds, so parallelizing 4 of them is actually
        worth timing.
        """
    )
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.default_rng(7)
    N, K = 500, 10
    X = rng.normal(size=(N, K))
    beta_true = rng.normal(0, 2, size=K)
    alpha_true, sigma_true = 1.5, 1.2
    y = alpha_true + X @ beta_true + rng.normal(0, sigma_true, size=N)

    stan_code = """
    data {
      int<lower=0> N;
      int<lower=0> K;
      matrix[N, K] X;
      vector[N] y;
    }
    parameters {
      real alpha;
      vector[K] beta;
      real<lower=0> sigma;
    }
    model {
      alpha ~ normal(0, 5);
      beta ~ normal(0, 5);
      sigma ~ exponential(1);
      y ~ normal(alpha + X * beta, sigma);
    }
    """
    data = {"N": N, "K": K, "X": X.tolist(), "y": y.tolist()}
    N_CHAINS = 4
    return N_CHAINS, data, stan_code


@app.cell
def _(mo):
    mo.md(r"""## Sequential: `N_CHAINS` calls to `sampling()`""")
    return


@app.cell
async def _(N_CHAINS, data, mo, stan, stan_code):
    import time

    model = stan.StanModel(stan_code, stanwasm_path="/stanwasm/pkg/stanwasm.js")

    t0 = time.perf_counter()
    sequential_fits = [
        await model.sampling(data=data, iter=2000, warmup=1000, seed=42 + i)
        for i in range(N_CHAINS)
    ]
    sequential_time = time.perf_counter() - t0
    mo.md(f"sequential: {sequential_time:.2f}s for {N_CHAINS} chains")
    return model, sequential_fits, sequential_time, time


@app.cell
def _(mo):
    mo.md(r"""## Parallel: one `sampling_parallel()` call""")
    return


@app.cell
async def _(N_CHAINS, data, mo, model, sequential_time, time):
    t1 = time.perf_counter()
    parallel_fits = await model.sampling_parallel(data=data, n_chains=N_CHAINS, iter=2000, warmup=1000)
    parallel_time = time.perf_counter() - t1

    mo.md(
        f"parallel:   {parallel_time:.2f}s for {N_CHAINS} chains\n\n"
        f"speedup:    {sequential_time / parallel_time:.1f}x"
    )
    return parallel_fits, parallel_time


@app.cell
def _(N_CHAINS, parallel_time, sequential_time):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.bar(["sequential", "parallel"], [sequential_time, parallel_time], color=["tab:gray", "tab:blue"])
    plt.ylabel("wall-clock seconds")
    plt.title(f"{N_CHAINS} chains, sequential vs Worker-parallel")
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sanity check: same posterior either way

        Parallelizing changes nothing about what's being computed -- each
        chain is independent either way, just with a different seed.
        `beta[0]`'s posterior mean per chain should agree closely between
        the two runs.
        """
    )
    return


@app.cell
def _(N_CHAINS, parallel_fits, sequential_fits):
    import pandas as pd

    beta0_name = [n for n in sequential_fits[0].param_names if n.startswith("beta")][0]
    seq_means = [f[beta0_name].mean() for f in sequential_fits]
    par_means = [f[beta0_name].mean() for f in parallel_fits]
    pd.DataFrame(
        {"sequential": seq_means, "parallel": par_means},
        index=[f"chain {i}" for i in range(N_CHAINS)],
    )
    return


if __name__ == "__main__":
    app.run()
