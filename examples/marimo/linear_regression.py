import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Linear regression

        Compile and sample a small linear regression with
        [`pystanwasm`](https://github.com/habakan/pystanwasm), mirroring
        `Stan-in-Pyodide.ipynb` in the JupyterLite demo. Same bridge, same model.
        """
    )
    return (mo,)


@app.cell
async def _(mo):
    import micropip
    from pyodide.code import run_js

    # self.location.origin alone is wrong on a GitHub-Pages-style project site
    # (https://host/repo-name/) -- it omits "/repo-name". self.location.href
    # resolves to the *kernel worker's own* script URL, which does include it,
    # at a stable path under a host-specific marker (pystanwasm tries a few).
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
    mo.md(r"""## Compile and sample a Stan model""")
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
    # stanwasm's assets live at /stanwasm/pkg/stanwasm.js here (see
    # scripts/copy-stanwasm.sh and the Makefile's marimo-build target),
    # not JupyterLite's /files/stanwasm/pkg/... convention.
    model = stan.StanModel(stan_code, stanwasm_path="/stanwasm/pkg/stanwasm.js")
    fit = await model.sampling(data=data, iter=2000, warmup=1000, seed=42)

    mo.md(f"parameters: `{fit.param_names}`")
    return (fit,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Inspect the draws

        `fit.to_dataframe()` returns a pandas `DataFrame` over the post-warmup
        draws, on their natural (constrained) scale -- `sigma` is already
        positive here, not log-scale.
        """
    )
    return


@app.cell
def _(fit):
    fit.summary()
    return


@app.cell
def _(fit):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.hist(fit["beta"], bins=30)
    plt.title("beta posterior")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
