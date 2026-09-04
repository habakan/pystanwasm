import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Logistic regression

        A binary classifier: two predictors, a decision boundary, and the
        posterior over where that boundary actually sits (not just a point
        estimate). Mirrors `Logistic-Regression.ipynb` in the JupyterLite demo.
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
        ## Data

        Two predictors, a linear decision boundary, Bernoulli-logit noise on
        top -- generated so the true boundary is known, to check the fit
        against it later.
        """
    )
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.default_rng(0)
    N = 300
    X = rng.normal(size=(N, 2))
    alpha_true, beta_true = -0.5, np.array([2.0, -1.5])
    p_true = 1 / (1 + np.exp(-(alpha_true + X @ beta_true)))
    y = rng.binomial(1, p_true)

    stan_code = """
    data {
      int<lower=0> N;
      int<lower=0> K;
      matrix[N, K] X;
      array[N] int<lower=0, upper=1> y;
    }
    parameters {
      real alpha;
      vector[K] beta;
    }
    model {
      alpha ~ normal(0, 5);
      beta ~ normal(0, 5);
      y ~ bernoulli_logit(alpha + X * beta);
    }
    """
    data = {"N": N, "K": 2, "X": X.tolist(), "y": y.tolist()}
    return X, alpha_true, beta_true, data, np, stan_code, y


@app.cell
async def _(data, mo, stan, stan_code):
    model = stan.StanModel(stan_code, stanwasm_path="/stanwasm/pkg/stanwasm.js")
    fit = await model.sampling(data=data, iter=2000, warmup=1000, seed=42)
    mo.md(f"parameters: `{fit.param_names}`")
    return (fit,)


@app.cell
def _(mo):
    mo.md(
        r"""
        `alpha` and `beta` are unconstrained reals in this model, so --
        unlike a `<lower=0>` parameter -- there's no constrained/unconstrained
        scale gap to account for here; these draws are directly comparable to
        `alpha_true`/`beta_true` below.

        ## Decision boundary
        """
    )
    return


@app.cell
def _(X, alpha_true, beta_true, fit, np, y):
    import matplotlib.pyplot as plt

    beta_names = [n for n in fit.param_names if n.startswith("beta")]
    beta_draws = np.column_stack([fit[n] for n in beta_names])
    beta_mean = beta_draws.mean(axis=0)
    alpha_mean = fit["alpha"].mean()

    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.plot(xs, -(alpha_true + beta_true[0] * xs) / beta_true[1], "k--", label="true boundary")
    plt.plot(xs, -(alpha_mean + beta_mean[0] * xs) / beta_mean[1], "g-", label="posterior mean")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("logistic regression: recovered vs true boundary")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
