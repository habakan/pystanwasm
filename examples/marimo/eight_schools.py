import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Eight schools

        The classic partial-pooling example (Rubin, 1981): eight schools each
        ran a coaching program and measured its effect on SAT scores, with a
        reported effect and standard error per school. A hierarchical model
        lets each school's estimate borrow strength from the others, shrinking
        noisy small-effect estimates toward the group mean rather than
        treating every school in isolation. Mirrors `Eight-Schools.ipynb` in
        the JupyterLite demo.
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
        ## Model

        Non-centered parameterization (`theta = mu + tau * theta_tilde` in a
        `transformed parameters` block, with `theta_tilde ~ normal(0, 1)`)
        rather than sampling `theta` directly with `theta ~ normal(mu, tau)`
        -- the direct form couples `theta` and `tau` tightly when `tau` is
        small, which is the textbook "funnel" geometry NUTS struggles with.
        This reparameterization is the standard fix.
        """
    )
    return


@app.cell
def _():
    stan_code = """
    data {
      int<lower=0> J;
      vector[J] y;
      vector[J] sigma;
    }
    parameters {
      real mu;
      real<lower=0> tau;
      vector[J] theta_tilde;
    }
    transformed parameters {
      vector[J] theta = mu + tau * theta_tilde;
    }
    model {
      theta_tilde ~ normal(0, 1);
      y ~ normal(theta, sigma);
    }
    """
    schools = ["A", "B", "C", "D", "E", "F", "G", "H"]
    y = [28, 8, -3, 7, -1, 1, 18, 12]
    sigma = [15, 10, 16, 11, 9, 11, 10, 18]
    data = {"J": 8, "y": y, "sigma": sigma}
    return data, schools, sigma, stan_code, y


@app.cell
async def _(data, mo, stan, stan_code):
    model = stan.StanModel(stan_code, stanwasm_path="/stanwasm/pkg/stanwasm.js")
    fit = await model.sampling(data=data, iter=2000, warmup=1000, seed=42)
    mo.md(f"parameters: `{fit.param_names}`")
    return (fit,)


@app.cell
def _(fit):
    fit.summary()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Shrinkage

        Each school's raw observed effect (`y ± sigma`) against its posterior
        estimate (`theta ± sd`). Schools with a large reported effect and a
        large standard error (noisy, unreliable) get pulled the furthest
        toward the group mean; schools with tight standard errors barely move.
        """
    )
    return


@app.cell
def _(fit, schools, sigma, y):
    import matplotlib.pyplot as plt
    import numpy as np

    theta_names = [n for n in fit.param_names if n.startswith("theta") and "tilde" not in n]
    theta_draws = np.column_stack([fit[n] for n in theta_names])
    theta_mean = theta_draws.mean(axis=0)
    theta_sd = theta_draws.std(axis=0)

    y_arr = np.array(y)
    sigma_arr = np.array(sigma)
    positions = np.arange(len(schools))

    plt.figure(figsize=(7, 5))
    plt.errorbar(y_arr, positions - 0.1, xerr=sigma_arr, fmt="o", label="observed", color="tab:gray")
    plt.errorbar(theta_mean, positions + 0.1, xerr=theta_sd, fmt="o", label="posterior (pooled)", color="tab:blue")
    plt.axvline(fit["mu"].mean(), linestyle="--", color="k", label="population mean")
    plt.yticks(positions, schools)
    plt.xlabel("effect")
    plt.legend()
    plt.title("eight schools: observed vs partially-pooled effects")
    plt.gca().invert_yaxis()
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
