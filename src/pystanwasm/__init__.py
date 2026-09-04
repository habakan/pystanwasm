"""pystanwasm: run stanwasm — Stan compiled to WebAssembly — from Python.

This only does anything useful inside a Pyodide runtime (e.g. a JupyterLite
notebook running in a browser): it bridges Pyodide to stanwasm's own wasm
module via JS interop, behind a small, PyStan-flavored API. It is not
affiliated with PyStan and is not a clone of it — see `StanModel`'s
docstring for what's simplified.

    import pystanwasm as stan
    fit = await stan.StanModel(stan_code).sampling(data=data, seed=42)
    fit.summary()
"""

from ._bridge import StanFit, StanModel

__all__ = ["StanModel", "StanFit"]
__version__ = "0.1.0"
