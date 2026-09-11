"""`to_inference_data` runs in plain CPython: a `StanFit` is only numpy arrays.

    pip install -e ".[test]" && pytest
"""

import numpy as np
import pytest

from pystanwasm import StanFit, to_inference_data

# stanwasm's order: 1-based indices, matrices row-major.
NAMES = ["alpha", "beta[1]", "beta[2]", "L[1,1]", "L[1,2]", "L[2,1]", "L[2,2]"]
WARMUP, DRAWS = 3, 5


def fit(chain):
    # Every entry encodes (chain, draw, column), so a misplaced value is visible.
    rows = np.arange(WARMUP + DRAWS)[:, None]
    cols = np.arange(len(NAMES))[None, :]
    return StanFit(NAMES, chain * 1000 + rows * 10 + cols, WARMUP)


def test_groups_indexed_names_into_shaped_variables():
    post = to_inference_data([fit(0), fit(1)]).posterior
    assert post["alpha"].shape == (2, DRAWS)
    assert post["beta"].shape == (2, DRAWS, 2)
    assert post["L"].shape == (2, DRAWS, 2, 2)
    # chain 1, first post-warmup draw (row 3): L[2,1] is column 5.
    assert float(post["L"].values[1, 0, 1, 0]) == 1000 + 30 + 5
    assert float(post["beta"].values[0, 4, 1]) == 70 + 2


def test_keeps_warmup_only_when_asked():
    assert "warmup_posterior" not in to_inference_data([fit(0)]).children
    idata = to_inference_data([fit(0)], save_warmup=True)
    assert idata.warmup_posterior["alpha"].shape == (1, WARMUP)


def test_accepts_a_single_fit_as_one_chain():
    assert to_inference_data(fit(0)).posterior["alpha"].shape == (1, DRAWS)


def test_refuses_chains_of_different_lengths():
    short = StanFit(NAMES, np.zeros((WARMUP + DRAWS - 1, len(NAMES))), WARMUP)
    with pytest.raises(ValueError):
        to_inference_data([fit(0), short])
