import os

import numpy as np
import pytest

from driftlab import datasets


def test_registry_paths_exist():
    for name in datasets.MODELS.values():
        assert os.path.isdir(os.path.join(datasets.SD_REPO, "results", name)), name


@pytest.mark.parametrize("model", list(datasets.MODELS))
def test_cache_contract(model):
    cdir = os.path.join(datasets.CACHE_ROOT, datasets.MODELS[model])
    if not os.path.isdir(cdir):
        pytest.skip(f"no cache for {model}")
    c = datasets.load_cache(model)
    n, T, L, d = c["resid"].shape
    assert c["dims"].shape == (n, T - 1, 5)
    assert c["gave_in"].shape == (n,) and c["gave_in"].dtype == bool
    assert len(c["meta"]) == n
    # self-reports are 0-10 ratings wherever parsed
    vals = c["dims"][~np.isnan(c["dims"])]
    assert vals.size and vals.min() >= 0 and vals.max() <= 10
