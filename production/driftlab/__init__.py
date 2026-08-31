"""driftlab — production code for the cross-model stance-drift study.

Layout contract for this bench (and future ones):

  production/   this package + its unit tests. Everything a claim rests on
                lives here, tested.
  review/       Jupyter notebooks that review this package section by
                section: intent, expectations, the code run line by line
                next to its tests, and heavy visualization (shapes, sizes,
                plots). Every claim gets Methods / Results / Discussion.

Nothing in review/ defines logic; notebooks import from here so that what
is reviewed is exactly what runs.
"""

from .datasets import DIMS, MODELS, load_turns, load_all_turns, load_cache
from .outcomes import wilson, outcome_table
from .trends import trend_table, plot_trends, plot_spaghetti
from .interp import dim_layer_map, gavein_turn_probe, ridge_cv_r

__all__ = [
    "DIMS", "MODELS", "load_turns", "load_all_turns", "load_cache",
    "wilson", "outcome_table",
    "trend_table", "plot_trends", "plot_spaghetti",
    "dim_layer_map", "gavein_turn_probe", "ridge_cv_r",
]
