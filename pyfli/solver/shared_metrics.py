"""
Alias of :mod:`pyfli.cm_stats.shared_metrics`, kept so existing
``pyfli.solver.shared_metrics`` imports resolve to the same module object.
"""

import sys

from pyfli.cm_stats import shared_metrics as _shared_metrics

sys.modules[__name__] = _shared_metrics
