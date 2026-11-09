"""
Alias of :mod:`pyfli.cm_funcs.forward_model`, kept so existing
``pyfli.solver.forward_model`` imports resolve to the same module object.
"""

import sys

from pyfli.cm_funcs import forward_model as _forward_model

sys.modules[__name__] = _forward_model
