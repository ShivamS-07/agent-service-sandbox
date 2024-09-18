import os

from agent_service.tools.commentary import *  # noqa
from agent_service.tools.kpis import *  # noqa
from agent_service.tools.LLM_analysis import *  # noqa
from agent_service.tools.product_comparison import *  # noqa
from agent_service.tools.profiler import *  # noqa
from agent_service.tools.stock_rank_by_text import *  # noqa

# This is hacky, but we want to make sure that "from tools import *" gets ALL tools

dirname = os.path.dirname(os.path.abspath(__file__))

modules = []
for f in os.listdir(dirname):
    if f != "__init__.py" and os.path.isfile(f"{dirname}/{f}") and f[-3:] == ".py":
        modules.append(f[:-3])

__all__ = modules
