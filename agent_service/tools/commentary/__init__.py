import os

# This is hacky, but we want to make sure that "from tools import *" gets ALL tools

dirname = os.path.dirname(os.path.abspath(__file__))

modules = []
for f in os.listdir(dirname):
    if f != "__init__.py" and os.path.isfile(f"{dirname}/{f}") and f[-3:] == ".py":
        modules.append(f[:-3])

__all__ = modules