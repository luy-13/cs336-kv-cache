import importlib.metadata

try:
    __version__ = importlib.metadata.version("kvlm")
except importlib.metadata.PackageNotFoundError:
    pass
