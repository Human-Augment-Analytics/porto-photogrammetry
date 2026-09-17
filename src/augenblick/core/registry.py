"""Name-to-class registries backing method lookup and CLI discovery."""
import logging

from augenblick.core.errors import MethodNotFound

logger = logging.getLogger(__name__)

SFM_REGISTRY: dict[str, type] = {}
RECONSTRUCTION_REGISTRY: dict[str, type] = {}
MASK_REGISTRY: dict[str, type] = {}


def _register(registry: dict[str, type], cls: type, kind: str) -> type:
    """Add cls to registry under cls.name, rejecting duplicates."""
    if cls.name in registry:
        raise ValueError(f"duplicate {kind} method name {cls.name!r}")
    registry[cls.name] = cls
    return cls


def get_method(registry: dict[str, type], name: str, kind: str) -> type:
    """Look up a method class, reporting the available names when it is absent.

    Args:
        registry: The stage's name-to-class registry.
        name: Method name as typed on the command line.
        kind: Stage label used in the error message, e.g. "SfM".

    Returns:
        The registered class.
    """
    try:
        return registry[name]
    except KeyError:
        available = ", ".join(sorted(registry)) or "none"
        raise MethodNotFound(f"unknown {kind} method {name!r}; available: {available}") from None


def register_sfm(cls: type) -> type:
    """Register an SfM method class under its `name` attribute."""
    return _register(SFM_REGISTRY, cls, "SfM")


def register_reconstruction(cls: type) -> type:
    """Register a reconstruction method class under its `name` attribute."""
    return _register(RECONSTRUCTION_REGISTRY, cls, "reconstruction")


def register_mask(cls: type) -> type:
    """Register a masking method class under its `name` attribute."""
    return _register(MASK_REGISTRY, cls, "mask")


def get_sfm(name: str) -> type:
    """Return the registered SfM method class, or raise MethodNotFound."""
    return get_method(SFM_REGISTRY, name, "SfM")


def get_reconstruction(name: str) -> type:
    """Return the registered reconstruction method class, or raise MethodNotFound."""
    return get_method(RECONSTRUCTION_REGISTRY, name, "reconstruction")


def get_mask(name: str) -> type:
    """Return the registered mask method class, or raise MethodNotFound."""
    return get_method(MASK_REGISTRY, name, "mask")
