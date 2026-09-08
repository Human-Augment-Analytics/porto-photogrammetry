"""Registry duplicate rejection and MethodNotFound discoverability."""
import pytest

from augenblick.core import registry
from augenblick.core.errors import MethodNotFound


@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(registry, "SFM_REGISTRY", {})
    return registry.SFM_REGISTRY


def test_register_returns_class_unchanged(empty_registry):
    class Dummy:
        name = "dummy"

    assert registry.register_sfm(Dummy) is Dummy
    assert registry.get_sfm("dummy") is Dummy


def test_duplicate_name_rejected(empty_registry):
    class A:
        name = "dup"

    class B:
        name = "dup"

    registry.register_sfm(A)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register_sfm(B)


def test_missing_name_lists_available(empty_registry):
    class A:
        name = "alpha"

    class B:
        name = "beta"

    registry.register_sfm(A)
    registry.register_sfm(B)
    with pytest.raises(MethodNotFound) as exc:
        registry.get_sfm("nope")
    assert "alpha, beta" in str(exc.value)


def test_real_registries_populated():
    import augenblick.reconstruction  # noqa: F401
    import augenblick.sfm  # noqa: F401

    assert set(registry.RECONSTRUCTION_REGISTRY) == {"2dgs", "sugar", "pgsr", "gw"}
    assert set(registry.SFM_REGISTRY) == {"vggt", "colmap", "turntable", "hull"}
