import importlib
import pytest

def test_import_phoskhemia():
    import phoskhemia  # noqa

@pytest.mark.parametrize("mod", [
    "phoskhemia.data",
    "phoskhemia.fitting.global_fit",
])
def test_import_module(mod):
    importlib.import_module(mod)

def test_reexported_symbols():
    from phoskhemia import TransientAbsorption, KineticModel  # noqa: F401
    from phoskhemia.data import TransientAbsorption as TA2  # noqa: F401
