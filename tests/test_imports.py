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
