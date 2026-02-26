import importlib

def test_import_phoskhemia():
    import phoskhemia  # noqa: F401

def test_import_key_modules():
    importlib.import_module("phoskhemia.data")
    importlib.import_module("phoskhemia.fitting")
    importlib.import_module("phoskhemia.fitting.global_fit")
    importlib.import_module("phoskhemia.simulation.transient_absorption")
    importlib.import_module("phoskhemia.visualization.plotting")
