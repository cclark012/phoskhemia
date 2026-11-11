"""Public import smoke tests, runnable without a third-party test runner."""
import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_import_phoskhemia(self):
        importlib.import_module("phoskhemia")

    def test_import_data(self):
        importlib.import_module("phoskhemia.data")

    def test_import_global_fit(self):
        importlib.import_module("phoskhemia.fitting.global_fit")

    def test_reexported_symbols(self):
        from phoskhemia import TransientAbsorption, KineticModel
        from phoskhemia.data import TransientAbsorption as TA2
        self.assertIs(TransientAbsorption, TA2)
        self.assertTrue(callable(KineticModel))


if __name__ == "__main__":
    unittest.main()
