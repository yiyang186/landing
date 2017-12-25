import unittest
from test_data import TestDataModule

if __name__ == '__main__':
    suite = unittest.TestSuite()
    tests = [
        TestDataModule("test_get_target"), 
        TestDataModule("test_get_data"),
        TestDataModule("test_split_data"),
    ]
    suite.addTests(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)