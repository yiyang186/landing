import sys
sys.path.append("..")

import unittest
import numpy as np

from data import *
from configure import *

filenames = [
    '2016-03-02-17-18-10_A320__ZJSY_6372_160302142102817.csv',
    '2016-03-03-06-39-21_A320__ZUCK_6208_160303044001817.csv',
    '2016-03-05-04-46-16_A320__ZUCK_6264_160305023820817.csv'
]

class TestDataModule(unittest.TestCase):
    """Test data.py"""
    
    @classmethod
    def setUpClass(cls):
        pass

    def test_get_target(self): 
        """ """
        time_range = (300, 305)
        a = np.array([1.26370847, 1.26453853, 1.29455566])
        self.assertTrue(np.allclose(a, get_target(filenames, time_range), 
                        rtol=1.e-6))

    def test_get_data(self):
        time_range = (200, 300)
        start, end = time_range
        self.assertEqual(get_data(filenames, time_range).shape,
                         (len(filenames), end - start, COL_NUM))

    def test_split_data(self):
        num_train = 50
        num_valid = 5
        num_test = 5
        X_time_range = (240, 300)
        y_time_range = (300, 305)
        start, end = X_time_range

        splited = split_data(
            num_train=num_train, num_validation=num_valid,
            num_test=5, seed=0, X_time_range=X_time_range, 
            y_time_range=y_time_range
        )

        self.assertEqual(splited['X_val'].shape, 
                        (num_valid, end - start, COL_NUM))
        self.assertEqual(splited['X_train'].shape, 
                        (num_train, end - start, COL_NUM)) 
        self.assertEqual(splited['X_test'].shape, 
                        (num_test, end - start, COL_NUM))

        self.assertTrue(np.allclose(splited['y_val'], 
                        get_target(splited['f_val'], y_time_range)))        
        self.assertTrue(np.allclose(splited['y_test'], 
                        get_target(splited['f_test'], y_time_range)))   
        self.assertTrue(np.allclose(splited['y_train'], 
                        get_target(splited['f_train'], y_time_range)))     



if __name__ == '__main__':
    unittest.main()