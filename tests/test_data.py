import sys
sys.path.append(".")

import unittest
import time
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

    def assertAllclosed(self, first, second):
        test = np.allclose(first, second, rtol=1.e-5)
        if not test:
            print("Arrays are not equal:", first, second, sep='\n')
        self.assertTrue(test)

    def test_get_target(self): 
        """ """
        time_range = (300, 305)
        a = np.array([1.26370847, 1.26453853, 1.29455566])
        self.assertAllclosed(a, get_target(filenames, time_range))

    def test_get_data(self):
        time_range = (200, 300)
        start, end = time_range
        self.assertEqual(get_data(filenames, time_range).shape,
                         (len(filenames), end - start, COL_NUM))

    def test_all_slice(self): 
        """ """
        a = [COLUMNS[i] for i in SINGLE_COL]
        b = ['_GS', '_DRIFT', '_IVV']
        self.assertListEqual(a, b)
        
        a = [COLUMNS[i] for i in [SS_CP,PITH_CP,ROLL_CP,SS_FO,PITH_FO,ROLL_FO]]
        b = [['_SSTICK_CAPT', '_SSTICK_CAPT-1', '_SSTICK_CAPT-2', 
             '_SSTICK_CAPT-3'],
             ['_PITCH_CAPT_SSTICK', '_PITCH_CAPT_SSTICK-1', 
              '_PITCH_CAPT_SSTICK-2', '_PITCH_CAPT_SSTICK-3'],
             ['_ROLL_CAPT_SSTICK', '_ROLL_CAPT_SSTICK-1', '_ROLL_CAPT_SSTICK-2',
              '_ROLL_CAPT_SSTICK-3'],
             ['_SSTICK_FO', '_SSTICK_FO-1', '_SSTICK_FO-2', '_SSTICK_FO-3'],
             ['_PITCH_FO_SSTICK', '_PITCH_FO_SSTICK-1', '_PITCH_FO_SSTICK-2', 
             '_PITCH_FO_SSTICK-3'],
             ['_ROLL_FO_SSTICK', '_ROLL_FO_SSTICK-1', '_ROLL_FO_SSTICK-2', 
              '_ROLL_FO_SSTICK-3']]
        self.assertListEqual(a, b)

    def test_split_data(self):
        num_train = 50
        num_valid = 5
        num_test = 5
        X_time_range = (240, 300)
        y_time_range = (300, 305)
        start, end = X_time_range

        splited = split_data(
            num_train=num_train, num_validation=num_valid, num_test=5,
            seed=int(time.time()), X_time_range=X_time_range, 
            y_time_range=y_time_range
        )

        self.assertEqual(splited['X_val'].shape, 
                        (num_valid, end - start, COL_NUM))
        self.assertEqual(splited['X_train'].shape, 
                        (num_train, end - start, COL_NUM)) 
        self.assertEqual(splited['X_test'].shape, 
                        (num_test, end - start, COL_NUM))

        self.assertAllclosed(splited['y_val'],
                             get_target(splited['f_val'], y_time_range))      
        self.assertAllclosed(splited['y_test'], 
                             get_target(splited['f_test'], y_time_range))
        self.assertAllclosed(splited['y_train'], 
                             get_target(splited['f_train'], y_time_range))



if __name__ == '__main__':
    unittest.main()