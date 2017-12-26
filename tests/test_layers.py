import sys
sys.path.append(".")

import unittest
import time
import numpy as np

from layers import *
from configure import *
from .gradient_check import *

class TestLayersModule(unittest.TestCase):
    """Test layers.py"""
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(437)

    def assertRelerror(self, first, second):
        """ test relative error """
        x, y = first, second
        rerr = np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
        self.assertAlmostEqual(rerr, 0.0, places=6)

    def assertAllclosed(self, first, second):
        """"""
        test = np.allclose(first, second, rtol=1.e-5)
        if not test:
            print("Arrays are not equal:", first, second, sep='\n')
        self.assertTrue(test)

    def test_temporal_dot_forward(self):
        """"""
        N, T, D = 10, 20, 12
        x = np.random.rand(N, T, D)
        w = np.random.randn(D)
        b = np.random.randn(1)
        out, _ = temporal_dot_forward(x, w, b)
        self.assertTupleEqual(out.shape, (N, T))

    def test_temporal_dot_backward(self):
        """"""
        N, T, D = 10, 20, 12
        x = np.random.rand(N, T, D)
        w = np.random.randn(D)
        b = np.random.randn(1)
        out, cache = temporal_dot_forward(x, w, b)
        dout = np.random.randn(*out.shape)

        fx = lambda x: temporal_dot_forward(x, w, b)[0]
        fw = lambda w: temporal_dot_forward(x, w, b)[0]
        fb = lambda b: temporal_dot_forward(x, w, b)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        dw_num = eval_numerical_gradient_array(fw, w, dout)
        db_num = eval_numerical_gradient_array(fb, b, dout)
        
        dx, dw, db = temporal_dot_backward(dout, cache)

        self.assertTupleEqual(dx.shape, x.shape)
        self.assertTupleEqual(dw.shape, w.shape)
        self.assertTupleEqual(np.array([db]).shape, b.shape)

        self.assertRelerror(dx_num, dx)
        self.assertRelerror(dw_num, dw)
        self.assertRelerror(db_num, db)

    def test_softmax_forward(self):
        """"""
        N, T = 10, 20
        x = np.random.randn(N, T)
        y, _ = softmax_forward(x)
        self.assertTupleEqual(y.shape, (N, T))
        self.assertAllclosed(y.sum(axis=1), np.ones(N))

    def test_softmax_backward(self):
        """"""
        N, T = 10, 20
        x = np.random.randn(N, T)
        y, cache = softmax_forward(x)

        dout = np.random.randn(N, T)
        dx = softmax_backward(dout, cache)
        self.assertTupleEqual(dx.shape, x.shape)

        fx = lambda x: softmax_forward(x)[0]
        dx_num = eval_numerical_gradient_array(fx, x, dout)
        self.assertRelerror(dx_num, dx)

    def test_attension_forward(self):
        """"""
        N, T, H = 32, 60, 32
        x = np.random.randn(N, H)
        w = np.random.randn(H, T)
        b = np.random.randn(T)
        out, _ = attension_forward(x, w, b)
        self.assertTupleEqual(out.shape, (N, T))
        self.assertAllclosed(out.sum(axis=1) , np.ones(N))

    def test_attension_backward(self):
        """"""
        N, H, T = 128, 32, 60
        x = np.random.randn(N, H)
        w = np.random.randn(H, T)
        b = np.random.randn(T)
        out, cache = attension_forward(x, w, b)

        dout = np.random.randn(N, T)
        dx, dw, db = attension_backward(dout, cache)
        self.assertTupleEqual(dx.shape, x.shape)
        self.assertTupleEqual(dw.shape, w.shape)
        self.assertTupleEqual(db.shape, b.shape)

        fx = lambda x: attension_forward(x, w, b)[0]
        fw = lambda w: attension_forward(x, w, b)[0]
        fb = lambda b: attension_forward(x, w, b)[0]
        dx_num = eval_numerical_gradient_array(fx, x, dout)
        dw_num = eval_numerical_gradient_array(fw, w, dout)
        db_num = eval_numerical_gradient_array(fb, b, dout)
        self.assertRelerror(dx_num, dx)
        self.assertRelerror(dw_num, dw)
        self.assertRelerror(db_num, db)

    def test_temporal_leastsquare_loss(self):
        """"""
        N, T = 128, 60
        x = np.random.randn(N, T).astype(np.float64)
        y = np.random.randn(N, T).astype(np.float64) + 1
        loss, dx = temporal_leastsquare_loss(x, y)

        fx = lambda x: temporal_leastsquare_loss(x, y)[0]
        dx_num = eval_numerical_gradient(fx, x, verbose=False)
        self.assertRelerror(dx_num, dx)


if __name__ == '__main__':
    unittest.main()