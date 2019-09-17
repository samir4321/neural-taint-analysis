# test_derivative_ops.py

import unittest
import torch

import derivative_ops


class TestJacobian(unittest.TestCase):

    def setUp(self):
        self.D_in, self.D_out = 3, 3
        w = torch.ones(3, self.D_out)
        self.f = lambda x: x @ w + x
        self.x = torch.randn(2, self.D_in)
        self.exp_jac = [[2., 1., 1.],
                   [1., 2., 1.],
                   [1., 1., 2.]]
        self.exp_frobenius = 4.2426

    def test_jacobian(self):
        jac = derivative_ops.jacobian(self.f, self.x)
        assert torch.all(torch.eq(jac, torch.Tensor([self.exp_jac, self.exp_jac])))

    def test_batch_jac(self):
        batch_jac = derivative_ops.mean_saliency_map(self.f,
                                                     self.x)
        assert torch.all(torch.eq(torch.Tensor(self.exp_jac), batch_jac))

    def test_hessian(self):
        pass

    def test_sensitivity(self):
        sens = derivative_ops.sensitivity(self.f, self.x)
        expected = torch.Tensor([self.exp_frobenius, self.exp_frobenius])
        assert(torch.all(torch.lt(torch.abs(torch.add(sens, -expected)), 1e-4)))

    # def test_batch_jacobian(self):
    #    print(derivative_ops.batch_jacobian(self.f, self.x))


