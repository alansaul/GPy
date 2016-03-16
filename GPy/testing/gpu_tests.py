import unittest
import numpy as np
import GPy

class RBF_GPU_Psi_statistics_Tests(unittest.TestCase):

    def setUp(self):
        from GPy.core.parameterization.variational import NormalPosterior
        N,M,Q = 100,20,3

        X = np.random.randn(N,Q)
        X_var = np.random.rand(N,Q)+0.01
        self.Z = np.random.randn(M,Q)
        self.qX = NormalPosterior(X, X_var)

        self.w1 = np.random.randn(N)
        self.w2 = np.random.randn(N,M)
        self.w3 = np.random.randn(M,M)
        self.w3n = np.random.randn(N,M,M)
        self.w3[:] = 0.

    def test_psi_statistics(self):
        try:
            from GPy.kern.src.psi_comp import PSICOMP_RBF_Cython, PSICOMP_RBF_GPU
            from GPy.kern import RBF
            psicomp_cython = PSICOMP_RBF_Cython()
            psicomp_gpu = PSICOMP_RBF_GPU()
            kern = RBF(self.Z.shape[1],ARD=True, variance = 0.3, lengthscale=.7)

            for return_psicov in [True]:
                for return_n in [True, False]:
                    rs1 = psicomp_cython.psicomputations(kern, self.Z, self.qX, return_psicov, return_n)
                    rs2 = psicomp_gpu.psicomputations(kern, self.Z, self.qX, return_psicov, return_n)
                    print return_psicov, return_n
                    print rs1, rs2
                    comp = [np.allclose(r1,r2) for r1, r2 in zip(rs1, rs2)]
                    self.assertTrue(np.all(comp))
        except:
            pass

    def test_kernels(self):
        try:
            from GPy.kern import RBF
            Q = self.Z.shape[1]
            kernels = [RBF(Q,ARD=True, useGPU=True),RBF(Q,ARD=False, useGPU=True)]

            for k in kernels:
                k.randomize()
                self._test_kernel_param(k)
                self._test_Z(k)
                self._test_qX(k)
                # self._test_kernel_param(k, psi2n=True)
                # self._test_Z(k, psi2n=True)
                # self._test_qX(k, psi2n=True)
        except:
            pass
            
    def _test_kernel_param(self, kernel, psi2n=False):

        def f(p):
            kernel.param_array[:] = p
            psi0 = kernel.psi0(self.Z, self.qX)
            psi1 = kernel.psi1(self.Z, self.qX)
            if not psi2n:
                psicov = kernel.psicov(self.Z, self.qX)
                return (self.w1*psi0).sum() + (self.w2*psi1).sum() + (self.w3*psicov).sum()
            else:
                psicov = kernel.psicovn(self.Z, self.qX)
                return (self.w1*psi0).sum() + (self.w2*psi1).sum() + (self.w3n*psicov).sum()
            
        def df(p):
            kernel.param_array[:] = p
            kernel.update_gradients_expectations_psicov(self.w1, self.w2, self.w3 if not psi2n else self.w3n, self.Z, self.qX)
            return kernel.gradient.copy()

        from GPy.models import GradientChecker
        f(kernel.param_array.copy())
        m = GradientChecker(f, df, kernel.param_array.copy())
        m.checkgrad(verbose=1)
        self.assertTrue(m.checkgrad())

    def _test_Z(self, kernel, psi2n=False):

        def f(p):
            psi0 = kernel.psi0(p, self.qX)
            psi1 = kernel.psi1(p, self.qX)
            if not psi2n:
                psicov = kernel.psicov(p, self.qX)
                return (self.w1*psi0).sum() + (self.w2*psi1).sum() + (self.w3*psicov).sum()
            else:
                psicov = kernel.psicovn(p, self.qX)
                return (self.w1*psi0).sum() + (self.w2*psi1).sum() + (self.w3n*psicov).sum()

        def df(p):
            return kernel.gradients_Z_expectations_psicov(self.w1, self.w2, self.w3 if not psi2n else self.w3n, p, self.qX)

        from GPy.models import GradientChecker
        f(self.Z.copy())
        m = GradientChecker(f, df, self.Z.copy())
        self.assertTrue(m.checkgrad())

    def _test_qX(self, kernel, psi2n=False):

        def f(p):
            self.qX.param_array[:] = p
            self.qX._trigger_params_changed()
            psi0 = kernel.psi0(self.Z, self.qX)
            psi1 = kernel.psi1(self.Z, self.qX)
            if not psi2n:
                psicov = kernel.psicov(self.Z, self.qX)
                return (self.w1*psi0).sum() + (self.w2*psi1).sum() + (self.w3*psicov).sum()
            else:
                psicov = kernel.psicovn(self.Z, self.qX)
                return (self.w1*psi0).sum() + (self.w2*psi1).sum() + (self.w3n*psicov).sum()

        def df(p):
            self.qX.param_array[:] = p
            self.qX._trigger_params_changed()
            kernel.psicov(self.Z, self.qX)
            grad =  kernel.gradients_qX_expectations_psicov(self.w1, self.w2, self.w3 if not psi2n else self.w3n, self.Z, self.qX)
            self.qX.set_gradients(grad)
            return self.qX.gradient.copy()

        from GPy.models import GradientChecker
        f(self.qX.param_array.copy())
        m = GradientChecker(f, df, self.qX.param_array.copy())
        self.assertTrue(m.checkgrad())
