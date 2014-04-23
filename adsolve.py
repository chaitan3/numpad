import os
import sys
import unittest
import numbers
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

sys.path.append(os.path.realpath('..')) # for running unittest

from numpad.adstate import _add_ops
from numpad.adarray import *
from numpad.adarray import __DEBUG_MODE__, _DEBUG_perturb_new

class ResidualState(IntermediateState):
    def __init__(self, prev_state):
        host = prev_state.host()
        IntermediateState.__init__(self, host, prev_state, 1, None)

    # --------- recursive functions for adjoint differentiation -------- #

    def clear_f_diff_self(self):
        IntermediateState.clear_f_diff_self(self)
        if hasattr(self, 'solution') and self.solution():
            self.solution().clear_f_diff_self()
    
    def adjoint_recurse(self, f):
        if f is self or hasattr(self, '_f_diff_self') or \
                self.solution() is None:
            return IntermediateState.adjoint_recurse(self, f)

        f_diff_soln = self.solution().adjoint_recurse(f)
        if f_diff_soln is 0:
            return IntermediateState.adjoint_recurse(self, f)
        else:
            if hasattr(f_diff_soln, 'todense'):
                f_diff_soln = f_diff_soln.todense()
            f_diff_soln = np.array(f_diff_soln)
            # inverse of Jacobian matrix
            self_diff_soln = self.solution().jacobian.T
            soln_diff_self = splinalg.factorized(self_diff_soln.tocsc())
            f_diff_self = np.array([-soln_diff_self(b) for b in f_diff_soln])
            f_diff_self = np.matrix(f_diff_self.reshape(f_diff_soln.shape))

            f_diff_self_0 = IntermediateState.adjoint_recurse(self, f)
            f_diff_self = _add_ops(f_diff_self, f_diff_self_0)

            self.f_diff_self = f_diff_self
            return f_diff_self


class SolutionState(IntermediateState):
    def __init__(self, host, residual_state, jacobian):
        IntermediateState.__init__(self, host, None, None, None)
        assert isinstance(residual_state, ResidualState)
        assert residual_state.size == self.size
        residual_state.solution = weakref.ref(self)
        self.residual = residual_state
        if not isinstance(jacobian, numbers.Number):
            assert jacobian.shape == (self.size, self.size)
        self.jacobian = jacobian

    def obliviate(self):
        self.residual = None
        self.jacobian = None

    # --------- recursive functions for tangent differentiation -------- #

    def clear_self_diff_u(self):
        IntermediateState.clear_self_diff_u(self)
        if self.residual:
            self.residual.clear_self_diff_u()
    
    def diff_recurse(self, u):
        if u is self or hasattr(self, '_self_diff_u') or self.residual is None:
            return IntermediateState.diff_recurse(self, u)

        resid_diff_u = self.residual.diff_recurse(u)
        if resid_diff_u is 0:
            self_diff_u = 0
        else:
            if hasattr(resid_diff_u, 'todense'):
                resid_diff_u = resid_diff_u.todense()
            resid_diff_u = np.array(resid_diff_u)
            # inverse of Jacobian matrix
            resid_diff_self = self.jacobian
            self_diff_resid = splinalg.factorized(resid_diff_self.tocsc())
            self_diff_u = np.transpose([-self_diff_resid(b) \
                                        for b in resid_diff_u.T])
            self_diff_u = np.matrix(self_diff_u.reshape(resid_diff_u.shape))

        self.self_diff_u = self_diff_u
        return self_diff_u


class adsolution(adarray):
    def __init__(self, solution, residual, n_Newton):
        assert isinstance(solution, adarray)
        assert isinstance(residual, adarray)

        residual._current_state = ResidualState(residual._current_state)

        adarray.__init__(self, solution._base)
        self._current_state = SolutionState(self, residual._current_state,
                                            residual.diff(solution))
        self._n_Newton = n_Newton
        self._res_norm = np.linalg.norm(residual._base)

        _DEBUG_perturb_new(self)

    def obliviate(self):
        self._initial_state.obliviate()
        del self._n_Newton
        del self._res_norm

import time
def solve(func, u0, args=(), kargs={},
          max_iter=10, abs_tol=1E-6, rel_tol=1E-6, verbose=True):
    u = adarray(base(u0).copy())
    _DEBUG_perturb_new(u)

    for i_Newton in range(max_iter):
        start = time.time()
        res = func(u, *args, **kargs)  # TODO: how to put into adarray context?
        res_norm = np.linalg.norm(res._base, np.inf)
        if verbose:
            print('    ', i_Newton, res_norm)
        if not np.isfinite(res_norm):
            break

        if i_Newton == 0:
            res_norm0 = res_norm
        if res_norm < max(abs_tol, rel_tol * res_norm0):
            return adsolution(u, res, i_Newton + 1)

        # Newton update
        J = res.diff(u).tocsr()
        start2 = time.time()
        minus_du = splinalg.spsolve(J, np.ravel(res._base))
        print time.time()-start2

#        P = splinalg.spilu(J, drop_tol=1e-5)
#        M_x = lambda x: P.solve(x)
#        M = splinalg.LinearOperator((n * m, n * m), M_x)
#        minus_du = splinalg.gmres(J, np.ravel(res._base), M=M,tol=1e-6)

        u._base -= minus_du.reshape(u.shape)
        u = adarray(u._base)  # unlink operation history if any
        _DEBUG_perturb_new(u)

        print time.time()-start
    # not converged
    return adsolution(u, res, np.inf)



# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _Poisson1dTest(unittest.TestCase):
    def residual(self, u, f, dx):
        res = -2 * u
        res[1:] += u[:-1]
        res[:-1] += u[1:]
        return res / dx**2 + f

    def testPoisson1d(self):
        N = 256
        dx = adarray(1. / N)

        f = ones(N-1)
        u = zeros(N-1)

        u = solve(self.residual, u, (f, dx), verbose=False)

        x = np.linspace(0, 1, N+1)[1:-1]
        self.assertAlmostEqual(0, np.abs(u._base - 0.5 * x * (1 - x)).max())

        # solve tangent equation
        dudx = np.array(u.diff(dx)).reshape(u.shape)
        self.assertAlmostEqual(0, np.abs(dudx - 2 * u._base / dx._base).max())

        # solve adjoint equation
        J = u.sum()
        dJdf = J.diff(f)
        self.assertAlmostEqual(0, np.abs(dJdf - u._base).max())


class _Poisson2dTest(unittest.TestCase):
    def residual(self, u, f, dx, dy):
        res = -(2 / dx**2 + 2 / dy**2) * u
        res[1:,:] += u[:-1,:] / dx**2
        res[:-1,:] += u[1:,:] / dx**2
        res[:,1:] += u[:,:-1] / dy**2
        res[:,:-1] += u[:,1:] / dy**2
        res += f
        return res

    def testPoisson2d(self):
        #N, M = 256, 512
        N, M = 256, 64
        dx, dy = adarray([1. / N, 1. / M])

        f = ones((N-1, M-1))
        u = ones((N-1, M-1))

        u = solve(self.residual, u, (f, dx, dy), verbose=False)

        x = np.linspace(0, 1, N+1)[1:-1]
        y = np.linspace(0, 1, M+1)[1:-1]

        # solve tangent equation
        dudx = np.array(u.diff(dx)).reshape(u.shape)
        dudy = np.array(u.diff(dy)).reshape(u.shape)

        self.assertAlmostEqual(0,
            abs(2 * u._base - (dudx * dx._base + dudy * dy._base)).max())

        # solve adjoint equation
        J = u.sum()
        dJdf = J.diff(f)

        self.assertAlmostEqual(0, abs(np.ravel(u._base) - dJdf).max())


if __name__ == '__main__':
    # _Poisson2dTest().testPoisson2d()
    unittest.main()
