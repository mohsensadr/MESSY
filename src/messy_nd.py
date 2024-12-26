import numpy as np
import sympy as sp
import random
from scipy import integrate
from scipy.linalg import solve, LinAlgError
from numpy.linalg import norm
from sympy import oo, zoo, nan
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import pandas as pd
from sklearn.neighbors import KernelDensity
import statistics as stat
from collections import defaultdict
import sys
import warnings
warnings.filterwarnings("ignore")
import time

class MESSY_nd:
    def __init__(self, dim=2, x_order=2, n_bases=2, poly_order=4, tree_depth=2, binary_operators = [sp.Mul], unary_functions = pow):
        self.x = sp.symbols('x:' + str(dim), real=True)
        self.x_order = x_order
        self.n_bases = n_bases
        self.poly_order = poly_order
        self.tree_depth = tree_depth
        self.binary_operators = binary_operators
        self.unary_functions = unary_functions
        self.dim = 2

    def np_lambdify(self, varname, func):
        lamb = sp.lambdify(varname, func, modules=['numpy'])
        if func.is_constant():
            return lambda t: np.full_like(t, lamb(t))
        else:
            return lambda t: lamb(np.array(t))

    def moments(self, h, z, w=None):
      return np.array([np.average(np.apply_along_axis(lambda args: hh(*args), 1, z), weights=w) for hh in h])

    def KL_Divergence(self, true_dist, pred_dist, xmin, xmax, ymin, ymax):
      xx = np.concatenate((np.linspace(xmin,xmax,10000).reshape(-1,1), np.linspace(ymin,ymax,10000).reshape(-1,1)), axis=1)
      p, q = true_dist(xx), np.apply_along_axis(lambda args: pred_dist(*args), 1, xx) #.reshape(xx.shape)
      p = np.asarray(p, dtype=np.float)
      q = np.asarray(q, dtype=np.float)
      return np.sum(p * np.log( p / (q+1e-10) + 1e-10))

    def multi_dim_uniform_sample(self, samples, num_samples=10000):
        mins = np.min(samples, axis=0)
        maxs = np.max(samples, axis=0)
        return np.array([np.random.uniform(mins[i], maxs[i], num_samples) for i in range(samples.shape[1])]).T

    def kl_div(self, XX, f, w=None):
        return - np.average(np.apply_along_axis(lambda args: np.log(f(*args)) + 1e-10, 1, XX), weights=w)

    def generate_polynomials(self, dimension, order):
        x = sp.symbols('x:' + str(dimension), real=True)

        result = set()

        def add_terms(term, remaining_order, last_var_idx):
            if remaining_order == 0:
                result.add(term)
                return
            for idx in range(last_var_idx, dimension):
                add_terms(term*x[idx], remaining_order - 1, idx)

        for o in range(1, order + 1):
            for d in range(dimension):
                add_terms(x[d], o - 1, d)

        return sorted(result, key=str)

    def generate_symbolic_expr(self, depth=2):
        x = sp.symbols('x:' + str(self.dim), real=True)
        if depth == 0 or random.random() < 0.3:
            return random.choice(self.unary_functions)(random.choice(x))
        if random.random() < 0.7:
            op = random.choice(self.binary_operators)
            left = self.generate_symbolic_expr(depth - 1)
            right = self.generate_symbolic_expr(depth - 1)
            return op(left, right)
        else:
            return random.choice(self.unary_functions)(self.generate_symbolic_expr(depth - 1))

    def Hess(self, X, dh):
        Nm = len(dh)
        L = np.zeros((Nm, Nm))
        for i in range(Nm):
            for j in range(Nm):
                L[i, j] = np.mean(np.apply_along_axis(lambda args: dh[i](*args), 1, X) * np.apply_along_axis(lambda args: dh[j](*args), 1, X))
        return L

    def orthogonalize_basis_MGS(X, h):
        x = self.x
        p = h
        Nm = len(p)
        for i in range(0, Nm):
            p[i] = p[i] / np.mean(sp.lambdify(x, p[i], 'numpy')(X) ** 2) ** 0.5
            for j in range(i + 1, Nm):
                project = np.mean(sp.lambdify(x, p[j], 'numpy')(X) * sp.lambdify(x, p[i], 'numpy')(X))
                p[j] = p[j] - project * p[i]
        return p

    # Find the index of the closest bin center and return the probability density at that bin
    def pdf_hist(self, x, hist, bin_edges, bin_centers):
      if isinstance(x, float):
        if x < bin_edges[0] or x > bin_edges[-1]:
          return 0.
        # Find the index of the closest bin center
        idx = np.argmin(np.abs(bin_centers - x))

        # Return the probability density at that bin
        return hist[idx]
      else:
        return np.array([pdf_hist(item, hist, bin_edges, bin_centers) for item in x])

    def mgf_statio(self, X_, ZZ, H_lambdify, tol, targ_grad=None, weights=None, max_steps=100):
        n_moms = len(H_lambdify)
        step = 0
        conv = False
        max_cond = 0
        momX = self.moments(H_lambdify, X_)  # , dim)
        samples_ = np.array([np.apply_along_axis(lambda args: H_lambdify[i](*args), 1, ZZ) for i in range(0, n_moms)]) - momX[:, None]

        if targ_grad is None:
            targ_grad = np.zeros(n_moms)

        if weights is None:  # or sum(weights) == 0:
            weights = np.ones(ZZ.shape[0])

        # initial lambda and corresponding likelihood ratios
        lam, lrs = np.zeros(n_moms), weights

        np.seterr(over='raise')
        lams_hist = [lam]
        grad_ = []
        grad_hist = []
        cond_hist = []
        while True:
            if sum(lrs) == 0:
                lrs = np.ones(ZZ.shape[0])
                lam *= 0
                break
            grad = np.average(samples_, weights=lrs, axis=1) - targ_grad
            grad_hist.append(np.linalg.norm(grad))
            grad_.append(np.average(samples_, weights=lrs, axis=1) - targ_grad)
            if all(np.absolute(grad) < tol) or step == max_steps:
                if step < max_steps:
                    conv = True
                break
            # hessian of MGF
            hess = np.zeros((n_moms, n_moms))
            for k, sample in enumerate(samples_):
                hess[k, k:] = np.average(sample * samples_[k:], weights=lrs, axis=1)
                hess[k:, k] = hess[k, k:]
            if np.linalg.cond(hess) > max_cond:
              max_cond = np.linalg.cond(hess)
            cond_hist.append(np.linalg.cond(hess))
            # Newton step
            try:
                new_lam = lam - solve(hess, grad)
            except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                print('Matching stopped: singular hessian')
                lam *= 0.
                lrs = np.ones_like(lrs)
                break
            # likelihood ratios corresponding
            # to new Lagrange multipliers
            if weights is None:
                weights = np.ones(ZZ.shape[0])
            try:
                lrs = np.exp(np.dot(new_lam, samples_) + np.log(weights))
            except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                print('Matching stopped: too big likelihood ratios')
                lam *= 0.
                lrs = np.ones_like(lrs)
                break

            # if no exceptions, accept new_lam
            lam = new_lam
            lams_hist.append(lam)
            step += 1

        return lam, lrs, conv, max_cond

    def Multi_Level_SDE(self, X, H, threshold_sample=1e-1, verbose=False):
        x = self.x
        dt = 1e-20
        tau = 100.  # 1e20
        N_t = len(X)

        dH = [[sp.diff(h, xx) for h in H] for xx in x]

        dH_new = [[sp.lambdify(x, dh, "numpy") for dh in dHH] for dHH in dH]
        start_time = time.time()
        lam = np.zeros_like(dH_new[0])

        lam_sum = np.zeros(len(lam))
        L = np.zeros((len(dH_new[0]), len(dH_new[0])))
        for dh_new in dH_new:
          L += self.Hess(X, dh_new)
        cond = np.linalg.cond(L)

        H_new = [sp.lambdify(x, h, "numpy") for h in H]
        d2H = [[sp.diff(dh, x[i]) for dh in dH[i]] for i in range(len(dH))]
        d2H_new = [[sp.lambdify(x, d2h, "numpy") for d2h in d2HH] for d2HH in d2H]

        momY0 = self.moments(H_new, X)  # , n=Nm)

        invL = np.linalg.inv(L)

        exponent = sum([ll * hh for ll, hh in zip(lam, H)])

        momY = self.moments(H_new, X)  # , n=Nm)
        dmom = momY0 - momY
        d2h = sum(np.array([[np.average(np.apply_along_axis(lambda args: d2h_new(*args), 1, X)) for d2h_new in d2HH_new] for d2HH_new in d2H_new]))

        b = -(dmom) / tau - d2h

        lam = invL @ b
        end_time = time.time()
        elapsed_time = end_time - start_time

        exponent = sum([ll * hh for ll, hh in zip(lam, H)])
        f = sp.exp(exponent)
        f_lambdify = sp.lambdify(x, f, 'numpy')
        Z = integrate.nquad(f_lambdify, [[np.min(X), np.max(X)]]*self.dim)[0] # MC_integral(f_lambdify, np.min(Y), np.max(Y)) # integrate.quad(f_lambdify, np.min(Y), np.max(Y))[0]
        # Z = MC_integral(f_lambdify, np.min(X), np.max(X))
        # f_g = lambda x: f_lambdify(x) / Z

        f_g_s = f/Z
        f_g = sp.lambdify(x, f_g_s, 'numpy')

        # f_g = sp.lambdify(x, f_g_s, 'numpy')

        return f_g, f_g_s, elapsed_time #cond

    def CrossEntropy(self, X, R_s, R, f_g, f_g_s, max_counter=10):
        x = self.x
        conv = False
        counter = 0
        while not conv:
            ZZ = self.multi_dim_uniform_sample(X)
            # ZZ = np.random.uniform(np.min(X), np.max(X), 10000)
            try:
                p = np.apply_along_axis(lambda args: f_g(*args), 1, ZZ)
                # p = f_g(ZZ)
            except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                max_cond = 0
                return f_g, f_g_s, ZZ, None, 1., max_cond
            indices = np.arange(len(p))
            idx = np.random.choice(indices, size=len(indices), p=p / sum(p))
            ZZ = ZZ[idx]

            lam, weights, conv, max_cond = self.mgf_statio(X, ZZ, R, tol=1e-9)
            counter += 1
            if counter > max_counter:
                break
        corrector = sp.exp(sum([ll * hh for ll, hh in zip(lam, R_s)]))

        f_g_s_new = f_g_s * corrector
        f_g_s_new_lambdify = sp.lambdify(x, f_g_s_new, 'numpy')

        Z = integrate.nquad(f_g_s_new_lambdify, [[np.min(X), np.max(X)]]*self.dim)[0]
        # Z = MC_integral(f_g_s_new_lambdify, np.min(X), np.max(X))
        f_g_s_new = f_g_s_new / Z
        f_g_new = sp.lambdify(x, f_g_s_new, 'numpy')
        # f_g_new = lambda x: f_g_s_new_lambdify(x) / Z
        return f_g_new, f_g_s_new, ZZ, weights, corrector, max_cond

    def get_pdf(self, X, N_iters, threshold_sample=1e-1, verbose=False):
        x = self.x
        dic = {}
        min_rel_err = np.inf
        R_s = self.generate_polynomials(self.dim, self.x_order)
        R = [sp.lambdify(x, r_s, "numpy") for r_s in R_s]
        for iter in range(N_iters):
          if iter == 0:
            H = self.generate_polynomials(self.dim, self.poly_order)
          else:
            H = [self.generate_symbolic_expr(depth=self.tree_depth) for _ in range(self.n_bases)]

          f_g, f_g_s, sde_cond = self.Multi_Level_SDE(X, H, threshold_sample, verbose=verbose)
        # return sde_cond   ## MOOS sde_cond here is the elapsed_time, so if you want to check the time just comment out this line and comment all lines after.
          f_g_new, f_g_s_new, samples, weights, corrector, closure_cond = self.CrossEntropy(X, R_s, R, f_g, f_g_s)
          rel_err = self.kl_div(X, f_g_new)
          dic[iter] = [f_g_new, f_g_s_new, rel_err, samples, weights, max(sde_cond, closure_cond)]
          if rel_err < min_rel_err:  # and rel_err_h < min_rel_err_h:
                min_rel_err = rel_err
                best_iter = iter
        return dic, best_iter
