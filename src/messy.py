import numpy as np
from numpy.linalg import norm
import sympy as sp
from sympy import oo, zoo, nan
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.linalg import solve, LinAlgError
from scipy.special import rel_entr
import pandas as pd
from sklearn.neighbors import KernelDensity
import statistics as stat
from collections import defaultdict
import warnings
import random
import time
import sys
from scipy.special import fresnel, gamma, hyp2f1

class MESSY:
    def __init__(self, n_levels=2, nm_xe=2, highest_order=2, nb_l=2, nb_u=4, poly_order=4, tree_depth=2, binary_operators = [sp.Mul], unary_functions = pow):
        self.x = sp.symbols('x', real=True)
        self.highest_order = highest_order
        self.nb_l = nb_l
        self.nb_u = nb_u
        self.poly_order = poly_order
        self.tree_depth = tree_depth
        self.binary_operators = binary_operators
        self.unary_functions = unary_functions
        self.n_levels = n_levels
        self.nm_xe = nm_xe

    def np_lambdify(self, varname, func):
        lamb = sp.lambdify(varname, func, modules=['numpy'])
        if func.is_constant():
            return lambda t: np.full_like(t, lamb(t))
        else:
            return lambda t: lamb(np.array(t))

    def moments(self, h, z, w=None):
        return np.array([np.average(hh(z), weights=w) for hh in h])

    def KL_Divergence(self, true_dist, pred_dist, xmin, xmax):
        xx = np.linspace(xmin, xmax, 10000)
        p, q = true_dist(xx), pred_dist(xx)
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        return np.sum(np.where(q > 1e-10, p * np.log(p / q), 0))

    def random_even_expr(self, depth=2, even=False):
        if depth == 0 or random.random() < 0.3:
            return self.x**2 if even else random.choice(self.unary_functions)(self.x)
        if random.random() < 0.7:
            op = random.choice(self.binary_operators)
            left = self.random_even_expr(depth - 1, even)
            right = self.random_even_expr(depth - 1, even)
            return op(left, right)
        else:
            return random.choice(self.unary_functions)(self.random_even_expr(depth - 1, even))

    def random_tree(self, depth=2, even=False, xmax=10):
        done = False
        while not done:
            expr = self.random_even_expr(depth, even)
            if sp.lambdify(self.x, expr, 'numpy')(xmax) < xmax ** self.highest_order:
                done = True
        return expr

    def get_unique(self, H_s):
        unique_expr_set = set()
        for expr in H_s:
            expr_hashable = sp.sympify(expr).simplify().as_expr()
            unique_expr_set.add(expr_hashable)
        return list(unique_expr_set)

    def is_even(self, func):
        func_neg_var = func.subs(self.x, -self.x)
        return func.equals(func_neg_var)

    def make_fastest_even(self, H_s, xmax=10):
        H = [np.vectorize(sp.lambdify(self.x, h_s, 'numpy')) for h_s in H_s]
        H_rates = [h(xmax) for h in H]
        idx_fastest, max_rate = np.argmax(H_rates), np.max(H_rates)
        H_s[idx_fastest], H_s[-1] = H_s[-1], H_s[idx_fastest]
        if not self.is_even(H_s[-1]):
            power, done = 2, False
            while not done and power <= self.highest_order:
                new_basis_s = self.x ** power
                new_basis = np.vectorize(sp.lambdify(self.x, new_basis_s, 'numpy'))
                if new_basis(xmax) >= max_rate:
                    done = True
                power += 2
            H_s[-1], H[-1] = new_basis_s, new_basis
        return H_s

    def create_basis(self, n_bases=6, tree_depth=2):
        done = False
        while not done:
            H_s = [self.random_tree(depth=tree_depth, even=random.choice([True, False])) for _ in range(n_bases)]
            H_s = self.make_fastest_even(H_s)
            if len(self.get_unique(H_s)) == n_bases:
                done = True
        return H_s

    def Hess(self, X, dh):
        Nm = len(dh)
        L = np.zeros((Nm, Nm))
        for i in range(Nm):
            for j in range(Nm):
                L[i, j] = np.mean(dh[i](X) * dh[j](X))
        return L

    def orthogonalize_basis_MGS(self, X, h):
        p = h
        Nm = len(p)
        for i in range(0, Nm):
            p[i] = p[i] / np.mean(sp.lambdify(self.x, p[i], 'numpy')(X) ** 2) ** 0.5
            for j in range(i + 1, Nm):
                project = np.mean(sp.lambdify(self.x, p[j], 'numpy')(X) * sp.lambdify(self.x, p[i], 'numpy')(X))
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

    def mgf_statio(self, X_, ZZ, H_lambdify, dim, tol, targ_grad=None, weights=None, max_steps=1000):
        step = 0
        conv = False

        momX = self.moments(H_lambdify, X_)#, dim)
        samples_ = np.array([H_lambdify[i](ZZ) for i in range(0, dim)]) - momX[:,None]

        if targ_grad is None:
          targ_grad = np.zeros(dim)

        if weights is None:# or sum(weights) == 0:
          weights = np.ones_like(ZZ)

        # initial lambda and corresponding likelihood ratios
        lam, lrs = np.zeros(dim), weights

        np.seterr(over='raise')
        lams_hist = [lam]
        grad_ = []
        grad_hist = []
        while True:
          # gradient of MGF
          if sum(lrs) == 0:
            lrs = np.ones_like(ZZ)
            lam *= 0
            break
          grad = np.average(samples_, weights=lrs , axis=1) - targ_grad
          grad_hist.append(np.linalg.norm(grad))
          grad_.append(np.average(samples_, weights=lrs , axis=1) - targ_grad)
          if all(np.absolute(grad) < tol) or step == max_steps:
              if step < max_steps:
                  conv = True
              break
          # hessian of MGF
          hess = np.zeros((dim, dim))
          for k, sample in enumerate(samples_):
            hess[k, k:] = np.average(sample * samples_[k:], weights=lrs , axis=1)
            hess[k:, k] = hess[k, k:]
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
            weights = np.ones_like(ZZ)
          try:
            lrs  = np.exp(np.dot(new_lam, samples_) + np.log(weights))
          except (LinAlgError, ValueError, OverflowError, FloatingPointError):
            print('Matching stopped: too big likelihood ratios')
            lam *= 0.
            lrs = np.ones_like(lrs)
            break

          # if no exceptions, accept new_lam
          lam = new_lam
          lams_hist.append(lam)
          step += 1

        return lam, lrs, conv

    def check_exponent_overflow(self, expr, samples):#, test_values=[np.min(Y), np.max(Y)]):
      x = self.x
      test_values = [np.min(samples), np.max(samples), np.min(samples)-.5, np.max(samples)+.5]#, np.min(samples)-0.5, np.min(samples)+0.5]
      for val in test_values:
        try:
          result = expr.subs(x, val).evalf() # for discontinous
          if abs(result) < 10**6: # for discontinous
            return True
        except (LinAlgError, ValueError, OverflowError, FloatingPointError):
          return True
      return False

    def vectorized_piecewise_evaluator(self, expr, input_array):
      x = self.x
      result = np.zeros_like(input_array, dtype=float)
      for e, c in expr.args:
        if c == True:
          continue
        cond = sp.lambdify(x, c, 'numpy')(input_array)
        if isinstance(cond,np.ndarray):
          result[cond==True] = sp.lambdify(x, e, 'numpy')(input_array[cond==True])
        else:
          if cond == True:
            result = sp.lambdify(x, e, 'numpy')(input_array)
      return result

    def weighted_sum_piecewise_evaluator(self, weight_list, expr_list, input_array):
      res = np.zeros_like(input_array)
      for i,expr in enumerate(expr_list):
        res += weight_list[i]*self.vectorized_piecewise_evaluator(expr, input_array)
      return res


    def Z_MC(self, X, f_lam, minX, maxX):
        L = maxX-minX
        dx = L / len(X)
        return np.sum(f_lam(X)) * dx

    def Multi_Level_SDE(self, X, threshold_sample=1e-1, poly_bases=False, verbose = False, piecewise=False):
      flag = False
      x = self.x
      n_levels = self.n_levels
      while not flag:
        Nt = 20
        dt = 1e-20
        tau = 100. # 1e20
        list_pdf, list_mass = [], []
        w = np.ones_like(X, dtype=int)
        Y = X.copy()
        N_t = len(X)
        for k in range(n_levels):
          #if verbose:
          print('Level:', k)

          Y = np.array([j for i,j in enumerate(Y) if w[i] == 1])
          Y0 = Y.copy()
          w = np.ones_like(Y, dtype=int)

          done = False
          while not done:
            while True:
              ## construct basis function for the data
              if not poly_bases:
                OKBasis = False
                while OKBasis is not True:
                  H = self.create_basis(random.randint(self.nb_l,self.nb_u), self.tree_depth)
                  H = list(set(H))
                  dH = [sp.diff(h, x) for h in H]
                  dH_new  = [sp.lambdify(x, dh, "numpy") for dh in dH]
                  L = self.Hess(Y, dH_new)
                  print("cond(L)", np.linalg.cond(L))
                  if np.linalg.cond(L) < 10:
                    OKBasis = True
              else:
                H = [x**i for i in range(1, self.poly_order + 1)]
              print('generate a basis')
              dH = [sp.diff(h, x) for h in H]
              dH = self.orthogonalize_basis_MGS(Y, dH)
              H  = [sp.integrate(dh, x) for dh in dH]

              dH_new  = [self.np_lambdify(x, dh) for dh in dH]

              lam   = np.zeros_like(dH_new)

              lam_sum = np.zeros(len(lam))
              L = self.Hess(Y, dH_new)
              print("cond(L)", np.linalg.cond(L), " after orthogonalization")
              break
              try:
                if np.linalg.cond(L) < 10000.:
                  break
              except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                print('large condition number')
                break
            print('found a good basis')
            H = [h.doit() for h in H]  # Evaluate the integrals
            H_new   = [sp.lambdify(x, h, "numpy") for h in H]
            d2H     = [sp.diff(dh, x) for dh in dH]
            d2H_new = [sp.lambdify(x, d2h, "numpy") for d2h in d2H]

            momX  = self.moments(H_new, X)#, n=Nm)
            momY0 = self.moments(H_new, Y)#, n=Nm)

            invL = np.linalg.inv(L)

            exponent = sum([ll * hh for ll,hh in zip(lam,H)])
            for i in range(Nt):

              momY = self.moments(H_new, Y)#, n=Nm)
              dmom = momX - momY
              d2h = np.array([np.average(d2h_new(Y)) for d2h_new in d2H_new])

              b = -(dmom) / tau - d2h

              lam = invL @ b

              lam_sum += lam
              for i in range(10):
                A = np.dot(lam, np.array([dh_new(Y) for dh_new in dH_new]))
                dW = np.random.normal(0., 1., len(Y))
                Y = Y + A * dt + (2. * dt) ** 0.5 * dW
                momY = self.moments(H_new, Y)#, n=Nm)
                dmom = momX - momY
              exponent = sum([ll * hh for ll, hh in zip(lam,H)]) #sum([lam[i] * H[i] for i in range(Nm)])
            lam = lam_sum / Nt
            Y = Y0
            exponent = sum([ll * hh for ll, hh in zip(lam,H)])
            f = sp.exp(exponent)
            print(f)
            #if not (f.has(oo, -oo, zoo, nan) or check_exponent_overflow(f,X)) or poly_bases:
            #  print('done')
            #  done = True
            done = True
            f_lambdify = sp.lambdify(x, f,'numpy')
          try:
            Z = integrate.quad(f_lambdify, np.min(Y), np.max(Y))[0]
            #Z = self.Z_MC(Y, f_lambdify, np.min(Y), np.max(Y))
            #Z = MC_moments_from_pdf(f_lambdify, [lambda x: x**0], np.random.uniform(np.min(Y),np.max(Y),1000))[0]
          except (LinAlgError, ValueError, OverflowError, FloatingPointError):
             print('Z integral overflow')
             break
          f_g = lambda x: f_lambdify(x) / Z

          # Compute histogram and bin edges
          hist, bin_edges = np.histogram(Y, bins=100, density=True)

          # Compute bin centers
          bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

          for i, y in enumerate(Y):
            w[i] = 0 if np.random.rand() < f_g(y)/self.pdf_hist(y, hist, bin_edges, bin_centers) else 1

          if sum(list_mass) > 1.-threshold_sample or k == n_levels-1:# or sum(w==0) / N_t < threshold_sample:
            w = np.zeros_like(Y)

          mass = sum(w==0) / N_t

          if mass < threshold_sample:
            f = sp.exp(-(x)**2/(2*np.var(Y)))#/(2*sp.pi*np.var(Y))**0.5 -np.mean(Y)
            f_lambdify = sp.lambdify(x, f,'numpy')
            f_g = lambda x: f_lambdify(x) / (2*np.pi*np.var(Y))**0.5
            for i, y in enumerate(Y):
              w[i] = 0 if np.random.rand() < f_g(y)/self.pdf_hist(y, hist, bin_edges, bin_centers) else 1
            mass = sum(w==0) / N_t
          # print(mass)
          list_mass.append(mass)

          Y_loc = np.array([j for i,j in enumerate(Y) if w[i] == 0])

          #Z = MC_moments_from_pdf(f_lambdify, [lambda x: x**0], np.random.uniform(np.min(Y_loc),np.max(Y_loc),1000))[0]
          Z = integrate.quad(f_lambdify, np.min(Y_loc), np.max(Y_loc))[0]
          #Z = self.Z_MC(Y_loc, f_lambdify, np.min(Y_loc), np.max(Y_loc))

          if piecewise:
            list_pdf.append(sp.Piecewise((f/Z, (x>=np.min(Y_loc)) & (x<=np.max(Y_loc))), (0, True)))
          else:
            list_pdf.append(f/Z)

          if sum(list_mass) > 1.-threshold_sample or len(w[w==1])<20 or mass < threshold_sample: # we cannot recover distribution when there are not enough samples
            break
          # print(f/Z)
          # print(mass)
        f_g_s = sum([mass*f for mass,f in zip(list_mass, list_pdf)])

        flag = True
        #if not (f_g_s.has(oo, -oo, zoo, nan) or check_exponent_overflow(f_g_s, Y)) or poly_bases:
        #  print('flag true')
        #  flag = True
        #if not poly_bases and (f_g_s.has(oo, -oo, zoo, nan) or check_exponent_overflow(f_g_s, Y)):
        #  if verbose:
        #    print("\nRepeat!, found inf, nan on pdf values")

      f_g = sp.lambdify(x, f_g_s, 'numpy')

      return f_g, f_g_s, list_mass, list_pdf

    def CrossEntropy(self, X,H,H_new,f_g,f_g_s,list_mass,list_pdf,poly_bases=False, piecewise=False):
      x = self.x
      done = False
      n_m = self.nm_xe
      while not done:
        conv = False
        while not conv:
          ZZ = np.random.uniform(np.min(X),np.max(X), 10000)
          try:
            p = f_g(ZZ)
          except (LinAlgError, ValueError, OverflowError, FloatingPointError):
            return f_g, f_g_s, ZZ, None, 1.
          ZZ = np.random.choice(ZZ, size=len(ZZ), p=p/sum(p))
          lam, weights, conv = self.mgf_statio(X, ZZ, H_new, n_m, tol=1e-14)

        corrector = sp.exp(sum([ll * hh for ll, hh in zip(lam,H)]))
        f_g_s_new = 0
        if piecewise:
          piece_list = []
          for i in range(len(list_mass)):
            piece = sp.Piecewise(*[(expr * corrector, cond) for expr, cond in list_pdf[i].args])
            piece_list.append(piece)
            f_g_s_new += list_mass[i] * piece
        else:
          for i in range(len(list_mass)):
            f_g_s_new += list_mass[i] * list_pdf[i] * corrector

        if (f_g_s_new.has(oo, -oo, zoo, nan) or self.check_exponent_overflow(f_g_s_new, ZZ)):
          return f_g, f_g_s, ZZ, None, 1.

        if piecewise:
          f_g_s_new_lambdify = lambda x: self.weighted_sum_piecewise_evaluator(list_mass, piece_list, x)
        else:
          f_g_s_new_lambdify = sp.lambdify(x, f_g_s_new, 'numpy')
        if True: #not poly_bases:
          try:
            #Z = MC_moments_from_pdf(f_g_s_new_lambdify, [lambda x: x**0], np.random.uniform(np.min(X),np.max(X),1000))[0]
            Z = integrate.quad(f_g_s_new_lambdify, np.min(X), np.max(X))[0]
            #Z = self.Z_MC(X, f_g_s_new_lambdify, np.min(X), np.max(X))
            done = True
          except (LinAlgError, ValueError, OverflowError, FloatingPointError):
            n_m -= 2
            break
        else:
          #Z = MC_moments_from_pdf(f_g_s_new_lambdify, [lambda x: x**0], np.random.uniform(np.min(X),np.max(X),1000))[0]
          Z = integrate.quad(f_g_s_new_lambdify, np.min(X), np.max(X))[0]
          #Z = self.Z_MC(X, f_g_s_new_lambdify, np.min(X), np.max(X))
          done = True
      f_g_new = lambda x: f_g_s_new_lambdify(x) / Z
      return f_g_new, f_g_s_new, ZZ, weights, corrector

    def get_pdf(self, X, N_iters=5, threshold_sample=1e-1, verbose=False, piecewise=False):
      min_rel_err, min_rel_err_h = np.inf, np.inf
      dic = {}
      i = 0
      R_low = [self.x**i for i in range(1, self.nm_xe+1)]
      R_low_ = [sp.lambdify(self.x,r_s,"numpy") for r_s in R_low]
      X_moms_low = self.moments(R_low_, X)
      while i < N_iters:
        #if verbose:
        print('\n Round:', i+1, " out of ", N_iters, "\n")
        f_g, f_g_s, list_mass, list_pdf = self.Multi_Level_SDE(X, threshold_sample, poly_bases=(i==0), verbose=verbose, piecewise=piecewise)
        print('SDE done, Starting MxED')
        f_g_new, f_g_s_new, samples, weights, corrector = self.CrossEntropy(X,R_low, R_low_,f_g,f_g_s,list_mass,list_pdf,poly_bases=(i==0),piecewise=piecewise)
        print('MxED done')
        #if i != 0 and (f_g_s_new.has(oo, -oo, zoo, nan) or check_exponent_overflow(f_g_s_new, X)):
        #  continue
        #print('Passed Checking!')
        mom_low, mom_high = 1, self.nm_xe+3

        R_s = [self.x**i for i in range(mom_low, mom_high)]
        R = [sp.lambdify(self.x,r_s,"numpy") for r_s in R_s]
        if weights is None: # if cross entropy didn't work out
          ss = np.random.uniform(np.min(X)-np.std(X),np.max(X)+np.std(X), 10000)
          try:
            wghts = f_g_new(ss)/sum(f_g_new(ss))
          except (LinAlgError, ValueError, OverflowError, FloatingPointError):
            continue
        else:
          ss = samples
          wghts = weights

        est_moms = self.moments(R, ss, w=wghts)
        X_moms = self.moments(R, X)
        rel_err = np.linalg.norm(est_moms-X_moms)/np.linalg.norm(X_moms)
        dic[i] = [f_g_new, f_g_s_new, rel_err, ss, wghts]
        if rel_err < min_rel_err:
          min_rel_err = rel_err
          best_iter = i
        i+=1
        if verbose:
          print('Done with Round:', i, " out of ", N_iters, "\n")
      if verbose:
        print('best_iter:', best_iter)
      return dic, best_iter

