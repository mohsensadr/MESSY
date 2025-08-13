import numpy as np
import sympy as sp
import random
from scipy import integrate
from scipy.linalg import solve, LinAlgError
import warnings
from sympy import oo, zoo, nan
warnings.filterwarnings("ignore")

x = sp.symbols('x', real=True)

## MESSY
constant_range=(1,3)
def cos_(x):
  integer = random.randint(*constant_range)
  half = integer + random.choice([0, 0.5])
  return sp.cos(sp.Rational(half)*x)
def sin_(x):
  integer = random.randint(*constant_range)
  half = integer + random.choice([0, 0.5])
  return sp.cos(sp.Rational(half)*x)

# def MC_integral(f, lb=-10,ub=10, N=10**5):
#   samples = np.random.uniform(lb, ub, (N, dim))
#   return np.mean(np.apply_along_axis(f, 1, samples)) * (ub - lb)**dim
class MESSY:
    def __init__(self, dim=1, highest_order=2, nb_l=2, nb_u=4, poly_order=4, tree_depth=2, binary_operators = [sp.Mul], unary_functions = pow):
        self.dim = dim
        self.highest_order = highest_order
        self.nb_l = nb_l
        self.nb_u = nb_u
        self.poly_order = poly_order
        self.tree_depth = tree_depth
        self.binary_operators = binary_operators
        self.unary_functions = unary_functions

    def moments(self, h, z, w=None):
      return np.array([np.average(hh(z), weights=w) for hh in h])

    def random_even_expr(self, depth=2, even=False):
        if depth == 0 or random.random() < 0.3:
            return x ** 2 if even else random.choice(self.unary_functions)(x)
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
            if sp.lambdify(x, expr, "numpy")(xmax) < xmax ** self.highest_order:
                done = True
        return expr

    def get_unique(self, H_s):
        unique_expr_set = set()
        for expr in H_s:
            expr_hashable = sp.sympify(expr).simplify().as_expr()
            unique_expr_set.add(expr_hashable)
        return list(unique_expr_set)

    def is_even(self, func):
        func_neg_var = func.subs(x, -x)
        return func.equals(func_neg_var)

    def make_fastest_even(self, H_s, xmax=10):
        x = sp.symbols('x', real=True)
        H = [np.vectorize(sp.lambdify(x, h_s, "numpy")) for h_s in H_s]
        H_rates = [h(xmax) for h in H]
        idx_fastest, max_rate = np.argmax(H_rates), np.max(H_rates)
        H_s[idx_fastest], H_s[-1] = H_s[-1], H_s[idx_fastest]
        if not self.is_even(H_s[-1]):
            power, done = 2, False
            while not done and power <= self.highest_order:
                new_basis_s = x ** power
                new_basis = np.vectorize(sp.lambdify(x, new_basis_s, "numpy"))
                if new_basis(xmax) >= max_rate:
                    done = True
                power += 2
            H_s[-1], H[-1] = new_basis_s, new_basis
        return H_s

    def create_basis(self, n_bases=6, tree_depth=2):
        x = sp.symbols('x', real=True)
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
            p[i] = p[i] / np.mean(sp.lambdify(x, p[i], 'numpy')(X) ** 2) ** 0.5
            for j in range(i + 1, Nm):
                project = np.mean(sp.lambdify(x, p[j], 'numpy')(X) * sp.lambdify(x, p[i], 'numpy')(X))
                p[j] = p[j] - project * p[i]
        return p

    def pdf_hist(self, x, hist, bin_edges, bin_centers):
        if isinstance(x, float):
            if x < bin_edges[0] or x > bin_edges[-1]:
                return 0.
            idx = np.argmin(np.abs(bin_centers - x))
            return hist[idx]
        else:
            return np.array([self.pdf_hist(item, hist, bin_edges, bin_centers) for item in x])

    def mgf_statio(self, X_, ZZ, H_lambdify, dim, tol, targ_grad=None, weights=None, max_steps=100):
        step = 0
        conv = False
        max_cond = 0
        momX = self.moments(H_lambdify, X_)  # , dim)
        samples_ = np.array([H_lambdify[i](ZZ) for i in range(0, dim)]) - momX[:, None]

        if targ_grad is None:
            targ_grad = np.zeros(dim)

        if weights is None:  # or sum(weights) == 0:
            weights = np.ones_like(ZZ)

        # initial lambda and corresponding likelihood ratios
        lam, lrs = np.zeros(dim), weights

        np.seterr(over='raise')
        lams_hist = [lam]
        grad_ = []
        grad_hist = []
        cond_hist = []
        while True:
            # gradient of MGF
            if sum(lrs) == 0:
                lrs = np.ones_like(ZZ)
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
            hess = np.zeros((dim, dim))
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
                weights = np.ones_like(ZZ)
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

        return lam, lrs, conv, max_cond  # , lams_hist, grad_, grad_hist, cond_hist

    import sys

    def check_exponent_overflow(self, expr, samples_, piecewise=False):  # , test_values=[np.min(Y), np.max(Y)]):
        if not piecewise:
            test_values = [np.min(samples_), np.max(samples_), np.min(samples_) - .5,
                           np.max(samples_) + .5]  # , np.min(samples_)-0.5, np.min(samples_)+0.5]
            for val in test_values:
                try:
                    result = expr.subs(x, val).evalf()/np.max(sp.lambdify(x,expr,'numpy')(samples_)) # for continuous
                    if abs(result) > 0.1: # for continuous
                    #result = expr.subs(x, val).evalf()  # for discontinous
                    #if abs(result) < 10 ** 6:  # for discontinous ---> we don't need, because we should not check for piecewise
                        return True
                except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                    return True
            return False
        else:
            try:
                result = np.max(sp.lambdify(x, expr, 'numpy')(samples_))
                if abs(result) > 1e2:
                    return True
            except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                return True
            return False

    def vectorized_piecewise_evaluator(self, expr, input_array):
        result = np.zeros_like(input_array, dtype=float)
        for e, c in expr.args:
            if c == True:
                continue
            cond = sp.lambdify(x, c, 'numpy')(input_array)
            if isinstance(cond, np.ndarray):
                result[cond == True] = sp.lambdify(x, e, 'numpy')(input_array[cond == True])
            else:
                if cond == True:
                    result = sp.lambdify(x, e, 'numpy')(input_array)
        return result

    def weighted_sum_piecewise_evaluator(self, weight_list, expr_list, input_array):
        res = np.zeros_like(input_array)
        for i, expr in enumerate(expr_list):
            res += weight_list[i] * self.vectorized_piecewise_evaluator(expr, input_array)
        return res

    def Multi_Level_SDE(self, X, N_iters=10, threshold_sample=1e-1, poly_bases=False, verbose=False):
        flag = False
        dim = self.dim
        while not flag:
            Nt = 1 #20
            dt = 1e-20
            tau = 100.  # 1e20
            list_pdf, list_mass = [], []
            w = np.ones_like(X, dtype=int)
            Y = X.copy()
            N_t = len(X)
            for k in range(N_iters):
                if verbose:
                    print('Level:', k)
                # rem_mass, pdf_mass, rem_w = find_pdf(Y,w)

                Y = np.array([j for i, j in enumerate(Y) if w[i] == 1])
                Y0 = Y.copy()
                w = np.ones_like(Y, dtype=int)

                done = False
                while not done:
                    while True:
                        ## construct basis function for the data
                        if not poly_bases:
                            OKBasis = False
                            while OKBasis is not True:
                                H = self.create_basis(random.randint(self.nb_l, self.nb_u), self.tree_depth)
                                H = list(set(H))
                                dH = [sp.diff(h, x) for h in H]
                                dH_new = [sp.lambdify(x, dh, "numpy") for dh in dH]
                                L = self.Hess(Y, dH_new)
                                # print("cond of test basis:", np.linalg.cond(L))
                                if np.linalg.cond(L) < 1e15:
                                    OKBasis = True
                        else:
                            H = [x ** i for i in range(1, self.poly_order + 1)]
                            # H = [x1 x2 x1^2 x2^2 x1x2]

                        dH = [sp.diff(h, x) for h in H]
                            # dH = [[1 0 2x1 0 x2] for dim 1,
                                   #[0 1 0 2x2 x1] for dim 2]    resulting_dH = [1 1 2x1 2x2 x1+x2]



                            # d2H = [[0 0 2 0 0],
                            #        [0 0 0 2 0]]

                            # d2H [[0 0 2 2 0]]
                        dH = self.orthogonalize_basis_MGS(Y, dH)
                        H = [sp.integrate(dh, x) for dh in dH]
                        dH_new = [sp.lambdify(x, dh, "numpy") for dh in dH]

                        lam = np.zeros_like(dH_new)

                        lam_sum = np.zeros(len(lam))
                        L = self.Hess(Y, dH_new)

                        ## call Hess(Y, dh[1])

                        cond = np.linalg.cond(L)
                        break
                        try:
                            if np.linalg.cond(L) < 10.:
                                break
                        except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                            print('large condition number')
                            break
                    H_new = [sp.lambdify(x, h, "numpy") for h in H]
                    d2H = [sp.diff(dh, x) for dh in dH]
                    d2H_new = [sp.lambdify(x, d2h, "numpy") for d2h in d2H]

                    momY0 = self.moments(H_new, Y)  # , n=Nm)

                    invL = np.linalg.inv(L)

                    exponent = sum([ll * hh for ll, hh in zip(lam, H)])
                    for i in range(Nt):

                        momY = self.moments(H_new, Y)  # , n=Nm)
                        dmom = momY0 - momY
                        d2h = np.array([np.average(d2h_new(Y)) for d2h_new in d2H_new])

                        b = -(dmom) / tau - d2h

                        lam = invL @ b

                        lam_sum += lam
                        for i in range(0):
                            A = np.dot(lam, np.array([dh_new(Y) for dh_new in dH_new]))
                            dW = np.random.normal(0., 1., len(Y))
                            Y = Y + A * dt + (2. * dt) ** 0.5 * dW
                            momY = self.moments(H_new, Y)  # , n=Nm)
                            dmom = momY0 - momY
                        exponent = sum([ll * hh for ll, hh in zip(lam, H)])  # sum([lam[i] * H[i] for i in range(Nm)])
                    lam = lam_sum / Nt
                    Y = Y0
                    exponent = sum([ll * hh for ll, hh in zip(lam, H)])
                    f = sp.exp(exponent)
                    if not (f.has(oo, -oo, zoo, nan) or self.check_exponent_overflow(f, Y, piecewise = False)) or poly_bases:
                        done = True
                    f_lambdify = sp.lambdify(x, f, 'numpy')
                try:
                    Z = integrate.nquad(f_lambdify, [[np.min(Y), np.max(Y)]]*dim)[0] # MC_integral(f_lambdify, np.min(Y), np.max(Y)) # integrate.quad(f_lambdify, np.min(Y), np.max(Y))[0]
                    # Z = MC_moments_from_pdf(f_lambdify, [lambda x: x**0], np.random.uniform(np.min(Y),np.max(Y),10000))[0]
                except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                    print('Z quad overflow')
                    break
                f_g = lambda x: f_lambdify(x) / Z

                hist, bin_edges = np.histogram(Y, bins=100, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                for i, y in enumerate(Y):
                    w[i] = 0 if np.random.rand() < f_g(y) / self.pdf_hist(y, hist, bin_edges, bin_centers) else 1

                if sum(list_mass) > 1. - threshold_sample or k == N_iters - 1:  # or sum(w==0) / N_t < threshold_sample:
                    w = np.zeros_like(Y)

                mass = sum(w == 0) / N_t

                if mass < threshold_sample:
                    f = sp.exp(-(x) ** 2 / (2 * np.var(Y)))  # /(2*sp.pi*np.var(Y))**0.5 -np.mean(Y)
                    f_lambdify = sp.lambdify(x, f, 'numpy')
                    f_g = lambda x: f_lambdify(x) / (2 * np.pi * np.var(Y)) ** 0.5
                    for i, y in enumerate(Y):
                        w[i] = 0 if np.random.rand() < f_g(y) / self.pdf_hist(y, hist, bin_edges, bin_centers) else 1
                    # if sum(list_mass) > 1.-threshold_sample or k == N_iters-1: # or sum(w==0) / N_t < threshold_sample:
                    #   w = np.zeros_like(Y)
                    mass = sum(w == 0) / N_t
                list_mass.append(mass)

                Y_loc = np.array([j for i, j in enumerate(Y) if w[i] == 0])

                Z = integrate.nquad(f_lambdify, [[np.min(Y_loc), np.max(Y_loc)]]*dim)[0] #MC_integral(f_lambdify, np.min(Y_loc), np.max(Y_loc)) # integrate.quad(f_lambdify, np.min(Y_loc), np.max(Y_loc))[0]
                list_pdf.append(f / Z)

                if sum(list_mass) > 1. - threshold_sample or len(w[w == 1]) < 20 or mass < threshold_sample:
                    break
            f_g_s = sum([mass * f for mass, f in zip(list_mass, list_pdf)])

            if not (f_g_s.has(oo, -oo, zoo, nan) or self.check_exponent_overflow(f_g_s, Y_loc, piecewise = False)) or poly_bases:
                flag = True
            if not poly_bases and (f_g_s.has(oo, -oo, zoo, nan) or self.check_exponent_overflow(f_g_s, Y_loc, piecewise = False)):
                if verbose:
                    print("\nRepeat!")

        f_g = sp.lambdify(x, f_g_s, 'numpy')

        return f_g, f_g_s, list_mass, list_pdf, cond

    def CrossEntropy(self, X, H, H_new, f_g, f_g_s, Nm_xe, list_mass, list_pdf, poly_bases=False, piecewise=False, max_counter=10):
        done = False
        dim = self.dim
        n_m = Nm_xe

        while not done:
            conv = False
            counter = 0
            while not conv:
                ZZ = np.random.uniform(np.min(X), np.max(X), 10000)
                try:
                    p = f_g(ZZ)
                except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                    max_cond = 0
                    return f_g, f_g_s, ZZ, None, 1., max_cond
                ZZ = np.random.choice(ZZ, size=len(ZZ), p=p / sum(p))

                lam, weights, conv, max_cond = self.mgf_statio(X, ZZ, H_new, n_m, tol=1e-9)
                counter += 1
                if counter > max_counter:
                    break
            corrector = sp.exp(sum([ll * hh for ll, hh in zip(lam, H)]))
            # f_g_s_new = sp.Piecewise(*[(expr * corrector, cond) for expr, cond in f_g_s.args])
            # display(f_g_s_new)
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

            if (f_g_s_new.has(oo, -oo, zoo, nan) or self.check_exponent_overflow(f_g_s_new, X, piecewise)):
                return f_g, f_g_s, ZZ, None, 1., max_cond

            if piecewise:
                f_g_s_new_lambdify = lambda x: self.weighted_sum_piecewise_evaluator(list_mass, piece_list, x)
            else:
                f_g_s_new_lambdify = sp.lambdify(x, f_g_s_new, 'numpy')
            if True:  # not poly_bases:
                try:
                    Z = integrate.nquad(f_g_s_new_lambdify, [[np.min(X), np.max(X)]]*dim)[0] #MC_integral(f_g_s_new_lambdify, np.min(X), np.max(X)) # integrate.quad(f_g_s_new_lambdify, np.min(X), np.max(X))[0]
                    done = True
                except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                    n_m -= 2
                    break
            else:
                Z = integrate.nquad(f_g_s_new_lambdify, [[np.min(X), np.max(X)]]*dim)[0] #MC_integral(f_g_s_new_lambdify, np.min(X), np.max(X)) # integrate.quad(f_g_s_new_lambdify, np.min(X), np.max(X))[0]
                done = True
        f_g_s_new = f_g_s_new / Z
        f_g_new = lambda x: f_g_s_new_lambdify(x) / Z
        return f_g_new, f_g_s_new, ZZ, weights, corrector, max_cond


    def get_pdf(self, X, N_iters=10, Nm_xe=2, N_levels=3, threshold_sample=1e-1, left=False, right=False, verbose=False):
        min_rel_err, min_rel_err_h = np.inf, np.inf
        dic = {}
        ii = 0
        R_low = [x ** i for i in range(1, Nm_xe + 1)]
        R_low_ = [sp.lambdify(x, r_s, "numpy") for r_s in R_low]
        X_moms_low = self.moments(R_low_, X)
        piecewise = left or right
        while ii < N_iters:
            if verbose:
                print('Round:', ii)
            f_g, f_g_s, list_mass, list_pdf, sde_cond = self.Multi_Level_SDE(X, N_levels, threshold_sample, poly_bases=(ii == 0), verbose=verbose)

            if left or right:
              if left and right:
                f_g_s = sp.Piecewise((f_g_s, (x >= np.min(X)) & (x <= np.max(X))), (0, True))
                for i, pdf_ in enumerate(list_pdf):
                  list_pdf[i] = sp.Piecewise((pdf_, (x >= np.min(X)) & (x <= np.max(X))), (0, True))
              elif left:
                f_g_s = sp.Piecewise((f_g_s, x >= np.min(X)), (0, True))
                for i, pdf_ in enumerate(list_pdf):
                  list_pdf[i] = sp.Piecewise((pdf_, x >= np.min(X)), (0, True))
              else:
                f_g_s = sp.Piecewise((f_g_s, x <= np.max(X)), (0, True))
                for i, pdf_ in enumerate(list_pdf):
                  list_pdf[i] = sp.Piecewise((pdf_, x <= np.max(X)), (0, True))

              f_g  = sp.lambdify(x, f_g_s, 'numpy')

            f_g_new, f_g_s_new, samples, weights, corrector, closure_cond = self.CrossEntropy(X, R_low, R_low_, f_g,
                                                                                         f_g_s, Nm_xe, list_mass,
                                                                                         list_pdf, poly_bases=(ii == 0),
                                                                                         piecewise=piecewise)

            if ii != 0 and (f_g_s_new.has(oo, -oo, zoo, nan) or self.check_exponent_overflow(f_g_s_new, X, piecewise)):
                continue
            mom_low, mom_high = Nm_xe+1, Nm_xe + 10

            R_s = [x ** i for i in range(mom_low, mom_high)]
            R = [sp.lambdify(x, r_s, "numpy") for r_s in R_s]
            if weights is None:  # if cross entropy didn't work out
                ss = np.random.uniform(np.min(X) - np.std(X), np.max(X) + np.std(X), 10000)
                try:
                    wghts = f_g_new(ss) / sum(f_g_new(ss))
                except (LinAlgError, ValueError, OverflowError, FloatingPointError):
                    continue
            else:
                ss = samples
                wghts = weights

            # hist, bin_edges = np.histogram(X, bins=70, density=True) #np.histogram(X, bins='doane', density=True)
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # hist_pdf = np.vectorize(
            #     lambda x: 0. if x < bin_edges[0] or x > bin_edges[-1] else hist[np.argmin(np.abs(bin_centers - x))])
            # rel_err = KL_Divergence(hist_pdf, f_g_new, np.min(X), np.max(X)) + KL_Divergence(f_g_new, hist_pdf, np.min(X), np.max(X))

            rel_err = self.kl_div(X,f_g_new)

            dic[ii] = [f_g_new, f_g_s_new, rel_err, ss, wghts, max(sde_cond, closure_cond)]
            if rel_err < min_rel_err:
                min_rel_err = rel_err
                best_iter = ii
            ii += 1
            if verbose:
                print('\n')
        if verbose:
            print('best_iter:', best_iter)
        return dic, best_iter

    def kl_div(self, XX, f, w=None):
      return - np.average(np.log(f(XX) + 1e-10), weights=w)

    def KL_Divergence(self, true_dist, pred_dist, xmin, xmax):
      xx = np.linspace(xmin,xmax,10000)
      p, q = true_dist(xx), pred_dist(xx)
      p = np.asarray(p, dtype=np.float)
      q = np.asarray(q, dtype=np.float)
      #return np.sum(np.where(p/q > 1e-10, p * np.log( p / (q+1e-10) + 1e-10), 0))#/len(p);
      return np.sum(p * np.log( p / (q+1e-10) + 1e-10))
