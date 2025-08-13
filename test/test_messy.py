# /tests/test_messy.py
import pytest
import numpy as np
import sympy as sp
from src import messy

x = sp.symbols('x', real=True)

# Fixture for MESSY instance
#@pytest.fixture
#def messy_instance():
#    return messy.MESSY(dim=1, highest_order=2, nb_l=2, nb_u=4, poly_order=4, tree_depth=2)

def test_cos_sin_output_type():
    # Test cos_ and sin_ functions return sympy expressions
    c = messy.cos_(x)
    s = messy.sin_(x)
    assert isinstance(c, sp.Basic)
    assert isinstance(s, sp.Basic)

def test_moments_basic(messy_instance):
    h = [lambda z: z**2, lambda z: z+1]
    z = np.array([1., 2., 3.])
    result = messy_instance.moments(h, z)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [np.mean(z**2), np.mean(z+1)])
'''
def test_is_even(messy_instance):
    f_even = x**2
    f_odd = x**3
    assert messy_instance.is_even(f_even) is True
    assert messy_instance.is_even(f_odd) is False

def test_get_unique(messy_instance):
    H_s = [x**2, x**2, x**3]
    unique = messy_instance.get_unique(H_s)
    assert len(unique) == 2
    assert all(isinstance(u, sp.Basic) for u in unique)

def test_random_tree_and_basis(messy_instance):
    expr = messy_instance.random_tree(depth=2)
    assert isinstance(expr, sp.Basic)
    basis = messy_instance.create_basis(n_bases=3, tree_depth=1)
    assert len(basis) == 3
    assert all(isinstance(b, sp.Basic) for b in basis)

def test_Hess_and_orthogonalization(messy_instance):
    X = np.linspace(-1,1,5)
    h_exprs = [x, x**2]
    dh = [sp.lambdify(x, sp.diff(hh, x), 'numpy') for hh in h_exprs]
    H = messy_instance.Hess(X, dh)
    assert H.shape == (2,2)
    ortho = messy_instance.orthogonalize_basis_MGS(X, h_exprs)
    assert len(ortho) == 2
    assert all(isinstance(o, sp.Basic) for o in ortho)

def test_pdf_hist(messy_instance):
    data = np.array([0.1, 0.5, 0.9])
    hist, bins = np.histogram(data, bins=3, density=True)
    bin_centers = (bins[:-1] + bins[1:])/2
    pdf_val = messy_instance.pdf_hist(0.5, hist, bins, bin_centers)
    assert isinstance(pdf_val, float)

def test_mgf_statio_basic(messy_instance):
    X_ = np.array([1.,2.,3.])
    ZZ = np.array([1.,2.,3.])
    H_lambdify = [lambda z: z, lambda z: z**2]
    lam, lrs, conv, max_cond = messy_instance.mgf_statio(X_, ZZ, H_lambdify, dim=2, tol=1e-2)
    assert lam.shape == (2,)
    assert lrs.shape == ZZ.shape
    assert isinstance(conv, bool)
    assert isinstance(max_cond, float)

def test_check_exponent_overflow(messy_instance):
    expr = x**2
    samples = np.array([0.1, 0.5, 0.9])
    overflow = messy_instance.check_exponent_overflow(expr, samples)
    assert isinstance(overflow, bool)

def test_vectorized_piecewise_evaluator(messy_instance):
    expr = sp.Piecewise((x, x<0), (x**2, True))
    arr = np.array([-1,0,2])
    res = messy_instance.vectorized_piecewise_evaluator(expr, arr)
    np.testing.assert_allclose(res, np.array([-1, 0, 4]))

def test_weighted_sum_piecewise_evaluator(messy_instance):
    expr1 = sp.Piecewise((x, x<0), (0, True))
    expr2 = sp.Piecewise((0, x<0), (x, True))
    arr = np.array([-1,0,2])
    res = messy_instance.weighted_sum_piecewise_evaluator([2,3], [expr1, expr2], arr)
    np.testing.assert_allclose(res, np.array([-2,0,6]))

'''
