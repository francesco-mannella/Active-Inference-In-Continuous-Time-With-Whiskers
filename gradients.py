
# %% cell

from IPython.display import display
import numpy as np
from matplotlib import pyplot as plt
import sympy as syp
from sympy import init_printing, symbols, sqrt, \
    exp, Inverse as inv, pi, log, re, Eq, diff, function
from sympy.matrices import Matrix, ones, zeros, eye
init_printing(use_unicode=True)


# %%

C = symbols("C", real="True")

sigma_s, sigma_x, sigma_nu = \
    symbols(r"\sigma_s \sigma_x \sigma_{\nu}", real=True)

s = Matrix(2, 1, symbols('s_p s_s'), real=True)
mux = Matrix(2, 1, symbols(r'\mu_{x_i} \mu_{x_e}'), real=True)
dmux = Matrix(2, 1, symbols(r'd\mu_{x_i} d\mu_{x_e}'), real=True)
munu = Matrix(2, 1, symbols(r'\mu_{\nu_i} \mu_{\nu_e}'), real=True)
dmunu = Matrix(2, 1, symbols(r'd\mu_{\nu_i} d\mu_{\nu_e}'), real=True)

Sigma_s = eye(2, real=True)*sigma_s
Sigma_x = eye(2, real=True)*sigma_x
Sigma_nu = eye(2, real=True)*sigma_nu


def g(x):
    W = eye(2, real=True)
    W[1, 0] = 1
    return W*x


def f(x, n):
    h = symbols("h", real=True, positive=True)
    return n - (eye(2, real=True)*h)*x


def normal(x, m, S):
    n = exp(-0.5*(x - m).T * inv(S) * (x - m)) \
        / sqrt(S.norm()*((2*pi)**2))
    return n


pF = normal(s, g(mux), Sigma_s) \
    * normal(dmux, f(mux, munu), Sigma_x)\
    * normal(dmunu, munu, Sigma_nu)

# %%
F = -log(pF[0]) - C
F = F.expand(force=True)
F = F.collect(Sigma_s)
F = F.collect(Sigma_x)
F = F.collect(Sigma_nu)
display(F)


# %%
d_mux = Eq(-diff("F", mux, evaluate=False),
           -syp.separatevars(diff(F, mux), force=True),
           evaluate=False)
display(d_mux)
print(syp.latex(d_mux))

# %%
d_dmux = Eq(-diff("F", dmux, evaluate=False),
            -diff(F, dmux).simplify(), evaluate=False)
display(d_dmux)
print(syp.latex(d_dmux))

# %%
a = symbols("a", real=True)
spa = syp.Function("s_p")(a)
ssa = syp.Function("s_s")(a)
F = F.subs(s[0], spa)
F = F.subs(s[1], ssa)

da = Eq(-diff("F", "a", evaluate=False),
        -diff(F, a).simplify(), evaluate=False)
display(da)
print(syp.latex(da))
