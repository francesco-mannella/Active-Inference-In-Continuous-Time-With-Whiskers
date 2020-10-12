from IPython.display import display
import sympy as syp
from sympy import init_printing, symbols, sqrt, \
    exp, Inverse as inv, pi, log, Eq, diff
from sympy.matrices import Matrix, eye, zeros
init_printing(use_unicode=True)
# %%

C = symbols("C", real="True")

sigma_s, sigma_x, sigma_nu = \
    symbols(r"\sigma_s \sigma_x \sigma_{\nu}", real=True)

s,h = symbols(r's,h', real=True)
fr = symbols(r'\phi', real=True)
mux = Matrix(3, 1, symbols(r'\mu_{x_1} \mu_{x_2} \mu_{x_3}'), real=True)
dmux = Matrix(3, 1, symbols(r'd\mu_{x_1} d\mu_{x_2} d\mu_{x_3}'), real=True)
munu = symbols(r'\mu_{\nu}', real=True)

Sigma_x = eye(3, real=True)*sigma_x


def g(x):
    return x[2]


def f(x, a, freq=fr):
    W = Matrix(3, 3,
               [0,           1,  0,
                -0.25*freq, 0,     0,
                a,          0,    -h])
    return W*x


def normal1d(x, m, S):
    n = exp(-0.5*(S**-2)*(x - m)**2) \
        / (S*sqrt(2*pi))
    return n


def normal(x, m, S):
    n = exp(-0.5*(x - m).T * inv(S) * (x - m)) \
        / sqrt(S.norm()*((2*pi)**2))
    return n


p_s_mu = normal1d(s, g(mux), sigma_s)
p_dmu_mu = normal(dmux, f(mux, munu), Sigma_x)
pF = p_s_mu*p_dmu_mu
pF = syp.nsimplify(pF)
display(p_s_mu)
display(syp.nsimplify(p_dmu_mu))

display(pF)

# %%
F = -log(pF[0]) - C
F = F.expand(force=True)
F = F.collect(sigma_s)
F = F.collect(Sigma_x)
F = syp.nsimplify(F)
display(F)


# %%
gd_mux = Eq(-diff("F", mux, evaluate=False),
            -syp.separatevars(diff(F, mux), force=True),
            evaluate=False)
display(gd_mux)
print(syp.latex(gd_mux))

# %%
d_dmux = Eq(-diff("F", dmux, evaluate=False),
            -diff(F, dmux).simplify(), evaluate=False)
display(d_dmux)
print(syp.latex(d_dmux))


# %%
dg_nu = Eq(-diff("F", munu, evaluate=False),
           -diff(F, munu).simplify(), evaluate=False)
display(dg_nu)
print(syp.latex(dg_nu))
# %%
a = symbols("a", real=True)
sa = syp.Function("s")(a)
F = F.subs(s, sa)


da = Eq(-diff("F", "a", evaluate=False),
        -diff(F, a).simplify(), evaluate=False)
display(da)
print(syp.latex(da))
