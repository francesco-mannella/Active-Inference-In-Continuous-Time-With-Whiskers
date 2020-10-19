from IPython.display import display
import sympy as syp
from sympy import init_printing
from sympy.physics.mechanics import dynamicsymbols
init_printing(use_unicode=True)
# %%

mux3 = dynamicsymbols('\mu_{x_3}')
t, mux1, h, munu = syp.symbols(r"t,\mu_{x_1} h a")
dynamics = munu*mux1 - h*mux3 - syp.diff(mux3, t)
display(dynamics)

# %%
mux3_dyn = syp.dsolve(dynamics, mux3)
display(mux3_dyn)

# %%
nu_diff = syp.diff(mux3_dyn.rhs, munu)
display(nu_diff)
