import numpy as np
import matplotlib.pyplot as plt
from aisailib import GP, GM, skewgauss

# %% markdown

### initialize process and model

# %%

gp = GP()
gm = GM()

# %% markdown

### Define time and inner cause curve

#%%

stime = 100000
t = np.arange(stime)
inner_cause = skewgauss(n=stime, relative_location=0.5, alpha=4)

plt.plot(inner_cause)
# %% markdown

###   Loop through timesteps

# %%

da = 0
data = []
for t in range(stime):
    gm.mu_nui = inner_cause[t]
    gp.update(da)
    gp.generate()
    sp, ss = gp.s
    da = gm.update(s)

data = np.vstack(data)

# %% markdown

### Plot sensory anticipation vs sensory perceptions

# %%
