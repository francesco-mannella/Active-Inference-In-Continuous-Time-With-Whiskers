import numpy as np
import matplotlib.pyplot as plt
from mpmath import sech, tanh
rng = np.random.RandomState()


def f(x, a, h):
    return a - h*x + np.pi/2


def p2std(p):
    return 10000*np.exp(-p)

# %%


class GP:
    """ Generative process.

    Implementation of the generative process :

    Attributes:
        pi_s: (float) Precision of sensory probabilities.
        pi_x: (float) Precision of hidden states probabilities.
        h: (float) Integration step of hidden states dynamics.
        gamma: (float) Attenuation factor of sensory prediction error.
        mu_s: (float)  sensory channel (central value).
        mu_x: (float) hidden state (central value).
        dmu_x: (float) Change of  hidden state (central value).
        da: (float) Increment of action
        dt: (float) Integration step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        a: (float) action
    """

    def __init__(self, dt=0.0005, freq=0.01, amp=0.1):

        self.pi_s = 9
        self.mu_x = np.array([1.,0.,amp*1])
        self.mu_s = 1
        self.omega_s = p2std(self.pi_s)
        self.dt = dt
        self.freq = freq
        self.a = amp
        self.t = 0

    def update(self, action):
        """ Update dynamics of the process.

        Args:
            action: (float) moves the current inner state.

        """

        self.a += self.dt*action
        self.mu_x[0] += self.dt*(self.freq*self.mu_x[1])
        self.mu_x[1] += self.dt*(-self.mu_x[0])
        self.mu_x[2] += self.dt*(self.a*self.mu_x[0] - self.mu_x[2])
        self.s = self.mu_x[2] + self.omega_s*rng.randn()

        return self.s


class GM:
    """ Generative Model.

    Attributes:
        pi_s: (float) Precision of sensory probabilities.
        pi_x: (float) Precision of hidden states probabilities.
        pi_nu: (float) Precision of hidden causes probabilities.
        h: (float) Integration step of hidden states dynamics.
        gamma: (float) Attenuation factor of sensory prediction error.
        mu_s: (float)  sensory channel (central value).
        mu_x: (float) hidden state (central value).
        dmu_x: (float) Change of  hidden state (central value).
        mu_nu: (float) Internal cause (central value).
        da: (float) Increment of action
        dt: (float) Integration step
        eta: (float) Free energy gradient step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        omega_nu : (float) Standard deviation of inner causes

    """

    def __init__(self, dt=0.0005, eta=0.0005,
                 freq=0.001, amp=np.pi/2):

        self.pi_s = np.array([9,9])
        self.pi_x = np.array([9,9,9])
        self.omega_s = p2std(self.pi_s)
        self.omega_x = p2std(self.pi_x)

        self.mu_x = np.array([1.,0.,amp*1])
        self.dmu_x = np.array([0.,-1/freq,0.])
        self.nu = amp

        self.da = 1
        self.dt = dt
        self.eta = eta
        self.freq = freq

    def f_touch(self, x, v):
        return sech(10*v)*(1/2 * tanh(10*x-2) + 1/2)

    def d_f_touch_dmu0(self, x, v):
        return -10*sech(10*x)*tanh(10*x)*(1/2 * tanh(10*x-2) + 1/2)

    def d_f_touch_dmu1(self, x, v):
        return sech(10*v)*5*(sech(10*x-2))**2

    def update(self, sensory_states):
        """ Update dynamics and give action

            Args:
                sensory_states: float current real proprioceptive and
                    somstosensory perception

            Returns:
                (float) current action increment
         """

        # update sensory states and dynamic precision
        self.s = sensory_states
        self.da = self.mu_x[0]

        s = self.s
        oms, omx = (self.omega_s, self.omega_x)
        mx = self.mu_x
        dmx = self.dmu_x
        n = self.nu
        da, fr = self.da, self.freq

        # TODO: gradient descent optimizations
        self.gd_mu_x = np.array([
            -(1/omx[2])*n*(n*mx[0]-mx[2]-dmx[2]) - (1/omx[1])*(mx[0]+dmx[1]) + (1/oms[1])*(s[1]-self.f_touch(mx[0],mx[1]))*self.d_f_touch_dmu0(mx[0],mx[1]) ,
            -(1/omx[0])*fr*(mx[1]*fr-dmx[0]) + (1/oms[1])*(s[1]-self.f_touch(mx[0],mx[1]))*self.d_f_touch_dmu1(mx[0],mx[1]),
            (1/oms[0])*(s[0]-mx[2]) - (1/omx[2])*(dmx[2]-(n*mx[0]-mx[2]))
            ])

        self.gd_dmu_x = np.array([
            -(1/omx[0])*(dmx[0] - fr*mx[1]),
            -(1/omx[1])*(mx[0] + dmx[1]),
            -(1/omx[2])*(dmx[2] - (n*mx[0] - mx[2]))])

        self.gd_nu = -(1/omx[2])*mx[0]*(n*mx[0] - mx[2] - dmx[2])
        self.gd_a = (1/oms[0])*da*(s[0]-mx[2]) - (1/oms[1])*(s[1]-self.f_touch(mx[0],mx[1]))*da

        # classic Active inference internal variables dynamics
        eta_mu = self.eta
        eta_dmu = 10000*self.eta
        d_dmu_x = self.dt*( eta_dmu*self.gd_dmu_x )
        self.mu_x = self.mu_x + self.dt*( self.dmu_x + eta_mu*self.gd_mu_x)
        self.dmu_x = self.dmu_x + d_dmu_x


        self.nu += self.dt*self.gd_a
        return self.gd_a


if __name__ == "__main__":

    gp = GP(dt=0.0005, freq=0.5, amp=1)
    gm = GM(dt=0.0005, eta=0.1, freq=0.5, amp=1)

    # %%
    data = []
    a = 0.0
    stime = 200000
    for t in range(stime):
        touch = 0.
        if t > 30000:
            gp.mu_x[2] = np.minimum(0.5, gp.mu_x[2])
            touch = 1.
        gp.update(a)
        s, gpm, gmm, gpa, gmn = gp.s, gp.mu_x[2], gm.mu_x[2], gp.a, gm.nu
        a = gm.update( [s,touch] )
        data.append([s, gpm, gmm, gpa, gmn])
    data = np.vstack(data)

    # %%

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(data[:, 1], c="red", lw=1, ls="dashed")
    plt.plot(data[:, 3], c="#aa6666", lw=3)
    plt.subplot(212)
    plt.plot(data[:, 2], c="green", lw=1, ls="dashed")
    plt.plot(data[:, 4], c="#66aa66", lw=3)
    plt.show()
