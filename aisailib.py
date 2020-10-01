import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
rng = np.random.RandomState()
# np.seterr(all='raise')


def f(x, a, h):
    return a - h*x


def skewgauss(n, relative_location=0, alpha=0):
    ''' Generate a squew gaussian.

        Args:
            n: (int) timesteps
            relative_location: (float) relative shifting of the
                distribution [0, 1] (0.1 indicates a shifting toward
                the beginning of the interval,0.9 indicates a shifting
                toward its end)
            alpha: (float) skewness

        Example:
            stime = 1500
            plt.plot(stime, skewgauss(stime, location=0.6, alpha=4))
    '''
    location = 10*relative_location - 5
    rng = np.array([-2, 2]) - location
    x = np.linspace(rng[0], rng[1], n)
    return skewnorm.pdf(x, alpha)


# %%
class GP:
    """ Generative process.

    Implementation of the generative process from the paper:

    Brown, H., Adams, R. A., Parees, I., Edwards, M., & Friston, K. (2013).
    Active inference, sensory attenuation and illusions. Cognitive Processing,
    14(4), 411–427. https://doi.org/10.1007/s10339-013-0571-3

    Attributes:
        pi_s: (float) Precision of sensory states probabilities.
        pi_x: (float) Precision of hidden state probabilities.
        h: (float) Integration step of hidden state dynamics.
        mu_x: (float) Hidden state.
        dmu_x: (float) Change of hidden state.
        mu_s: (float) Proprioceptive sensory channel (central value).
        eta: (float) integration step
        omega_s: (float) standard deviation of sensory states
        omega_x: (float)  standard deviation of inner state increments
        a: (float) action
    """

    def __init__(self):

        self.pi_s = 8
        self.pi_x = 8

        self.h = 1.0/4.0

        self.mu_x = 0
        self.dmu_x = 0
        self.mu_s = 0

        self.eta = 0.00025

        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)

        self.a = 0

    def update(self, action_increment):
        """ Update dynamics of the process.

        Args:
            action_increment: (float) moves the current inner state.

        """

        da = action_increment
        self.a += self.eta*da

        self.mu_s = self.mu_x
        self.dmu_x = f(self.x, self.a, self.h)

    def generate(self):
        """ Generate sensory states """

        self.s = self.mu_s
        self.dx = self.dmu_x
        self.mu_x += self.eta*self.dx


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
        eta: (float) Integration step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        omega_nu : (float) Standard deviation of inner causes

    """

    def __init__(self):

        self.pi_s = 4
        self.pi_x = 4
        self.pi_nu = 6

        self.h = 1.0/4.0
        self.gamma = 6

        self.mu_x = 0
        self.dmu_x = 0
        self.mu_nu = 0

        self.da = 1/self.h
        self.eta = 0.00025

        self.omega_s = np.exp(-self.pi_s_int)
        self.omega_x = np.exp(-self.pi_x)
        self.omega_nu = np.exp(-self.pi_nu)

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
        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)
        self.omega_nu = np.exp(-self.pi_nu)

        # dynamics of internal variables
        self.dmu_x = f(self.mu_x, self.mu_nu, self.h)

        s = self.s
        oms, omx, omn = (self.omega_s, self.omega_x, self.omega_nu)
        mx = self.mu_x
        dmx = self.dmu_x
        n = self.mu_nu
        h, da = self.h, self.da

        # TODO: gradient descent optimizations
        self.gd_dmu_x = -omx*(dmx - f(mx, n, h))
        self.gd_mu_x = oms*(s - mx)
        self.gd_mu_nu = omx*(dmx - f(mx, n, h))
        self.gd_a = 0

        # update with gradients
        self.dmu_x += self.eta*self.gd_dmu_x
        self.mu_x += self.eta*self.gd_mu_x

        self.sg = self.mu_x

        return self.gd_a


if __name__ == "__main__":

    gp = GP()
    gm = GM()

    # %%
    data = []

    stime = 100000
    t = np.arange(stime)
    ta = skewgauss(n=stime, relative_location=0.5, alpha=4)
    da = 0

    plt.plot(ta)
    # %%
    for t in range(stime):
        gm.mu_nui = ta[t]
        gp.update(da)
        gp.generate()
        sp, ss = gp.sp, gp.ss
        da = gm.update((sp, ss))
        spg, ssg = gm.spg, gm.ssg
        os = gm.omega_s
        data.append((sp, ss, spg, ssg, os))

    data = np.vstack(data)

    # %%
    sp, ss, spg, ssg, os = data.T

    t = np.arange(len(ss))
    plt.fill_between(t, ss - os, ss + os, color=[0.8, 0.8, 0.8])
    p1, = plt.plot(t, sp, c='black', lw=1)
    p2, = plt.plot(t, ss, c='black', lw=2)
    p3, = plt.plot(t, spg, c='blue', lw=1)
    p4, = plt.plot(t, ssg, c='blue', lw=2)
    plt.legend([p1, p2, p3, p4], ['sp', 'ss', 'spg', 'ssg'])
