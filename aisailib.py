import numpy as np
from harmonic import Harmonic
import matplotlib.pyplot as plt
rng = np.random.RandomState()


def f(x, a, h):
    return a - h*x + np.pi/2


# %%
class GP:
    """ Generative process.

    Implementation of the generative process :

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

        self.mu_x = np.pi/2
        self.dmu_x = 0
        self.mu_s = np.pi/2

        self.eta = 0.00025

        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)

        self.oscil = Harmonic(amplitude=0.5*np.pi, freq=0.0005, h=0.001)

        self.a = 0

    def update(self, action_increment):
        """ Update dynamics of the process.

        Args:
            action_increment: (float) moves the current inner state.

        """

        da = action_increment
        self.a += self.eta*da

        self.dmu_x = f(self.mu_x, 0.5*np.pi*np.tanh(self.a), self.h)
        self.dx = self.dmu_x + self.omega_x*rng.randn()
        self.mu_x += self.eta*self.dx

        self.oscil.amplitude = self.mu_x

        self.oscil.update()
        self.s = self.oscil.peak


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

        self.mu_x = np.pi/2
        self.dmu_x = 0
        self.mu_nu = np.pi/2

        self.da = 1/self.h
        self.eta = 0.00025

        self.omega_s = np.exp(-self.pi_s)
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
        oms, omx = (self.omega_s, self.omega_x)
        mx = self.mu_x
        dmx = self.dmu_x
        n = self.mu_nu
        h, da = self.h, self.da

        # TODO: gradient descent optimizations
        self.gd_dmu_x = -(1/omx)*(dmx - f(mx, n, h))
        self.gd_mu_x = (1/oms)*(s - mx)
        self.gd_mu_nu = (1/omx)*(dmx - f(mx, n, h))
        self.gd_a = (1/oms)*da*(s - mx)

        # update with gradients
        self.dmu_x += self.eta*self.gd_dmu_x
        self.mu_x += self.eta*self.gd_mu_x
        self.mu_nu += self.eta*self.gd_mu_nu

        self.sg = self.mu_x

        return self.gd_a


if __name__ == "__main__":

    gp = GP()
    gm = GM()

    # %%
    data = []
    da = 0
    stime = 28000
    for t in range(stime):
        gp.update(da)
        s = gp.s
        da = gm.update(s)
        data.append(gp.oscil.y)


plt.plot(data)
