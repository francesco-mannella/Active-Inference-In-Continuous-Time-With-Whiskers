import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState()


def f(x, a, h):
    return a - h*x + np.pi/2


def p2std(p):
    return 100*np.exp(-p)

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
        eta: (float) Integration step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        a: (float) action
    """

    def __init__(self, eta=0.0005, freq=0.01, amp=0.1):

        self.pi_s = 4.5
        self.pi_x = 4.5
        self.mu_x = np.ones(3)
        self.mu_s = 1
        self.omega_s = p2std(self.pi_s)
        self.omega_x = p2std(self.pi_x)
        self.eta = eta
        self.freq = freq
        self.a = amp
        self.t = 0

    def update(self, action):
        """ Update dynamics of the process.

        Args:
            action: (float) moves the current inner state.

        """

        self.a += self.eta*action
        self.mu_x[0] += self.eta*(self.freq*self.mu_x[1])
        self.mu_x[1] += self.eta*(-self.mu_x[0])
        self.mu_x[2] += self.eta*(self.a*self.mu_x[0] - self.mu_x[2])
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
        eta: (float) Integration step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        omega_nu : (float) Standard deviation of inner causes

    """

    def __init__(self, eta=0.0005, freq=0.001, amp=np.pi/2):

        self.pi_s = 4.5
        self.pi_x = 4.5

        self.mu_x = np.ones(3)
        self.dmu_x = np.ones(3)
        self.mu_nu = amp

        self.da = 1
        self.eta = eta
        self.h = eta
        self.freq = freq

        self.omega_s = p2std(self.pi_s)
        self.omega_x = p2std(self.pi_x)

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
        n = self.mu_nu
        da, fr = self.da, self.freq

        # TODO: gradient descent optimizations
        self.gd_mu_x = np.array([
            -(1/omx)*(n*(n*mx[0] - mx[2] - dmx[2]) + (mx[0] + dmx[1])),
            -(1/omx)*fr*(mx[1]*fr - dmx[0]),
            (1/oms**2)*(s - mx[2]) - (1/omx)*(dmx[2] - (n*mx[0] - mx[2]))])

        self.gd_dmu_x = np.array([
            -(1/omx)*(dmx[0] - fr*mx[1]),
            -(1/omx)*(mx[0] + dmx[1]),
            -(1/omx)*(dmx[2] - (n*mx[0] - mx[2]))])

        self.gd_mu_nu = -(1/omx)*mx[0]*(n*mx[0] - mx[2] - dmx[2])
        self.gd_a = (1/oms**2)*da*(s - mx[2])

        # dynamics of internal variables
        self.dmu_x[0] = self.freq*self.mu_x[1]
        self.dmu_x[1] = -self.mu_x[0]
        self.dmu_x[2] = self.mu_nu*self.mu_x[0] - self.mu_x[2]

        # update with gradients
        self.dmu_x += self.eta*self.gd_dmu_x
        self.mu_x += self.eta*self.dmu_x + self.h*self.gd_mu_x
        self.mu_nu += self.eta*self.gd_a
        # self.mu_nu += self.eta*self.gd_mu_nu

        return self.gd_a


if __name__ == "__main__":

    gp = GP(eta=0.002, freq=0.5, amp=1)
    gm = GM(eta=0.002, freq=0.5, amp=1)

    # %%
    data = []
    sens = []
    a = 0
    stime = 100000
    peaks = 0
    for t in range(stime):
        gp.a = np.minimum(1, gp.a)
        gp.update(a)
        s, yg, ym, aa, n = gp.s, gp.mu_x[2], gm.mu_x[2], gp.a, gm.mu_nu

        if len(sens) > 2:
            dd = np.diff(sens[-2:])[-1]
            if np.abs(dd) < 0.0008 and dd > 0:
                peaks += 1
                if peaks > 6:
                    gp.a = np.minimum(0.5, gp.a)
                    print(t)

        a = gm.update(s)

        data.append([s, yg, ym, aa, n])
        sens.append(s)
    data = np.vstack(data)

    # %%

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(data[:, 1], c="red", lw=1, ls="dashed")
    plt.plot(data[:, 3], c="#aa6666", lw=3)
    plt.plot([0, stime], [1.5, 1.5], c="red", lw=0.5)
    plt.plot([0, stime], [1, 1], c="red", lw=0.5)
    plt.plot([0, stime], [0.5, 0.5], c="red", lw=0.5)
    plt.subplot(212)
    plt.plot(data[:, 2], c="green", lw=1, ls="dashed")
    plt.plot(data[:, 4], c="#66aa66", lw=3)
    plt.plot([0, stime], [1.5, 1.5], c="green", lw=0.5)
    plt.plot([0, stime], [1, 1], c="green", lw=0.5)
    plt.plot([0, stime], [0.5, 0.5], c="green", lw=0.5)
    plt.show()
