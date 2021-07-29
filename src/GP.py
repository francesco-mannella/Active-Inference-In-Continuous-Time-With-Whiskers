# Copyright (c) 2021  Federico Maggiore, Francesco Mannella
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


class GP:

    def __init__(self, dt, omega2_GP=0.5, alpha=[1., 1.], rng=None):

        if rng is None:
            self.rng = np.random.RandomState()

        # Harmonic oscillator angular frequency (both x_0 and x_2)
        self.omega2 = omega2_GP
        # Variable representing the central pattern generator
        self.cpg = np.array([0., 0.5])
        # Parameter that regulates whiskers amplitude oscillation
        self.a = np.array(alpha)
        # Whiskers base angles
        self.x = np.zeros(len(self.a))
        # Array storing proprioceptive sensory inputs (whiskers angular velocity)
        self.s_p = np.ones(len(self.a))*(self.a*self.cpg[0]-self.x)
        # Array storing touch sensory inputs
        self.s_t = np.zeros(len(self.a))
        # Variance of the Gaussian noise that gives proprioceptive sensory inputs
        self.Sigma_s_p = np.ones(len(self.a))*0.05
        # Variance of the Gaussian noise that gives touch sensort inputs
        self.Sigma_s_t = np.ones(len(self.a))*0.
        # Size of a simulation step
        self.dt = dt
        # Time variable
        self.t = 0.
        self.effective_object_position = 1.0e10*np.ones(len(self.a))

    # Function that regulates object position
    def obj_pos(self, t, obj_interval):
        if t > obj_interval[0] and t < obj_interval[1]:
            return self.object_position
        else:
            return np.array([10., 10.])

    # Discrete function that return if a whisker has touched

    def touch(self, x, object):
        if x >= object:
            return 1.
        else:
            return 0.

    # Continuous function that return if a whisker has touched
    def touch_cont(self, x, platform_position, prec=100):
        return 0.5 * (np.tanh(prec*(x-platform_position)) + 1)

    # Function that implement dynamics of the process.
    def update(self, action):
        # Action argument (double) is the variable that comes from the GM that modifies alpha
        # variable affecting the amplitude of the oscillation.

        # Increment of time variable
        self.t += self.dt
        # Increment of alpha variable (that changes the amplitude) given by agent's action
        self.a += action
        # GP dynamics implementation
        self.cpg[0] += self.dt*(self.cpg[1])
        self.cpg[1] += -self.dt*(self.omega2*self.cpg[0])
        self.x += self.dt*(self.a*self.cpg[0] - self.x)

        # object Action on touch sensory inputs
        for i in range(len(self.x)):
            self.s_t[i] = self.touch_cont(
                self.x[i], self.effective_object_position[i]) \
                + self.Sigma_s_t[i]*self.rng.randn()
            if self.x[i] > self.effective_object_position[i]:
                self.x[i] = min(self.x[i], self.effective_object_position[i])
                self.s_p[i] = 0  # self.a[i]*self.cpg[0] - self.x[i]
            else:
                self.s_p[i] = self.a[i]*self.cpg[0] - self.x[i]
            self.s_p[i] += self.Sigma_s_p[i]*self.rng.randn()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    gp = GP(dt=0.005, omega2_GP=0.5, alpha=[1., 1.])

    stime = 15000
    data = np.zeros(stime)
    for t in range(stime):
        gp.update([0, 0])
        data[t] = gp.x[0]

    plt.plot(data)
    plt.show()
