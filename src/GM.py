# Copyright (c) 2021 Federico Maggiore, Francesco Mannella
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


class GM:

    def __init__(self, dt, eta=0.001, eta_d=1., eta_a=0.06, eta_nu=0.01, nu=[1., 1.]):

        # Parameter that regulates whiskers amplitude oscillation
        self.nu = np.array(nu)
        # Vector \vec{\mu} initialized with the GP initial conditions
        self.mu = np.zeros(len(nu))
        # Vector \dot{\vec{\mu}}
        self.dmu = np.zeros(len(nu))
        # Variances (inverse of precisions) of sensory proprioceptive inputs
        self.Sigma_s_p = np.ones(len(nu))*0.01
        # Variances (inverse of precisions) of sensory touch inputs
        self.Sigma_s_t = np.ones(len(nu))*0.032  # np.array(Sigma_s_t)
        # Internal variables precisions
        self.Sigma_mu = np.ones(len(nu))*0.01  # np.array(Sigma_mu)

        # Action variable (in this case the action is intended as the increment of the variable that the agent is allowed to modified)
        self.da = 0.
        # Size of a simulation step
        self.dt = dt
        # Gradient descent weights
        self.eta = eta
        self.eta_d = eta_d
        self.eta_a = eta_a
        self.eta_nu = eta_nu

    # Touch function
    def g_touch(self, x, v, prec=50):
        return 1/np.cosh(prec*v)*(0.5*np.tanh(prec*x)+0.5)

    # Derivative of the touch function with respect to v
    def dg_dv(self, x, v, prec=50):
        return -prec*1/np.cosh(prec*v)*np.tanh(prec*v)*(0.5 * np.tanh(prec*x) + 0.5)

    # Derivative of the touch function with respect to x
    def dg_dx(self, x, v, prec=50):
        return 1/np.cosh(prec*v)*0.5*prec*(1/np.cosh(prec*x))**2

    # Function that implement the update of internal variables.

    def update(self, touch_sensory_states, proprioceptive_sensory_states, x):
        # touch_sensory_states  and proprioceptive_sensory_states arguments come from GP (both arrays have dimension equal to the number of whiskers)
        # Returns action increment

        self.s_p = proprioceptive_sensory_states
        self.s_t = touch_sensory_states
        self.touch_pred = self.g_touch(x=self.mu, v=self.dmu)

        self.PE_mu = self.dmu - (self.nu*x - self.mu)
        self.PE_s_p = self.s_p-self.dmu
        self.PE_s_t = self.s_t-self.touch_pred

        self.dF_dmu = self.PE_mu/self.Sigma_mu \
            - self.dg_dx(x=self.mu, v=self.dmu)*self.PE_s_t/self.Sigma_s_t

        self.dF_d_dmu = self.PE_mu/self.Sigma_mu \
            - self.PE_s_p/self.Sigma_s_p \
            - self.dg_dv(x=self.mu, v=self.dmu) * \
            self.PE_s_t/self.Sigma_s_t

        # Action update
        # case with dg/da = 1
        self.da = -self.dt*self.eta_a * \
            (x*self.PE_s_p/self.Sigma_s_p + self.PE_s_t/self.Sigma_s_t)

        # Learning internal parameter nu
        self.nu += -self.dt*self.eta_nu*(-x*self.PE_mu/self.Sigma_mu)

        self.mu += self.dt*(self.dmu - self.eta*self.dF_dmu)
        self.dmu += -self.dt*self.eta_d*self.dF_d_dmu

        return self.da
