import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState()


class Harmonic:

    def __init__(self, h=0.01, hy=0.25):
        self.h = h
        self.hy = hy
        self.x = 1
        self.dx = 1
        self.ampl = 0
        self.freq = 0.1
        self.peak = 0

        self.y = 0
        self.yw = np.zeros(3)

    def update(self, ampl=np.pi*0.5, freq=0.1):

        self.ampl += self.h*(-self.ampl + ampl)
        self.freq += self.h*(-self.freq + 2*np.pi*freq)

        self.ddx = -self.freq * self.x
        self.dx += self.h * self.ddx
        self.x += self.h * self.dx

        self.y += self.h*(self.ampl*self.x - self.hy*self.y)
        return self.dx, self.x, self.y

if __name__ == "__main__":
    stime = 5000
    h = Harmonic()
    x = []
    p = []
    T = np.arange(stime)
    tt = np.exp(-0.5*((0.08*stime)**-2)*(T - stime*0.6)**2)

    for t in T:
        a = 0.5*np.pi*(1 - tt[t])
        f = 1 + 60*tt[t]
        h.update(ampl=a, freq=f)
        x.append(h.y)
    x = np.array(x)
    p = np.array(p)

    plt.figure(figsize=(10, 5))
    plt.plot(tt, lw=2, c="k")
    plt.figure(figsize=(10, 5))
    plt.plot(-100 + 100*p, c="y", lw=0.5)
    plt.plot(x, lw=2, c="k")
    plt.ylim([-np.max(x)*1, np.max(x)*1])

    plt.show()
