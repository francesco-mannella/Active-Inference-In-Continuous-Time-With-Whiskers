import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState()


class Harmonic:

    def __init__(self, h=0.01, hy=0.25):
        self.h = h
        self.x = np.ones(2)
        self.dx = np.ones(2)
        self.y = 0

    def update(self, ampl, freq, decay):

        decay += np.maximum(0, self.x[0] - ampl)

        self.dx = np.dot([
        [0, freq],
        [-1, -decay]], self.x)
        self.x += self.h * self.dx

        return self.x

if __name__ == "__main__":
    stime = 20000
    h = Harmonic()
    x = []
    T = np.arange(stime)

    for t in T:

        tt = np.exp(-0.5*(2000**-2)*(t-stime/2)**2)


        h.update(ampl=10-9*tt, freq=30-25*tt, decay=0.0)
        x.append(h.x[0])
    x = np.array(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, lw=2, c="k")

    plt.show()
