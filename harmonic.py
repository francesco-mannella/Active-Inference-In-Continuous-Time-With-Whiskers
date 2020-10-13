import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState()


class Harmonic:

    def __init__(self):
        self.h = 0.1
        self.x = 2*np.pi
        self.dx = 2*np.pi
        self.y = 2*np.pi

    def update(self, ampl, freq, decay):

        self.dx += -self.h*freq*self.x
        self.x +=  self.h*self.dx
        self.y += self.h*(ampl*self.x -decay*self.y)

        return self.x


if __name__ == "__main__":
    stime = 3000
    h = Harmonic()

    data = []
    for t in range(stime):
        if stime*0.5<t<stime*0.7:
            h.update(ampl=4, freq=2, decay=0.99)
        else:
            h.update(ampl=1, freq=7, decay=0.99)
        data.append(h.y)
    data = np.array(data)

    plt.figure(figsize=(10, 5))
    plt.plot(data, lw=2, c="k")

    plt.show()
