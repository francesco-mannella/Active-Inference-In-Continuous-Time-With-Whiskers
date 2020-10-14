import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState()

def sigm(x):
    return 1/(1+np.exp(-x*30))

class Harmonic:

    def __init__(self):
        self.h = 0.1
        self.x1 = 1
        self.x2 =  1
        self.x3 =  1

    def update(self, ampl, freq):
        th = np.abs(self.x1) - ampl
        self.x1 += self.h*(freq*self.x2) 
        self.x2 += self.h*(-self.x1 - self.x3*self.x2)
        self.x3 += self.h*(th*sigm(th) - self.x3)

        return self.x1

if __name__ == "__main__":
    stime = 10000
    h = Harmonic()
    x = []
    T = np.arange(stime)

    for t in T:

        tt = np.exp(-0.5*(200**-2)*(t-stime/2)**2)

        ttt = stime*0.3 < t < stime*0.7
        h.update(ampl=1+0.4*ttt, freq=0.001 + 0.1*ttt)
        x.append(h.x1)
    x = np.array(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, lw=2, c="k")

    plt.show()
