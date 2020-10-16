import matplotlib
from mkvideo import vidManager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
Path = mpath.Path
# matplotlib.use("Agg")


class Plotter:

    def __init__(self, name, labels, color, stime):
        self.fig = plt.figure(figsize=(3, 2))
        self.vm = vidManager(self.fig, name=name, dirname=name, duration=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Generative process")
        self.x, = self.ax.plot(0, 0, c=color, lw=1.5, ls="dashed")
        self.nu, = self.ax.plot(0, 0, c=color, lw=3)
        self.x_head = self.ax.scatter(0, 0, s=100, c=[color])
        self.nu_head = self.ax.scatter(0, 0, s=60, c=[color])
        self.ax.set_xlim([-0.1*stime, stime*1.1])
        self.ax.set_ylim([-1.8, 3.2])
        self.ax.set_yticks([-1, 0, 1])
        self.ax.set_xticks([])
        plt.legend([self.x, self.nu], [labels["x"], labels["nu"]])
        self.x_array = []
        self.nu_array = []
        self.T = []
        self.fig.tight_layout()

    def update(self, vals, t):
        s, a = vals
        self.x_array.append(s)
        self.nu_array.append(a)
        self.T.append(t)
        self.x.set_data(self.T, self.x_array)
        self.x_head.set_offsets([[t, self.x_array[-1]]])
        self.nu.set_data(self.T, self.nu_array)
        self.nu_head.set_offsets([[t, self.nu_array[-1]]])
        self.fig.canvas.draw()
        self.vm.save_frame()

    def close(self):
        self.vm.mk_video()
