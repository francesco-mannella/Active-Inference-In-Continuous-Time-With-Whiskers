from mkvideo import vidManager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
Path = mpath.Path


class Plotter:

    def __init__(self, name, type, labels, wallcolor, color, stime):
        self.fig = plt.figure(figsize=(5, 2.5))
        self.vm = vidManager(self.fig, name=name, dirname=name, duration=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Generative "+type)
        self.wall, = self.ax.plot(-99, -99, c=wallcolor, lw=6)
        self.x, = self.ax.plot(0, 0, c=color, lw=1.5, ls="dashed")
        self.nu, = self.ax.plot(0, 0, c=color, lw=3)
        self.x_head = self.ax.scatter(0, 0, s=100, c=[color])
        self.nu_head = self.ax.scatter(0, 0, s=60, c=[color])
        self.sigma_fill = None
        self.ax.set_xlim([-0.1*stime, stime*1.1])
        self.ax.set_ylim([-1.8, 3.2])
        self.ax.set_yticks([-1, 0, 1])
        self.ax.set_xticks([])
        if np.sum(wallcolor) == 0:
            plt.legend([self.x, self.nu], [labels["x"], labels["nu"]])
        else:
            plt.legend([self.x, self.nu, self.wall],
                       [labels["x"], labels["nu"], "box"])
        self.x_array = []
        self.nu_array = []
        self.wall_array = []
        self.sigma_array = []
        self.T = []
        self.color = color
        self.fig.tight_layout()

    def update(self, vals, t):
        s, a, w, o = vals
        self.x_array.append(s)
        self.sigma_array.append(o)
        self.nu_array.append(a)
        if w is True:
            self.wall_array.append(t)
        self.T.append(t)
        if self.sigma_fill is not None:
            self.sigma_fill.remove()
        self.sigma_fill = self.ax.fill_between(
            self.T,
            np.array(self.x_array)-self.sigma_array,
            np.array(self.x_array)+self.sigma_array,
            facecolor=self.color,
            edgecolor=[0, 0, 0, 0],
            alpha=0.4,
            zorder=-10)

        self.x.set_data(self.T, self.x_array)
        self.x_head.set_offsets([[t, self.x_array[-1]]])
        self.nu.set_data(self.T, self.nu_array)
        self.nu_head.set_offsets([[t, self.nu_array[-1]]])
        self.wall.set_data(self.wall_array, 0.5*np.ones_like(self.wall_array))
        self.fig.canvas.draw()
        self.vm.save_frame()

    def plot_first(self, t):
        self.ax.scatter(t, 2, s=300, facecolor=[0, 0, 0, 0],
                        edgecolor=[0, 0, 0, 1])
        self.ax.text(t, 2, "1",
                     horizontalalignment="center",
                     verticalalignment="center")

    def plot_second(self, t):
        self.ax.scatter(t, 2, s=300, facecolor=[0, 0, 0, 0],
                        edgecolor=[0, 0, 0, 1])
        self.ax.text(t, 2, "2",
                     horizontalalignment="center",
                     verticalalignment="center")

    def close(self):
        self.vm.mk_video()


class PredErrPlotter:
    def __init__(self, name, type, stime):
        self.fig = plt.figure(figsize=(5, 1.5))
        self.vm = vidManager(self.fig, name=name+"_"+type,
                             dirname=name+"_"+type, duration=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Prediction error")
        self.pe, = self.ax.plot(0, 0, c="k", lw=1)
        self.pe_head = self.ax.scatter(0, 0, c="k", s=60)
        self.ax.set_xlim([-0.1*stime, stime*1.1])
        self.ax.set_ylim([-0.2, 0.5])
        self.ax.set_yticks([-1, 0, 1])
        self.ax.set_xticks([])
        self.pe_array = []
        self.T = []
        self.fig.tight_layout()

    def update(self, vals, t):
        gs, ms = vals
        self.pe_array.append(ms-gs)
        self.T.append(t)

        self.pe.set_data(self.T, self.pe_array)
        self.pe_head.set_offsets([[self.T[-1], self.pe_array[-1]]])
        self.fig.canvas.draw()
        self.vm.save_frame()
