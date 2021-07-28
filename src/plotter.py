from mkvideo import vidManager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import sys
Path = mpath.Path


class SimPlotter:

    def __init__(self, sim):
        self.sim = sim
        self.fig = plt.figure(figsize=(5, 5))
        self.vm = vidManager(self.fig, name=sim.name,
                             dirname=sim.name, duration=0.1)
        self.ax = self.fig.add_subplot(111, aspect="equal")

        head_shape = mpatches.PathPatch(
            Path(np.vstack([sim.head_points, (0, 0)]),
                 [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
            fc="none", transform=self.ax.transData)
        self.head = self.ax.add_patch(head_shape)
        self.head.set_facecolor([0.8, 0.8, 0.8])

        self.whisker, = self.ax.plot(0, 0, c="k", lw=5)
        self.set_whisker()
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-0.8, 2.5])

        self.whisker_model, = self.ax.plot(0, 0, c="g", lw=5, alpha=0.5)
        self.set_whisker_model()
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-0.8, 2.5])

        self.box = None
        self.set_box()

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

    def set_box(self):
        if self.box is not None:
            self.box.remove()
        box_shape = mpatches.PathPatch(
            Path(np.vstack([self.sim.box_points, (0, 0)]),
                 [Path.MOVETO, Path.LINETO,
                  Path.LINETO, Path.LINETO,
                  Path.CLOSEPOLY]),
            fc="none", transform=self.ax.transData)
        self.box = self.ax.add_patch(box_shape)
        self.box.set_facecolor([0.9, 0.9, 0.9])

    def set_whisker(self):
        self.whisker.set_data(*self.sim.whisker_vertices.T)

    def set_whisker_model(self):
        self.whisker_model.set_data(*self.sim.whisker_model_vertices.T)

    def update(self):
        self.set_box()
        self.set_whisker()
        self.set_whisker_model()
        self.fig.canvas.draw()
        self.vm.save_frame()

    def close(self):
        self.vm.mk_video()


class SeriesPlotter:

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
        self.wall_array.append(w)
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
        self.wall.set_data(self.T, self.wall_array)
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
        self.ax.set_yticks([-0.1, 0, 0.5])
        self.ax.set_yticklabels(["-.1", "0", ".5"])
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


class Plotter:

    def __init__(self, sim, stime, type):
        self.stime = stime
        self.type = type
        self.prederr = PredErrPlotter("prederr", self.type, self.stime)
        self.genProcPlot = SeriesPlotter("gen_proc_" + self.type, type="process",
                                         wallcolor=[0.2, 0.2, 0, 0.2],
                                         labels={"x": "proprioception",
                                                 "nu": "action (oscil. ampl.)"},
                                         color=[.5, .2, 0], stime=self.stime)

        self.genModPlot = SeriesPlotter("gen_mod_"+self.type, type="model",
                                        wallcolor=[0, 0, 0, 0],
                                        labels={"x": "proprioception prediction",
                                                "nu": "internal cause (repr. oscill. ampl.)"},
                                        color=[.2, .5, 0], stime=self.stime)

        self.simPlot = SimPlotter(sim)
        self.sens = np.zeros(stime)
        self.sens_model = np.zeros(stime)
        self.ampl = np.zeros(stime)
        self.ampl_model = np.zeros(stime)
        self.touch = np.zeros(stime)
        self.current_touch = 0
        self.limit = 1000

    def update(self, t, gm, gp, limit, collision):

        # get state
        self.t = t
        self.sens[t] = gp.x[0]
        self.sens_model[t] = gm.mu[0]
        self.ampl[t] = gp.a[0]
        self.ampl_model[t] = gm.nu[0]
        self.current_touch = gp.s_t[0]
        self.touch[t] = gm.touch_pred[0]
        self.imit = limit
        self.collision = collision

    def draw(self):
        t = self.t
        self.prederr.update([self.sens[t], self.sens_model[t]], t)
        self.genProcPlot.update([self.sens[t], self.ampl[t],
                                 self.limit if self.collision
                                 is True else None, 0], t)
        self.genModPlot.update([self.sens_model[t], self.ampl_model[t],
                                self.limit if self.collision
                                is True else None, 0], t)
        self.simPlot.update()

    def close(self):
        self.simPlot.close()
        np.savetxt(self.type+"_touch", self.touch)
