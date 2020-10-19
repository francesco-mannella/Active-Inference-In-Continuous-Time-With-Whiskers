from mkvideo import vidManager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
Path = mpath.Path

# %%


def a2xy(angle, radius=1):
    return np.cos(angle)*radius, np.sin(angle)*radius


class Sim:

    def __init__(self, name, points):
        self.fig = plt.figure(figsize=(5, 5))
        self.vm = vidManager(self.fig, name=name, dirname=name, duration=0.1)
        self.ax = self.fig.add_subplot(111, aspect="equal")

        head_points = np.array([(-1, -0.75), (0, 1.5), (1, -0.75)])
        head_shape = mpatches.PathPatch(
            Path(np.vstack([head_points, (0, 0)]),
                 [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
            fc="none", transform=self.ax.transData)
        self.head = self.ax.add_patch(head_shape)
        self.head.set_facecolor([0.8, 0.8, 0.8])

        self.whisker, = self.ax.plot(0, 0, c="k", lw=5)
        self.whisker_base = np.array([-0.25, 0])
        self.whisker_len = 2
        self.whisker_init_angle = np.pi/4
        self.set_whisker(self.whisker_init_angle)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-0.8, 2.5])

        self.whisker_model, = self.ax.plot(0, 0, c="g", lw=5, alpha=0.5)
        self.whisker_model_base = np.array([-0.25, 0])
        self.whisker_model_len = 2
        self.whisker_model_init_angle = np.pi/4
        self.set_whisker_model(self.whisker_model_init_angle)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-0.8, 2.5])

        self.angle = None
        self.box = None
        self.box_points_init = np.array(points)
        self.box_pos = 3
        self.move_box([0, self.box_pos])
        self.set_box()

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

    def move_box(self, pos):
        self.box_points = self.box_points_init + pos

    def set_box(self):
        if self.box is not None:
            self.box.remove()
        box_shape = mpatches.PathPatch(
            Path(np.vstack([self.box_points, (0, 0)]),
                 [Path.MOVETO, Path.LINETO,
                  Path.LINETO, Path.LINETO,
                  Path.CLOSEPOLY]),
            fc="none", transform=self.ax.transData)
        self.box = self.ax.add_patch(box_shape)
        self.box.set_facecolor([0.9, 0.9, 0.9])

    def set_whisker(self, angle):
        vertices = self.whisker_base + \
            np.vstack([(0, 0), a2xy(np.pi - angle, self.whisker_len)])
        self.whisker.set_data(*vertices.T)

    def set_whisker_model(self, angle):
        vertices = self.whisker_model_base + \
            np.vstack([(0, 0), a2xy(np.pi - angle, self.whisker_model_len)])
        self.whisker_model.set_data(*vertices.T)

    def update(self, angle, angle_model):
        angle = (0.3*angle + 0.2)*np.pi
        angle_model = (0.3*angle_model+0.2)*np.pi
        self.set_whisker(angle)
        self.set_whisker_model(angle_model)
        self.fig.canvas.draw()
        self.vm.save_frame()

    def close(self):
        self.vm.mk_video()


if __name__ == "__main__":
    plt.ion()
    sim = Sim("demo")
    sim.set_box([0, 1.3])
    for a in np.linspace(0, 50*np.pi, 1000):
        sim.update(np.sin(a), np.sin(a))
    sim.close()
