import numpy as np
import sys

# %%


def a2xy(angle, radius=1):
    return np.cos(angle)*radius, np.sin(angle)*radius


def ik_angle(origin, point):
    x, y = np.vstack([origin, point])
    angle = 0
    dx = (y[0] - x[0])
    dy = (y[1] - x[1])
    if np.abs(dx) > 1e-30:
        aa = dy/dx
        angle = np.arctan(aa)
    return angle


class Sim:

    def __init__(self, name, type, stime):

        self.name = name
        self.type = type
        self.stime = stime

        sidewidth = 0.8
        sideheight = 1.5
        center = [0, 1]
        still_box = np.array([
            (-sidewidth, -sideheight),
            (sidewidth, -sideheight),
            (sidewidth, sideheight),
            (-sidewidth, sideheight)]) + center

        sidewidth = 1.2
        sideheight = 1.5
        center = [0, 1]
        normal_box = np.array([
            (-sidewidth, -sideheight),
            (sidewidth, -sideheight),
            (sidewidth, sideheight),
            (-sidewidth, sideheight)]) + center

        sidewidth = 5
        sideheight = 1.5
        center = [0, 1]
        large_box = np.array([
            (-sidewidth, -sideheight),
            (sidewidth, -sideheight),
            (sidewidth, sideheight),
            (-sidewidth, sideheight)]) + center

        if type == "still":
            self.points = still_box
        elif type == "normal":
            self.points = normal_box
        elif type == "large":
            self.points = large_box

        self.head_points = np.array([(-1, -0.75), (0, 1.5), (1, -0.75)])

        self.whisker_base = np.array([-0.25, 0])
        self.whisker_angle_ampl_scale = 0.35
        self.whisker_base_angle = np.pi*0.05
        self.whisker_len = 1.3
        self.whisker_init_angle = np.pi/4
        self.set_whisker(self.whisker_init_angle)

        self.whisker_model_base = np.array([-0.25, 0])
        self.whisker_model_len = 1.3
        self.whisker_model_init_angle = np.pi/4
        self.set_whisker_model(self.whisker_model_init_angle)

        self.angle = None
        self.box_points_init = np.array(self.points)
        self.box_points = self.box_points_init.copy()
        self.box_pos = 3
        self.move_box(0)

    def move_box(self, t):

        t_ratio = t/self.stime
        # compute box position
        if self.type == "still":
            bottom = 1.2
            rng = 1.6
        elif self.type == "normal":
            bottom = 1.2
            rng = 1.6
        elif self.type == "large":
            bottom = 1.2
            rng = 1.6

        top = bottom + rng

        if t_ratio < 0.15:
            box_pos = np.array([0, top])
        elif 0.15 <= t_ratio < 0.25:
            e = 1 - (t_ratio - 0.15)/0.1
            box_pos = np.array([0, bottom + rng * e])
        elif 0.25 <= t_ratio < 0.5:
            box_pos = np.array([0, bottom])
        elif 0.5 <= t_ratio < 0.6:
            e = (t_ratio - 0.5)/0.1
            box_pos = np.array([0, bottom + rng * e])
        else:
            box_pos = np.array([0, top])

        self.box_points = self.box_points_init + box_pos
        collision, curr_angle_limit = self.detect_collision()
        return collision, curr_angle_limit

    def set_whisker(self, angle):
        self.angle = angle
        self.whisker_vertices = self.whisker_base + \
            np.vstack([(0, 0), a2xy(np.pi - self.angle, self.whisker_len)])

    def set_whisker_model(self, angle):
        self.whisker_model_vertices = self.whisker_model_base + \
            np.vstack([(0, 0), a2xy(np.pi - angle, self.whisker_model_len)])

    def update(self, angle, angle_model):
        angle = self.whisker_angle_ampl_scale*angle * np.pi + \
            self.whisker_base_angle
        angle_model = self.whisker_angle_ampl_scale * angle_model*np.pi + \
            self.whisker_base_angle
        self.set_whisker(angle)
        self.set_whisker_model(angle_model)

    def detect_collision(self):

        box_height = self.box_points[0][1]
        whisker_head = self.whisker_base + self.whisker_len

        in_collision_with_vertex = box_height < whisker_head[1]

        collision = False
        curr_angle_limit = np.pi
        if in_collision_with_vertex:

            if self.box_points[0][0] - 0.1 > self.whisker_vertices[1][0]:
                angle_to_box_vertex = ik_angle(self.whisker_base,
                                               self.box_points[0])
                angle_to_box_vertex = np.abs(angle_to_box_vertex +
                                             self.whisker_base_angle)
                curr_angle_limit = angle_to_box_vertex
            else:
                box_height_whisk_angle = -np.arcsin(
                    (box_height - 0.1 - self.whisker_base[1]) / self.whisker_len)
                box_height_whisk_angle = np.abs(
                    box_height_whisk_angle + self.whisker_base_angle)
                curr_angle_limit = box_height_whisk_angle

            collision = True

        return (collision, curr_angle_limit)

    def close(self):
        self.vm.mk_video()


if __name__ == "__main__":
    sim = Sim("demo", "still", 1000)
    sim.move_box(0)
    for a in np.linspace(0, 50*np.pi, 1000):
        sim.update(np.sin(a), np.sin(a))
