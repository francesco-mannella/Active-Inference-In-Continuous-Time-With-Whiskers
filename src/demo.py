from plotter import Plotter, PredErrPlotter
from sim import Sim
from aisailib import GP, GM
import numpy as np


def ik_angle(origin, point):
    x, y = np.vstack([origin, point])
    angle = 0
    dx = (y[0] - x[0])
    dy = (y[1] - x[1])
    if np.abs(dx) > 1e-30:
        aa = dy/dx
        angle = np.arctan(aa)

        # if dx < 0 and dy > 0:
        #     angle = 0.5*np.pi - angle
        # if dx < 0 and dy < 0:
        #     angle = np.pi + angle
        # if dx > 0 and dy < 0:
        #     angle = 1.5*np.pi - angle
    return angle


normal_box = [
    (-0.8, -.5), (0.8, -.5),
    (0.8, 2.5), (-0.8, 2.5)]
large_box = [
    (-1.3, -.5), (1.3, -.5),
    (1.3, 2.5), (-1.3, 2.5)]
for type in ["normal", "large", "still"]:
    stime = 180000

    gp = GP(eta=0.0005, freq=0.5, amp=1.2)
    gm = GM(eta=0.0005, freq=0.5, amp=1.2)

    sim = Sim("demo_"+type, points=normal_box
              if type == "normal" or type == "still" else large_box)
    prederr = PredErrPlotter("prederr", type, stime)
    genProcPlot = Plotter("gen_proc_"+type, type="process",
                          wallcolor=[0.2, 0.2, 0, 0.2],
                          labels={"x": "proprioception",
                                  "nu": "action"},
                          color=[.5, .2, 0], stime=stime)

    genModPlot = Plotter("gen_mod_"+type, type="model",
                         wallcolor=[0, 0, 0, 0],
                         labels={"x": "proprioception",
                                 "nu": "internal cause"},
                         color=[.2, .5, 0], stime=stime)

    delta_action = 0

    sens = np.zeros(stime)
    ampl = np.zeros(stime)
    sens_model = np.zeros(stime)
    ampl_model = np.zeros(stime)

    for t in range(stime):

        if type == "normal" or type == "large":
            box_pos = np.array([0, np.maximum(1.3, 2.5*np.exp(-3*t/stime)+0.7)])
        else:
            box_pos = np.array([0, 5]) if t < stime/3 else np.array([0, 1.48])

        sim.move_box(box_pos)

        angle = ik_angle(sim.whisker_base, sim.box_points[0])
        angle = np.abs(angle+0.2*np.pi)
        min_pos = + sim.whisker_base[1] + sim.whisker_len - sim.box_points_init[0][1]
        angle_pos = angle if box_pos[1] < min_pos else 10

        gp.update(delta_action)

        gp.mu_x[2] = np.minimum(angle_pos, gp.mu_x[2])

        sens[t] = gp.mu_x[2]
        sens_model[t] = gm.mu_x[2]
        ampl[t] = gp.a
        ampl_model[t] = gm.mu_nu

        delta_action = gm.update(sens[t])

        if t % 1200 == 0 or t == stime - 1:

            sim.set_box()
            sim.update(sens[t], sens_model[t])
            prederr.update([sens[t], sens_model[t]], t)
            genProcPlot.update([sens[t], ampl[t],
                                angle if box_pos[1] < min_pos else None, 0], t)
            genModPlot.update([sens_model[t], ampl_model[t],
                               angle if box_pos[1] < min_pos else None, 0], t)
