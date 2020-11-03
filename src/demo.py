from plotter import Plotter, PredErrPlotter
from sim import Sim
from aisailib import GP, GM
import numpy as np


sidewidth = 0.8
sideheight = 1.5
center = [0, 1]
normal_box = np.array([
    (-sidewidth, -sideheight),
    (sidewidth, -sideheight),
    (sidewidth, sideheight),
    (-sidewidth, sideheight)]) + center

sidewidth = 1.3
sideheight = 1.5
center = [0, 1]
large_box = np.array([
    (-sidewidth, -sideheight),
    (sidewidth, -sideheight),
    (sidewidth, sideheight),
    (-sidewidth, sideheight)]) + center

for type in ["normal", "large", "still"]:

    print("simulating", type, "...")

    stime = 180000

    gp = GP(dt=0.0005, freq=0.5, amp=1.2)
    gm = GM(dt=0.0005, freq=0.5, amp=1.2)

    sim = Sim("demo_"+type, points=normal_box
              if type == "normal" or type == "still" else large_box)
    prederr = PredErrPlotter("prederr", type, stime)
    genProcPlot = Plotter("gen_proc_"+type, type="process",
                          wallcolor=[0.2, 0.2, 0, 0.2],
                          labels={"x": "proprioception",
                                  "nu": "action (oscil. ampl.)"},
                          color=[.5, .2, 0], stime=stime)

    genModPlot = Plotter("gen_mod_"+type, type="model",
                         wallcolor=[0, 0, 0, 0],
                         labels={"x": "proprioception prediction",
                                 "nu": "internal cause (repr. oscill. ampl.)"},
                         color=[.2, .5, 0], stime=stime)

    delta_action = 0

    sens = np.zeros(stime)
    ampl = np.zeros(stime)
    sens_model = np.zeros(stime)
    ampl_model = np.zeros(stime)

    frame = 0
    for t in range(stime):

        # compute box position
        if type == "normal" or type == "large":
            box_pos = np.array([0,
                                np.maximum(1.3, 2.2*np.exp(-3*t/stime)+0.7)])
        else:
            box_pos = np.array([0, 5]) if t < stime*(36/100) \
                else np.array([0, 1.48])
        sim.move_box(box_pos)

        # compute collision
        collision, curr_angle_limit = sim.detect_collision()

        # update process
        gp.update(delta_action)
        gp.mu_x[2] = np.minimum(curr_angle_limit, gp.mu_x[2])

        # get state
        sens[t] = gp.mu_x[2]
        sens_model[t] = gm.mu_x[2]
        ampl[t] = gp.a
        ampl_model[t] = gm.mu_nu

        # update model and action
        delta_action = gm.update(sens[t])

        # plot
        if t % 1200 == 0 or t == stime - 1:

            print(frame)
            frame += 1

            sim.set_box()
            sim.update(sens[t], sens_model[t])
            prederr.update([sens[t], sens_model[t]], t)
            genProcPlot.update([sens[t], ampl[t],
                                curr_angle_limit if collision is True else None, 0], t)
            genModPlot.update([sens_model[t], ampl_model[t],
                               curr_angle_limit if collision is True else None, 0], t)
