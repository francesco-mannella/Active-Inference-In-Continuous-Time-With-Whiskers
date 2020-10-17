from plotter import Plotter
from sim import Sim
from aisailib import GP, GM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use("Agg")


def attenuation(stime):
    t = np.arange(stime)
    s = (3*(stime//8) < t) & (t < 5*(stime//8))
    s = np.convolve(s, np.exp(-0.5*np.linspace(-3, 3, stime//10)**2), mode="same")
    return s/np.max(s)


for type in ["normal", "attenuation"]:
    stime = 145000

    gp = GP(eta=0.0005, freq=0.5, amp=0.8)
    gm = GM(eta=0.0005, freq=0.5, amp=0.8)

    sim = Sim("demo_"+type)
    genProcPlot = Plotter("gen_proc_"+type, type="process",
                          wallcolor=[0.2, 0.2, 0, 0.2],
                          labels={"x": "proprioception", "nu": "action"},
                          color=[.5, .2, 0], stime=stime)

    genModPlot = Plotter("gen_mod_"+type, type="model",
                         wallcolor=[0, 0, 0, 0],
                         labels={"x": "proprioception", "nu": "internal cause"},
                         color=[.2, .5, 0], stime=stime)

    delta_action = 0

    sens = np.zeros(stime)
    ampl = np.zeros(stime)
    sens_model = np.zeros(stime)
    ampl_model = np.zeros(stime)
    sigma_s = attenuation(stime)


    peaks = 0
    peaks_max = 2
    box_time = 1000*stime
    for t in range(stime):

        gp.update(delta_action)
        gm.mu_x[2] += -1*(t == (stime//4))
        if type != "normal":
            gm.pi_s = 9 - 8.9*sigma_s[t]

        if t > box_time:
            gp.mu_x[2] = np.minimum(0.5, gp.mu_x[2])

        sens[t] = gp.mu_x[2]
        sens_model[t] = gm.mu_x[2]
        ampl[t] = gp.a
        ampl_model[t] = gm.mu_nu

        delta_action = gm.update(sens[t])

        if t == stime//4 and type=="normal":
            genModPlot.plot_first(t)
        if t == box_time and type=="normal":
            genProcPlot.plot_second(t)

        if len(sens[:t+1]) >= 2:
            dd = np.sum(2*(sens[t-1:t+1] > 0)-1)
            if dd == 0 and sens[t-1] > 0:
                peaks += 1
                if peaks == peaks_max+1:
                    box_time = t + stime//6
        if t % 1200 == 0 or t == stime - 1:
            if t > box_time:
                sim.set_box([0, 1.48])
            sim.update(0.3*np.pi*sens[t],
                       0.3*np.pi*sens_model[t])
            genProcPlot.update([0.3*np.pi*sens[t],
                                0.3*np.pi*ampl[t],
                                t>box_time,
                                0], t)
            genModPlot.update([0.3*np.pi*sens_model[t],
                               0.3*np.pi*ampl_model[t],
                               t>box_time,
                               sigma_s[t]*3 if type!= "normal" else 0], t)
    sim.close()
