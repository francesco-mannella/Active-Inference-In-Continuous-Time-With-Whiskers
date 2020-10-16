import matplotlib.patches as patches
from sim import Sim
from aisailib import GP, GM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")


for type in ["normal", "attenuation"]:

    # %%
    gp = GP(eta=0.0005, freq=0.5, amp=0.8)
    gm = GM(eta=0.0005, freq=0.5, amp=0.8)
    sim = Sim("demo_"+type)

    # %%
    delta_action = 0
    stime = 145000

    sens = np.zeros(stime)
    ampl = np.zeros(stime)
    sens_model = np.zeros(stime)
    ampl_model = np.zeros(stime)

    peaks = 0
    peaks_max = 3
    box_time = 0
    for t in range(stime):

        gp.update(delta_action)
        gm.mu_x[2] += -1*(t == (stime//4))
        if type != "normal":
            gm.pi_s -= 8.9*(t == (3*stime//8))
            gm.pi_s += 8.9*(t == (5*stime//8))

        if peaks > peaks_max:
            gp.mu_x[2] = np.minimum(0.5, gp.mu_x[2])

        sens[t] = gp.mu_x[2]
        sens_model[t] = gm.mu_x[2]
        ampl[t] = gp.a
        ampl_model[t] = gm.mu_nu

        delta_action = gm.update(sens[t])
        if len(sens[:t+1]) >= 2:
            dd = np.sum(2*(sens[t-1:t+1] > 0)-1)
            if dd == 0 and sens[t-1] > 0:
                peaks += 1
                print(peaks)
                if peaks == peaks_max+1:
                    box_time = t
        if t % 1200 == 0 or t == stime - 1:
            if peaks > peaks_max:
                sim.set_box([0, 1.48])
            sim.update(0.3*np.pi*sens[t], 0.3*np.pi*sens_model[t])
    sim.close()

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.title("Generative process")
    s, = plt.plot(sens, c="red", lw=1, ls="dashed")
    a, = plt.plot(ampl, c="#aa6666", lw=3)
    plt.plot([0, stime], [1.5, 1.5], c="red", lw=0.5)
    plt.plot([0, stime], [1, 1], c="red", lw=0.5)
    plt.plot([0, stime], [0.5, 0.5], c="red", lw=0.5)
    plt.text(box_time-(stime//20), 1.7, "box added")
    plt.plot([box_time, box_time], [-4, 4], c="black", lw=0.3)
    plt.xticks([])
    plt.legend([s, a], ["proprioception (current angle)",
                        "action (amplitude)"])
    ax = plt.subplot(212)
    x, = plt.plot(sens_model, c="green", lw=1, ls="dashed")
    n, = plt.plot(ampl_model, c="#66aa66", lw=3)
    plt.plot([0, stime], [1.5, 1.5], c="green", lw=0.5)
    plt.plot([0, stime], [1, 1], c="green", lw=0.5)
    plt.plot([0, stime], [0.5, 0.5], c="green", lw=0.5)
    plt.text(stime//4-(stime//20), sens_model[stime//4] + 0.2, "model changed")
    plt.scatter(stime//4, sens_model[stime//4], c="green", s=100)
    if type != "normal":
        rect = patches.Rectangle((3*stime//8, -1.5), 2*stime//8, 3,
                                 edgecolor='none', facecolor=[.6, .6,.6, .3])
        ax.add_patch(rect)


    plt.legend([x, n], ["proprioception (current angle)",
                        "internal cause (amplitude)"])
    plt.savefig("demo"+type+".png")
