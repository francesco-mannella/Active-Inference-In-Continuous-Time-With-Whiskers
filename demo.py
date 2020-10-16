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
    for t in range(stime):

        gp.update(delta_action)
        gm.mu_x[2] += -1*(t==(stime//4))
        if type != "normal":
            gm.pi_s -= 8.9*(t==(3*stime//8))
            gm.pi_s += 8.9*(t==(5*stime//8))

        if peaks > 3:
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
        if t % 1200 == 0 or t == stime - 1:
            if peaks > 3:
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
    plt.xticks([])
    plt.legend([s, a], ["proprioception (current angle)",
                        "action (amplitude)"])
    plt.subplot(212)
    x, = plt.plot(sens_model, c="green", lw=1, ls="dashed")
    n, = plt.plot(ampl_model, c="#66aa66", lw=3)
    plt.plot([0, stime], [1.5, 1.5], c="green", lw=0.5)
    plt.plot([0, stime], [1, 1], c="green", lw=0.5)
    plt.plot([0, stime], [0.5, 0.5], c="green", lw=0.5)
    plt.legend([x, n], ["proprioception (current angle)",
                        "internal cause (amplitude)"])
    plt.savefig("demo"+type+".png")
