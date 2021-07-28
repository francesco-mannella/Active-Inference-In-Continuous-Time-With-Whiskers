from plotter import Plotter, PredErrPlotter
from sim import Sim, a2xy
from GP import GP
from GM import GM
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type",
                    default="still",
                    help="type of demo. one of 'still', 'normal', ''large")
args = parser.parse_args()
type = args.type


print("simulating", type, "...")

stime = 20000

gp = GP(dt=0.01, omega2_GP=(0.5**2), alpha=[1., 1.])
gm = GM(dt=0.01, eta=0.001,
        eta_d=1.0, eta_a=0.01, eta_nu=0.002)
sim = Sim("demo_"+type, type, stime)
plotter = Plotter(sim, stime, type)

delta_action = np.zeros(2)

frame = 0
for t in range(stime):

    # move box with scheduling based on type
    # and conpute collision
    collision, curr_angle_limit = sim.move_box(t)

    # update process
    gp.effective_object_position[0] = curr_angle_limit
    gp.update(delta_action)

    #update model
    delta_action = gm.update(gp.s_t[0], gp.s_p[0], gp.cpg[0])

    # update data
    plotter.update(t, gm, gp, curr_angle_limit, collision)

    # plot
    if t % int(stime/200) == 0 or t == stime - 1:

        print(frame)
        frame += 1
        sim.update(gp.x[0], gm.mu[0])
        plotter.draw()
