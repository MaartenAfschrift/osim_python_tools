# test friedl her muscle model
#-------------------------------
from scipy.stats import alpha

from degroote2016_muscle_model import DeGrooteMuscle

import matplotlib.pyplot as plt
import numpy as np

import sympy as sp


# create a muscle
FMo = 1000 # maximal isometric force in N
lMo = 0.2 # optimal fiber length in m
lTs = 0.5 # tendon slack length in m
alpha = 0.1 # optimal pennation angle in rad
vMtildemax = 10 # maximal muscle fiber velocity in lMo/s
kT = 35 #

# create muscle object
muscle = DeGrooteMuscle(FMo, lMo, lTs, alpha, vMtildemax, kT)

# plot normalized force-length curve
muscle.plot_fl_curve_norm()
muscle.plot_fl_curve()

# plot normalized force-velocity curve
muscle.plot_fv_curve_norm()
muscle.plot_fv_curve()


# test explicit formulation
muscle.set_activation(0.5)
muscle.set_norm_fiber_length(1)
muscle.set_muscle_tendon_length(0.715)
lmtilde_dot = muscle.compute_norm_fiber_length_dot()


# test implicit formulation
muscle.set_activation(0.5)
muscle.set_norm_fiber_length(1)
muscle.set_muscle_tendon_length(0.715)
muscle.set_norm_fiber_velocity(lmtilde_dot/10)
hill_err = muscle.compute_hill_equilibrium()


# plot figures
plt.show()

