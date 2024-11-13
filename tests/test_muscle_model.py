# test friedl her muscle model
#-------------------------------
from scipy.stats import alpha

from degroote2016_muscle_model import DeGrooteMuscle

import matplotlib.pyplot as plt


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

plt.show()

