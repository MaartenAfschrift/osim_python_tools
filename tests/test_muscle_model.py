# test friedl her muscle model
#-------------------------------
from scipy.stats import alpha

from degroote2016_muscle_model import DeGrooteMuscle

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

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

# simple simulation with this muscle
# Test forward simulation with muscle model
def muscle_state_derivative(lmtilde, t, muscle):
    # dummy muscle activation as a function of time
    a = 0.2 * np.sin(2 * t) + 0.3
    # set the state of the muscle
    muscle.set_activation(a)
    muscle.set_norm_fiber_length(lmtilde)
    # get the state derivative
    lmtilde_dot = muscle.compute_norm_fiber_length_dot()
    return lmtilde_dot


t = np.linspace(0, 5, 500)
x0 = 1
x = odeint(muscle_state_derivative, x0, t, args=(muscle,))

plt.figure()
plt.plot(t, x)
plt.xlabel('time (s)')
plt.ylabel('normalized fiber length')



# test inverse force length tendon
norm_tendon_length = np.linspace(1,1.2,30)
ft = muscle.force_length_tendon(np.linspace(1,1.2,30),35, 0)
norm_tendon_length2 = muscle.inverse_force_length_tendon(ft,35,0)
print(norm_tendon_length - norm_tendon_length2)

plt.figure()
plt.plot(norm_tendon_length, ft)
plt.plot(norm_tendon_length2, ft)
plt.xlabel('normalized tendon length')
plt.ylabel('tendon force [N]')
# plot figures
plt.show()

