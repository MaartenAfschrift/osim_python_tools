

from degroote2016_muscle_model import DeGrooteMuscle

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Simple example to simulate a muscle actuated pointmass
# the main purpose of this example was to introduce myself to ode solvers in python
# the second purpose was to test some ideas prestend in the paper " Does elastic energy enhance work and efficiency
# in the stretch-shortening cycle?" by van Ingen Schenau, M. Bobbert and Arnold de Haan


# create a muscle
FMo = 1000 # maximal isometric force in N
lMo = 0.2 # optimal fiber length in m
lTs = 0.5 # tendon slack length in m
alpha = 0.1 # optimal pennation angle in rad
vMtildemax = 10 # maximal muscle fiber velocity in lMo/s
kT = 35 #

# create muscle object
muscle = DeGrooteMuscle(FMo, lMo, lTs, alpha, vMtildemax, kT)

# simple simulation with this muscle
# Test forward simulation with muscle model
def muscle_state_derivative(lmtilde, muscle, lmt):
    # assumed activation one
    a = 1
    # set the state of the muscle
    muscle.set_activation(a)
    muscle.set_norm_fiber_length(lmtilde)
    muscle.set_muscle_tendon_length(lmt)
    # get the state derivative
    lmtilde_dot = muscle.compute_norm_fiber_length_dot()
    return lmtilde_dot

# state derivative of pointmass
def pointmass_state_derivative(pos, vel, muscle, mass):
    # get the muscle force
    Fm = muscle.get_force()
    # get the acceleration (with gravity)
    Gy = -9.81 * mass
    acc = (Fm + Gy) / mass
    xdot = [vel, acc]
    return xdot

# state derivative of the system
def dx_dt(t, x, muscle, mass):
    pos = x[0]
    vel = x[1]
    lmtilde = x[2]
    # muscle dynamics
    a = 1
    muscle.set_activation(a)
    muscle.set_muscle_tendon_length(-pos)
    muscle.set_norm_fiber_length(lmtilde)
    lmtilde_dot = muscle.compute_norm_fiber_length_dot()

    # pointmass dynamics
    fm = muscle.get_tendon_force()
    gy = -9.81 * mass
    acc = (fm + gy) / mass
    xdot = [vel, acc, lmtilde_dot]
    return(xdot)

# stop condition for the ode solver
# simulation should stop if normalised muscle length is smaller than 0.4
def stop_condition(t, x, muscle, mass):
    return x[2] - 0.4

stop_condition.terminal = True  # Tell solve_ivp to stop when this condition is met
stop_condition.direction = -1   # Only trigger when crossing zero in the negative direction


t0 = 0
y0 = [-(lTs + lMo), 0 , 1] # initial state of the pointmass [position, velocity, norm fiber length]
t_span = (t0, 1)
mass = 5

# Solve the differential equation with stop condition
solution = solve_ivp(
    dx_dt,
    t_span,
    y0,
    method='RK45',
    rtol=1e-5,
    atol=1e-6,
    events=stop_condition,  # Add the stop condition
    args=(muscle, mass) # additional arguments
)

plt.figure()
plt.subplot(1,3,1)
plt.plot(solution.t, solution.y[0])
plt.subplot(1,3,2)
plt.plot(solution.t, solution.y[1])
plt.subplot(1,3,3)
plt.plot(solution.t, solution.y[2])


# compute mechanical work done by the muscle
Ekin0 = 0.5 * mass * solution.y[1][0]**2
Ekin_end = 0.5* mass * solution.y[1][-1]**2
Epot_end = 9.81* mass * (solution.y[0][-1] - solution.y[0][0])
Emech = Ekin_end - Ekin0
print('Mechanical work done by the muscle: ', Emech, ' J')
print('Final kinetic and potential energy of the pointmass: ', Ekin_end +Epot_end, ' J')

# show figures
plt.show()