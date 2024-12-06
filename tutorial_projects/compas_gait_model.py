
# simulate a model with pointmasses at the feet, two rigid legs and a pointmass at the pelvis.
# the model will walk with a gentle push in the back to simulate a slope

import sympy as sym
from sympy.utilities.lambdify import lambdify, implemented_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# simulation information
Settings ={}
Settings['dt'] = 0.001
Settings['OdeIntegrator'] = True
Settings['tfinal']  = 10
Settings['init_state'] = [0, 0, 0, 0]  # initial state of the system [fi1, fi2, fid1, fid2]

# model parameters
L1 =1
L2 = 1
m1 = 5
m2 = 5
m3 = 70-m1-m2

#--------------------------------------------
#  --------- Equations of motion ------------
#--------------------------------------------

# construct equations of motion [generalized coordinates, generalized velocities]
fi1, fid1, fidd1 = sym.symbols('fi1 fid1 fidd1', real=True)
fi2, fid2, fidd2 = sym.symbols('fi2 fid2 fidd2', real=True)
q = sym.Matrix([fi1, fi2])
q_dot = sym.Matrix([fid1, fid2])
q_ddot = sym.Matrix([fidd1, fidd2])

# compute kinematics of the point masses
r_m1 = sym.Matrix([0, 0])  # location COM1
r_m3 = L1 * sym.Matrix([sym.cos(fi1), sym.sin(fi1)])  # location COM3
r_m2 = r_m3 + L2 * sym.Matrix([sym.cos(fi2), sym.sin(fi2)])  # location COM2

r_m1_dot = sym.Matrix([0, 0])  # velocity COM1
r_m3_dot = r_m3.jacobian(q) * q_dot  # velocity COM3
r_m2_dot = r_m2.jacobian(q) * q_dot  # v # velocity COM2

r_m1_ddot = sym.Matrix([0, 0])  # acceleration COM1
r_m3_ddot = r_m3_dot.jacobian(q) * q_dot + r_m3_dot.jacobian(q_dot) * q_ddot  # acceleration COM3
r_m2_ddot = r_m2_dot.jacobian(q) * q_dot + r_m2_dot.jacobian(q_dot) * q_ddot  # acceleration COM3

# compute potential and kinetic energy of the system
Ekin = 0.5 * m1 * r_m1_dot.dot(r_m1_dot) + 0.5 * m2 * r_m2_dot.dot(r_m2_dot) + 0.5 * m3 * r_m3_dot.dot(r_m3_dot)
Epot = (m1 * r_m1[1] + m2 * r_m2[1] + m3 * r_m3[1]) * 9.81

# states
x = sym.Matrix([fi1, fi2, fid1, fid2])
xdot = sym.Matrix([fid1, fid2, fidd1, fidd2])

# Lagrange method:
L = Ekin - Epot
DL_Dq = sym.diff(L, q)
DL_Ddq = sym.diff(L, q_dot)
DDL_DtDdq = DL_Ddq.jacobian(x) * xdot
EoM = DL_Dq - DDL_DtDdq

# We know that our equations are of the form:
#             EoM = M(q,dq)*qdd + f(q,dq) == 0;
M = EoM.jacobian(q_ddot)  # mass matrix
CG = EoM.subs([(fidd1, 0), (fidd2, 0)])  # gravity and coriolis terms
QDD = -M.inv()*CG  # forward dynamics
tau = -EoM

# create functions to evaluate M and CG
f_M = lambdify([fi1, fi2], M)
f_CG = lambdify([fi1, fi2, fid1, fid2], CG)  # create function
f_M_inv = lambdify([fi1, fi2], M.inv())
f_qdd = lambdify([fi1, fi2, fid1, fid2], QDD)
f_Ekin = lambdify([fi1, fi2, fid1, fid2], Ekin)
f_EPot = lambdify([fi1, fi2], Epot)
f_rm2 = lambdify([fi1, fi2], r_m2)
f_rm3 = lambdify([fi1], r_m3)
f_id = lambdify([fi1, fi2, fid1, fid2], tau)

#--------------------------------------------
#  ---------    HEELSTRIKE MAP   ------------
#--------------------------------------------

# we want to collision here of the pointmass m2 with the ground with conservation of angular momentum
# velocity_m2 = 0
fi1_post = fi2
fi2_post = fi1
fid1_post, fidd1_post = sym.symbols('fid1_post fidd1_post', real=True)
fid2_post, fidd2_post = sym.symbols('fid2_post fidd2_post', real=True)
q_post = sym.Matrix([fi1_post, fi2_post])
q_dot_post = sym.Matrix([fid1_post, fid2_post])
q_ddot_post = sym.Matrix([fidd1_post, fidd2_post])

# compute kinematics of the point masses  post collision
r_m1_post = sym.Matrix([0, 0])  # location COM1
r_m3_post = L1 * sym.Matrix([sym.cos(fi1_post), sym.sin(fi1_post)])  # location COM3
r_m2_post = r_m3_post + L2 * sym.Matrix([sym.cos(fi2_post), sym.sin(fi2_post)])  # location COM2
r_m1_dot_post = sym.Matrix([0, 0])  # velocity COM1 post collision
r_m3_dot_post = r_m3_post.jacobian(q_post) * q_dot_post  # velocity COM3
r_m2_dot_post = r_m2_post.jacobian(q_post) * q_dot_post  # v # velocity COM2
r_m1_ddot_post = sym.Array([0, 0])  # acceleration COM1 post collision
r_m3_ddot_post = r_m3_dot_post.jacobian(q_post) * q_dot_post + r_m3_dot_post.jacobian(q_dot_post) * q_ddot_post  # acceleration COM3
r_m2_ddot_post = r_m2_dot_post.jacobian(q_post) * q_dot_post + r_m2_dot_post.jacobian(q_dot_post) * q_ddot_post  # acceleration COM3

def cross_2d(r,v):
    return r[0]*v[1] - r[1]*v[0]
# angular momentum before collision (w.r.t. point collision and joints)
L_r_m2 = cross_2d(r_m1- r_m2, m1 * r_m1_dot) + cross_2d(r_m3- r_m2, m3 * r_m3_dot)
L_r_hip = cross_2d(r_m1- r_m3, m1 * r_m1_dot)# velocity w.r.t. hip or in world frame ?

# angular momentum post collision (w.r.t. point collision and joints)
L_post_m1 = cross_2d(r_m2_post - r_m1_post, m2 * r_m2_dot_post) + cross_2d(r_m3_post - r_m1_post, m3 * r_m3_dot_post)
L_hip_post = cross_2d(r_m2_post- r_m3_post, m2 *  r_m2_dot_post)

# conservation of angular momentum
HS_L_ct = L_r_m2 - L_post_m1
HS_Lhip_ct =  L_r_hip - L_hip_post

# solve system of equations
solution = sym.solve((sym.Eq(HS_L_ct,0), sym.Eq(HS_Lhip_ct,0)), (fid1_post, fid2_post))

# equation to compute states post collision
f_x_hs_map = lambdify([fi1, fi2, fid1, fid2], [fi1_post, fi2_post, solution[fid1_post], solution[fid2_post]])


#--------------------------------------------
#  ---------     Simulation      ------------
#--------------------------------------------

def dx_dt(t, x):
    qdd = f_qdd(x[0], x[1], x[2], x[3])
    return [x[2], x[3], qdd[0][0], qdd[1][0]]



if Settings['OdeIntegrator']:
    t_span = (0, Settings['tfinal'])
    xinit = Settings['init_state']
    solution = solve_ivp(
        dx_dt,
        t_span,
        xinit,
        method='RK45',
        rtol=1e-10,
        atol=1e-10
    )
    # unpack solution
    time = solution.t
    state_matrix = solution.y.T

else:
    # Time parameters
    T_total = Settings['tfinal'] # total simulation time (s)
    dt = Settings['dt']
    time = np.arange(0, T_total, dt) # time vector

    # pre allocate state matrix
    state_matrix = np.zeros((len(time)+1, 4))
    angle_legs = 60*np.pi/180
    fi1_0 = np.pi-np.pi/2-angle_legs/2
    fi2_0 = 2* np.pi - fi1_0
    x0 = np.array([fi1_0, fi2_0, 0, 0])  # initial state of the system [fi1, fi2, fid1, fid2]
    state_matrix[0,:] = x0
    ct = 0
    for t in time:
        # extract state
        x = state_matrix[ct,:]

        # compute state derivative
        qdd = f_qdd(x[0], x[1], x[2], x[3])
        x_dot = np.array([x[2], x[3], qdd[0][0] , qdd[1][0]])

        state_matrix[ct+1, :] = x + x_dot*dt
        ct =ct +1
    state_matrix = state_matrix[0:-1,:]

#--------------------------------------------
#  ---------     Visualisation   ------------
#--------------------------------------------

# plot states
plt.figure()
for i in range(0,4):
    plt.subplot(1,4,i+1)
    plt.plot(time, state_matrix[:,i])

# compute energy
EkinV = np.zeros(len(time))
EpotV = np.zeros(len(time))
for i in range(0, len(time)):
    EkinV[i] = f_Ekin(state_matrix[i,0], state_matrix[i,1], state_matrix[i,2], state_matrix[i,3])
    EpotV[i] = f_EPot(state_matrix[i,0], state_matrix[i,1])
plt.figure()
plt.plot(time, EkinV+EpotV)
# set limits on y values axis
plt.gca().set_ylim([min(EkinV+EpotV-5), max(EkinV+EpotV+5)])


# test animation
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_title("Stick Figure Walking Animation")
ax.set(xlim=[-10, 10], ylim=[-2.1, 2.1], xlabel='Time [s]', ylabel='Angle [rad]')

# Line to represent the data
leg1,  = ax.plot([], [], 'b-', lw=2)  # Create a line for animation
leg2,  = ax.plot([], [], 'r-', lw=2)  # Create a line for animation

# interpolate for animation
time_int = np.arange(time[0], time[-1], 0.01)
f_cubic = [interp1d(time.flatten(), state_row, kind='cubic') for state_row in state_matrix.T]
state_matrix_int= np.array([f(time_int) for f in f_cubic]).T

def update(frame):
    if not hasattr(update, "xb"):
        update.xb = 0  # Initialize on the first call

    # extract plot points
    r_m2 = f_rm2(state_matrix_int[frame, 0], state_matrix[frame, 1])
    r_m3 = f_rm3(state_matrix_int[frame, 0])

    # test condition for heelstrike
    #if frame > 0:
    #    if (r_m2[1][0]>0) & (state_matrix[frame,1]<(np.pi-0.05)):
    #        update.xb = update.xb +  r_m2[0][0]

    leg1.set_data([update.xb , update.xb + r_m3[0][0]],
                  [0, r_m3[1][0]])
    leg2.set_data([update.xb + r_m3[0][0], update.xb + r_m2[0][0]],
                  [update.xb + r_m3[1][0], update.xb + r_m2[1][0]])
    # Update x and y data
    return leg1, leg2

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(time_int), blit=True, interval=10)



#--------------------------------------------
#  ---------     TESTS           ------------
#--------------------------------------------


# test equation of motion in a specific state
fi1_val = 0
fi2_val = 0
fid1_val = 0
fid2_val = 0
qdd_val = np.matmul(f_M_inv(fi1_val,fi2_val), f_CG(fi1_val,fi2_val,fid1_val,fid2_val))
qdd_val2 = f_qdd(fi1_val,fi2_val,fid1_val,fid2_val)

# test heelstrike map
hs_map = f_x_hs_map(fi1_val, fi2_val, fid1_val, fid2_val)

plt.show()





