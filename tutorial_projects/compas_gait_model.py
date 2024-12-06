
# simulate a model with pointmasses at the feet, two rigid legs and a pointmass at the pelvis.
# the model will walk with a gentle push in the back to simulate a slope

import sympy as sym
from sympy.utilities.lambdify import lambdify, implemented_function
import numpy as np

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

r_m1_ddot = sym.Array([0, 0])  # acceleration COM1
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
M = EoM.jacobian(q_ddot)  # mass matrix
CG = EoM.subs([(fidd1, 0), (fidd2, 0)])  # gravity and coriolis terms
QDD = M.inv()*CG

# create functions to evaluate M and CG
f_M = lambdify([fi1, fi2], M)
f_CG = lambdify([fi1, fi2, fid1, fid2], CG)  # create function
f_M_inv = lambdify([fi1, fi2], M.inv())
f_qdd = lambdify([fi1, fi2, fid1, fid2], QDD)
f_Ekin = lambdify([fi1, fi2, fid1, fid2], Ekin)
f_EPot = lambdify([fi1, fi2, fid1, fid2], Epot)

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
# angular momentum before collision (w.r.t. position m2
L_r_m2 = cross_2d(r_m1- r_m2, m1 * r_m1_dot) + cross_2d(r_m3- r_m2, m3 * r_m3_dot)
L_r_hip = cross_2d(r_m1- r_m3, m1 * r_m1_dot)# velocity w.r.t. hip or in world frame ?

# angular momentum post collision (we assume here that leg 1 becomes leg 2 and vice versa)
L_post_m1 = cross_2d(r_m2_post - r_m1_post, m2 * r_m2_dot_post) + cross_2d(r_m3_post - r_m1_post, m3 * r_m3_dot_post)
L_hip_post = cross_2d(r_m2_post- r_m3_post, m2 *  r_m2_dot_post)

# conservation of angular momentum
HS_L_ct = L_r_m2 - L_post_m1
HS_Lhip_ct =  L_r_hip - L_hip_post

# so there is apparantly no solution for this system of equations
solution = sym.solve((sym.Eq(HS_L_ct,0), sym.Eq(HS_Lhip_ct,0)), (fid1_post, fid2_post))

# equation to compute states post collision
f_x_hs_map = lambdify([fi1, fi2, fid1, fid2], [fi1_post, fi2_post, solution[fid1_post], solution[fid2_post]])


#--------------------------------------------
#  ---------     Simulation      ------------
#--------------------------------------------

# simulation with our pendulum model









#--------------------------------------------
#  ---------     TESTS           ------------
#--------------------------------------------


# test equation of motion in a specific state
fi1_val = 0.5
fi2_val = 0.1
fid1_val = 1
fid2_val = 2
qdd_val = np.matmul(f_M_inv(fi1_val,fi2_val), f_CG(fi1_val,fi2_val,fid1_val,fid2_val))
qdd_val2 = f_qdd(fi1_val,fi2_val,fid1_val,fid2_val)

# test heelstrike map
hs_map = f_x_hs_map(fi1_val, fi2_val, fid1_val, fid2_val)

# compute kinetic energy energy post collision
Ekin







