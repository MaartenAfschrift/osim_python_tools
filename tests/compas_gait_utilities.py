# utilities for simulation with compas gait model
#-----------------------------------------------------

# generate equations of motion for compas gait model

import sympy as sym
import numpy as np
from sympy.utilities.lambdify import lambdify, implemented_function
from scipy.integrate import solve_ivp

# function to generate equations of motion
def gen_equations_of_motion(L1, L2, m1, m2, m3):

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
    r_m3_ddot_post = r_m3_dot_post.jacobian(q_post) * q_dot_post + r_m3_dot_post.jacobian(
        q_dot_post) * q_ddot_post  # acceleration COM3
    r_m2_ddot_post = r_m2_dot_post.jacobian(q_post) * q_dot_post + r_m2_dot_post.jacobian(
        q_dot_post) * q_ddot_post  # acceleration COM3

    def cross_2d(r, v):
        return r[0] * v[1] - r[1] * v[0]

    # angular momentum before collision (w.r.t. point collision and joints)
    L_r_m2 = cross_2d(r_m1 - r_m2, m1 * r_m1_dot) + cross_2d(r_m3 - r_m2, m3 * r_m3_dot)
    L_r_hip = cross_2d(r_m1 - r_m3, m1 * r_m1_dot)  # velocity w.r.t. hip or in world frame ?

    # angular momentum post collision (w.r.t. point collision and joints)
    L_post_m1 = cross_2d(r_m2_post - r_m1_post, m2 * r_m2_dot_post) + cross_2d(r_m3_post - r_m1_post,
                                                                               m3 * r_m3_dot_post)
    L_hip_post = cross_2d(r_m2_post - r_m3_post, m2 * r_m2_dot_post)

    # conservation of angular momentum
    HS_L_ct = L_r_m2 - L_post_m1
    HS_Lhip_ct = L_r_hip - L_hip_post

    # solve system of equations
    solution = sym.solve((sym.Eq(HS_L_ct, 0), sym.Eq(HS_Lhip_ct, 0)), (fid1_post, fid2_post))

    # equation to compute states post collision
    f_x_hs_map = lambdify([fi1, fi2, fid1, fid2], [fi1_post, fi2_post, solution[fid1_post], solution[fid2_post]])



    # create functions to evaluate M and CG
    func_set = {}
    func_set['f_M'] = lambdify([fi1, fi2], M)
    func_set['f_CG'] = lambdify([fi1, fi2, fid1, fid2], CG)  # create function
    func_set['f_M_inv'] = lambdify([fi1, fi2], M.inv())
    func_set['f_qdd'] = lambdify([fi1, fi2, fid1, fid2], QDD)
    func_set['f_Ekin'] = lambdify([fi1, fi2, fid1, fid2], Ekin)
    func_set['f_EPot'] = lambdify([fi1, fi2], Epot)
    func_set['f_rm2'] = lambdify([fi1, fi2], r_m2)
    func_set['f_rm3'] = lambdify([fi1], r_m3)
    func_set['f_id'] = lambdify([fi1, fi2, fid1, fid2], tau)
    func_set['f_hs_map'] = f_x_hs_map
    return(func_set)


# generate equations of motion for compas gait model
def forward_sim(Settings, func_set):
    # unpack functions
    f_qdd = func_set['f_qdd']
    f_x_hs_map = func_set['f_hs_map']
    # internal function
    def dx_dt(t, x):
        qdd = f_qdd(x[0], x[1], x[2], x[3])
        return [x[2], x[3], qdd[0][0], qdd[1][0]]

    def step_transition(t, x):
        # check if condition for step transition is met
        fi1 = x[0]
        fi2 = x[1]
        condition = 99
        if fi1 < np.pi:
            condition = fi1 + fi2 - 2 * np.pi
        return condition


    t_span = (0, Settings['tfinal'])
    xinit = Settings['init_state']
    solution = solve_ivp(
        dx_dt,
        t_span,
        xinit,
        method='RK45',
        rtol=1e-10,
        atol=1e-10,
        events = step_transition
    )
    t_end = solution.t[-1]
    solution_store = []
    solution_store.append(solution)
    if not(Settings['sim_one_cycle']):
        while (t_end<Settings['tfinal'] ) & (len(solution_store)<Settings['max_steps']):
            # iterate simulation
            t_span = (t_end, Settings['tfinal'])
            x_pre_collision = solution.y[:, -1]
            #use collision map
            x_post_collision = f_x_hs_map(x_pre_collision[0], x_pre_collision[1], x_pre_collision[2], x_pre_collision[3])
            xinit = x_post_collision
            solution = solve_ivp(
                dx_dt,
                t_span,
                xinit,
                method='RK45',
                rtol=1e-10,
                atol=1e-10,
                events=step_transition
            )
            t_end = solution.t[-1]
            solution_store.append(solution)
            print(len(solution_store))

    return solution_store



# heelstrike map function



# define function for forward simulation


