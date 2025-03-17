# Tools for muscle redundancy solver in python
#---------------------------------------------
from numpy.distutils.conv_template import header

from osim_utilities import osim_subject
import casadi as ca
import opensim as osim
from osim_utilities import readMotionFile
from general_utilities import lowPassFilterDataFrame
import numpy as np
from pathlib import Path
from degroote2016_muscle_model import DeGrooteMuscle
import pandas as pd
import os
import pickle

# plot functions for debugging
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures



# List of tools:
# 0. tool for settings
# 1. tool to select muscles in model (that span selected dofs)
# 2. lmt and moment arm api
# 3. formulate and solve optimal control problem
# 4. output solution

# approach: create class for redundancy solver

class muscle_redundancy_solver:
    def __init__(self, modelfile, ikfile, idfile, dofs, muscles_selected = None,
                 outpath = None):

        # inputs
        self.modelfile = modelfile
        self.ikfile = ikfile
        self.idfile = idfile
        self.dofs = dofs # currently a required input arguments. In the future develop method to identify this if input is None
        self.out_path = outpath

        # create osim_subject object from inputs
        self.my_subject = osim_subject(modelfile)

        # read ik files
        self.my_subject.set_ikfiles(self.ikfile)
        self.my_subject.read_ikfiles()
        self.ikdat = self.my_subject.ikdat

        # read id files
        self.iddat = []
        # convert to a list if needed
        if not isinstance(idfile, list):
            self.idfiles = [idfile]
        else:
            self.idfiles = idfile
        # convert to Path objects
        if not isinstance(self.idfiles[0], Path):
            self.idfiles = [Path(i) for i in self.idfiles]
        for file in self.idfiles:
            id_dat_sel = readMotionFile(file)
            self.iddat.append(id_dat_sel)

        # read model file
        self.model = osim.Model(modelfile)

        # get muscles in model
        self.my_subject.get_n_muscles()
        self.muscle_names = self.my_subject.muscle_names
        self.muscles_selected = muscles_selected
        self.degroote_muscles = None
        if self.muscles_selected is not None:
            self.init_muscle_model()

        # some other variables
        self.lmt_dat = None
        self.dm_dat = None
        self.solution = None
        self.sol_static_opt = None
        self.filename = None # name output file

    def identify_dofs(self):
        if self.dofs is None:
            # identify all dofs actuated by muscles here
            print('test')

    def identify_muscles(self):
        # approach is quite simple here: we identify all muscles that span the dofs. this means that lmt of the muscles
        # changes when the dofs are changed.

        # first check if dofs are set
        if self.dofs is None:
            self.identify_dofs()

        # test if lmt changes when dofs values are changed
        state = self.model.initSystem()
        self.muscles_selected = []
        imuscles = []
        for dof in self.dofs:
            # set dof value to 0
            qdefault = self.model.getCoordinateSet().get(dof).getValue(state)
            self.model.getCoordinateSet().get(dof).setValue(state, qdefault - 0.1)
            self.model.realizePosition(state)
            # compute lmt for all muscles
            lmt = np.zeros([self.model.getMuscles().getSize(), 2])
            muscles = self.model.getMuscles()
            for im in range(0, self.model.getMuscles().getSize()):
                muscle = muscles.get(im)
                lmt[im, 0] = muscle.getLength(state)

            self.model.getCoordinateSet().get(dof).setValue(state, qdefault + 0.1)
            self.model.realizePosition(state)
            # compute lmt for all muscles
            for im in range(0, self.model.getMuscles().getSize()):
                muscle = muscles.get(im)
                lmt[im, 1] = muscle.getLength(state)

            # find muscles with changes in lmt
            lmt_diff = np.abs(np.diff(lmt))
            imuscles_sel =  np.where(lmt_diff > 0.00001)[0]
            imuscles.append(imuscles_sel)

        # store muscles that span one of the dofs
        self.imuscles_selected = np.unique(np.concatenate(imuscles))
        self.muscles_selected = [self.muscle_names[i] for i in self.imuscles_selected]
        # default values for muscles
        self.vmax = np.zeros([len(self.muscles_selected)])+10
        self.Atendon = np.zeros([len(self.muscles_selected)])+35
        self.get_muscle_properties()

        # overwrite degroote muscle objects
        self.init_muscle_model(bool_overwrite=True)

        # return muscles selected
        return(self.muscles_selected)

    def compute_lmt_dm(self, tstart = None, tend = None):
        # use osim_subject class to do this
        self.my_subject.compute_lmt(selected_muscles = self.muscles_selected,
                                    tstart = tstart, tend = tend)
        self.my_subject.compute_dM(selected_muscles = self.muscles_selected,
                                   selected_dofs = self.dofs,
                                   tstart = tstart, tend = tend)
        self.lmt_dat = self.my_subject.lmt_dat
        self.dm_dat = self.my_subject.dm_dat

    def filter_inputs(self, order = 2, cutoff_frequency = 6, tstart = None, tend = None):
        if self.lmt_dat is None:
            self.compute_lmt_dm(tstart=tstart, tend=tend)

        ntrials = len(self.iddat)
        for itrial in range(ntrials):
            # filter moment
            self.iddat[itrial] = lowPassFilterDataFrame(self.iddat[itrial], cutoff_frequency, order)
            # filter muscle-tendon-lengths
            self.lmt_dat[itrial] = lowPassFilterDataFrame(self.lmt_dat[itrial], cutoff_frequency, order)
            # filter moment arms
            self.dm_dat[itrial] = lowPassFilterDataFrame(self.dm_dat[itrial], cutoff_frequency, order)
            # ik data
            self.ikdat[itrial] = lowPassFilterDataFrame(self.ikdat[itrial], cutoff_frequency, order)

    def init_muscle_model(self, bool_overwrite = False):

        if (self.degroote_muscles is None) or bool_overwrite:
            # get muscle properties
            [lm_opt, fisom_opt, tendon_slack, alpha_opt] = self.get_muscle_properties()

            # init degroote2016 muscle model
            nmuscles = len(self.muscles_selected)
            self.degroote_muscles = []
            for m in range(nmuscles):
                vmax_default = 10 # maximal muscle fiber velocity in lMo/s
                kT_default = 35 # tendon F/l
                self.degroote_muscles.append(DeGrooteMuscle(fisom_opt[m], lm_opt[m],
                                                            tendon_slack[m], alpha_opt[m],
                                                            vmax_default, kT_default))

    def interpolate_inputs(self, dt = 0.01, tstart = None, tend = None, itrial = 0):

        # first version for a single trial
        iddat = self.iddat[itrial]
        lmt_dat = self.lmt_dat[itrial]
        dm_dat = self.dm_dat[itrial]
        ikdat = self.ikdat[itrial]
        # convert all columns in dataframe ik dat with angle to radians
        for col in ikdat.columns:
            if 'angle' in col:
                ikdat[col] = ikdat[col] * np.pi / 180



        # Create a discrete time axis
        if tstart is None:
            tstart = iddat.time.iloc[0]
        else:
            if tstart < iddat.time.iloc[0]:
                tstart = iddat.time.iloc[0]

        if tend is None:
            tend = iddat.time.iloc[-1]
        else:
            if tend > iddat.time.iloc[-1]:
                tend = iddat.time.iloc[-1]
        t = np.arange(tstart, tend, dt)
        N = len(t)

        # interpolate moment arms and muscle-tendon lengths to the time axis (t)
        # for moment arms create a matrix with moment arms for each dofs [nmuscles, N, ndof]
        nmuscles = len(self.muscles_selected)
        ndof = len(self.dofs)
        dm = np.zeros([nmuscles, N, ndof])
        for i in range(0, ndof):
            muscle_inds = self.my_subject.dofs_dm[self.dofs[i]]
            for j in muscle_inds:
                # get name of this muscle
                dm_name = self.muscle_names[j] + '_' + self.dofs[i]
                # find index of this muscle
                imuscle = self.muscles_selected.index(self.muscle_names[j])
                dm[imuscle, :, i] = np.interp(t, dm_dat.time, dm_dat[dm_name])
        lmt = np.zeros([nmuscles, N])
        vmt = np.zeros([nmuscles, N])
        for i in range(0, nmuscles):
            lmt[i, :] = np.interp(t, lmt_dat.time, lmt_dat[lmt_dat.columns[i + 1]])
            lmt_dot = np.gradient(lmt_dat[lmt_dat.columns[i + 1]], lmt_dat.time)
            vmt[i, :] = np.interp(t, lmt_dat.time, lmt_dot)

        # interpolate inverse dynamic moment
        id = np.zeros([ndof, N])
        for i in range(0, ndof):
            id[i, :] = np.interp(t, iddat.time, iddat[self.dofs[i] + '_moment'])

        # interpolate inverse kinematics
        ik = np.zeros([ndof, N])
        ik_dot = np.zeros([ndof, N])
        for i in range(0, ndof):
            ik[i,:] = np.interp(t, ikdat.time, ikdat[self.dofs[i]])
            ik_dot_sel = np.gradient(ikdat[self.dofs[i]], ikdat.time)
            ik_dot[i,:] = np.interp(t, ikdat.time, ik_dot_sel)

        self.lmt = lmt
        self.vmt = vmt
        self.dm = dm
        self.id = id
        self.ik = ik
        self.ikdot = ik_dot
        self.N = N
        self.t = t

    def formulate_solve_ocp(self, dt = 0.01, tstart = None, tend = None, bool_static_opt = True,
                            bool_write_solution = True, objective_function = 'min_act',
                            opt_var_info = 'default'):
        # formulate optimal control problem
        # this is the first version. In the future we want to speed-up computations bu using casadi functions
        # options for objective function are:
        # - min_act: minimize muscle activations
        # - min_pos_fiber_power: minimize positive fiber power

        # init degroote muscles (if needed)
        self.init_muscle_model(bool_overwrite = False)

        # create casadi functions
        if opt_var_info == 'default':
            muscle_dyn_func = self.casadi_func_muscle_dyn()
        elif opt_var_info == 'tendon_stiffness':
            muscle_dyn_func = self.casadi_func_muscle_dyn_opt_tendon_stiffness()

        # compue lmt and dm if needed
        if self.lmt_dat is None:
            self.compute_lmt_dm()

        # interpolate inputs
        self.interpolate_inputs(dt = dt, tstart = tstart, tend = tend)

        # static optimization to get initial guess
        if bool_static_opt:
            self.static_optimization(self.t, self.lmt, self.vmt, self.dm, self.id)

        # model info
        nmuscles = len(self.muscles_selected)

        # unpack some variables
        N = self.N
        ndof = len(self.dofs)

        # optimization variables
        opti = ca.Opti()
        e = opti.variable(nmuscles, N)
        a = opti.variable(nmuscles, N)
        lm_tilde = opti.variable(nmuscles, N) # muscle fiber length / opt length
        vm_tilde = opti.variable(nmuscles, N) # time derivative of lm_tilde
        tau_ideal_optvar = opti.variable(ndof, N) # ideal joint torque
        tau_ideal = tau_ideal_optvar # scaling factor

        #    - if minimizing positive fiber power add helper variables
        if objective_function == 'min_pos_fiber_power':
            s_pos_power = opti.variable(nmuscles, N)
            s_neg_power = opti.variable(nmuscles, N)

        #    - if optimizatin tendon stiffness
        if opt_var_info == 'tendon_stiffness':
            tendon_stiffness = opti.variable(nmuscles)
            opti.set_initial(tendon_stiffness, 35)

        # lower bounds on optimization variables
        opti.subject_to(0.01 < e[:])
        opti.subject_to(0.01 < a[:])
        opti.subject_to(0.2 < lm_tilde[:])
        opti.subject_to(-10 < vm_tilde[:])
        if objective_function == 'min_pos_fiber_power':
            opti.subject_to(s_pos_power[:] >= 0)
            opti.subject_to(s_neg_power[:] >= 0)
        if opt_var_info == 'tendon_stiffness':
            opti.subject_to(tendon_stiffness[:] > 20)

        # upper bounds on optimization variables
        opti.subject_to(e[:] < 1)
        opti.subject_to(a[:] < 1)
        opti.subject_to(lm_tilde[:] < 1.7)
        opti.subject_to(vm_tilde[:] < 10)
        if opt_var_info == 'tendon_stiffness':
            opti.subject_to(tendon_stiffness[:] < 50)

        # initial guess (in future based on static optimization solution)
        if bool_static_opt:
            opti.set_initial(e, self.sol_static_opt['a'])
            opti.set_initial(a, self.sol_static_opt['a'])
            opti.set_initial(lm_tilde, self.sol_static_opt['lm_tilde'])
            opti.set_initial(vm_tilde,self.sol_static_opt['vm_tilde'] )
        else:
            opti.set_initial(e[:], 0.1)
            opti.set_initial(a[:], 0.1)
            opti.set_initial(lm_tilde[:], 1)
            opti.set_initial(vm_tilde[:], 0)

        # activation dynamics
        tact = 0.015
        tdeact = 0.06
        #b = 0.1
        b = 0.1
        dadt_mx = ca.MX(nmuscles, N)
        for k in range(0, N):
            dadt_mx[:,k] = self.activation_dynamics_degroote2016(e[:, k], a[:, k], tact, tdeact, b)

        # trapezoidal integration
        x_mx = ca.vertcat(a, lm_tilde)
        xd_mx = ca.vertcat(dadt_mx, vm_tilde)
        int_error = self.trapezoidal_intergrator(x_mx[:, 0:-1], x_mx[:, 1:], xd_mx[:, 0:-1], xd_mx[:, 1:], dt)
        opti.subject_to(int_error == 0)

        # muscle dynamics as a constraint (with some additional outputs for post-processing)
        muscle_dyn_constr = ca.MX(nmuscles, N)
        muscle_torque = ca.MX(ndof, N)
        tendon_force= ca.MX(nmuscles, N)
        active_fiber_force = ca.MX(nmuscles, N)
        for k in range(0, N):
            if opt_var_info == 'default':
                muscle_dyn_constr[:, k], muscle_torque[:, k], tendon_force[:,k], active_fiber_force[:,k] = (
                    muscle_dyn_func(a[:, k], lm_tilde[:, k],vm_tilde[:, k], self.lmt[:, k], self.dm[:, k, :]))
            elif opt_var_info == 'tendon_stiffness':
                muscle_dyn_constr[:, k], muscle_torque[:, k], tendon_force[:, k], active_fiber_force[:, k] = (
                    muscle_dyn_func(a[:, k], lm_tilde[:, k], vm_tilde[:, k], self.lmt[:, k],
                                    self.dm[:, k, :], tendon_stiffness))

        moment_constr = (muscle_torque + tau_ideal)- self.id

        # add constraints for moment and hill equilibrium
        opti.subject_to(muscle_dyn_constr == 0)
        opti.subject_to(moment_constr == 0)

        # objective function
        if objective_function == 'min_act':
            J = (ca.sumsqr(e)/N/nmuscles +
                 ca.sumsqr(a)/N/nmuscles +
                 0.1*ca.sumsqr(tau_ideal_optvar)/N/ndof +
                 0.01 * ca.sumsqr(vm_tilde)/N/nmuscles)
        elif objective_function == 'min_pos_fiber_power':
            # get matrix with optimal fiber length
            lm_opt_mat = np.zeros([nmuscles, N])
            for m in range(nmuscles):
                lm_opt_mat[m, :] = self.degroote_muscles[m].get_optimal_fiber_length()
            # fiber velocity
            fiber_velocity = vm_tilde * lm_opt_mat
            # compute fiber power
            fiber_power = -fiber_velocity * active_fiber_force
            # constraint with helper variables
            opti.subject_to(fiber_power/100 == s_pos_power/100 - s_neg_power/100)
            J = (ca.sumsqr(s_pos_power/100)/N/nmuscles +
                 0.5 * ca.sumsqr(e) / N / nmuscles +
                 0.5 * ca.sumsqr(e) / N / nmuscles +
                 0.1 * ca.sumsqr(tau_ideal_optvar) / N / ndof +
                 0.01 * ca.sumsqr(vm_tilde) / N / nmuscles)
        elif objective_function == 'min_fiber_power_squared':
            # get matrix with optimal fiber length
            lm_opt_mat = np.zeros([nmuscles, N])
            for m in range(nmuscles):
                lm_opt_mat[m, :] = self.degroote_muscles[m].get_optimal_fiber_length()
            # fiber velocity
            fiber_velocity = vm_tilde * lm_opt_mat
            # compute fiber power
            fiber_power = -fiber_velocity * active_fiber_force
            J = (ca.sumsqr(fiber_power/10)/N/nmuscles +
                 0.1 * ca.sumsqr(e) / N / nmuscles +
                 0.1 * ca.sumsqr(e) / N / nmuscles +
                 0.1 * ca.sumsqr(tau_ideal_optvar) / N / ndof +
                 0.01 * ca.sumsqr(vm_tilde) / N / nmuscles)
        else:
            print('objective function not recognized')

        opti.minimize(J*10)

        p_opts = {"expand": True}
        #p_opts = {}
        s_opts = {"max_iter": 1000, "tol": 1e-5, "linear_solver": "mumps",
                  "nlp_scaling_method": "gradient-based"}
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        self.solution = {"t": self.t,
                         "e": sol.value(e),
                         "a": sol.value(a),
                         "lm_tilde": sol.value(lm_tilde),
                         "vm_tilde": sol.value(vm_tilde),
                         "muscle_dyn_constr": sol.value(muscle_dyn_constr),
                         "muscle_torque": sol.value(muscle_torque),
                         "tau_ideal": sol.value(tau_ideal),
                         "tendon_force": sol.value(tendon_force),
                         "moment_arm": self.dm,
                         "lmt": self.lmt,
                         "id": self.id,
                         "J": sol.value(J),
                         "dofs": self.dofs,
                         "muscles": self.muscles_selected,
                         "objective_function": objective_function}

        # post-processing
        muscle_dyn_constr_test = np.zeros_like(self.solution['e'])
        fiber_length = np.zeros_like(self.solution['e'])
        fiber_velocity = np.zeros_like(self.solution['e'])
        tendon_velocity = np.zeros_like(self.solution['e'])
        active_fiber_force = np.zeros_like(self.solution['e'])
        pennation_angle = np.zeros_like(self.solution['e'])
        tendon_length = np.zeros_like(self.solution['e'])
        passive_force = np.zeros_like(self.solution['e'])
        fiber_force_flv = np.zeros_like(self.solution['e'])
        for m in range(nmuscles):
            # set muscle state
            msel = self.degroote_muscles[m]
            # set tendon stiffness if needed
            if opt_var_info == 'tendon_stiffness':
                msel.set_tendon_stiffness(sol.value(tendon_stiffness)[m])
            # set muscle state
            msel.set_activation(self.solution['a'][m,:])
            msel.set_norm_fiber_length(self.solution['lm_tilde'][m,:])
            msel.set_norm_fiber_length_dot(self.solution['vm_tilde'][m,:])
            msel.set_muscle_tendon_length(self.lmt[m,:])
            muscle_dyn_constr_test[m,:] = msel.compute_hill_equilibrium()

            # analyze muscle info at current state
            active_fiber_force[m, :] = msel.get_active_fiber_force()
            tendon_length[m, :] = msel.get_tendon_length()
            passive_force[m, :] = msel.get_passive_force()
            fiber_length[m, :] = msel.get_fiber_length()
            fiber_velocity[m, :] = msel.get_fiber_velocity()
            pennation_angle[m, :] = msel.get_pennation_angle()
            fiber_force_flv[m, :] = msel.get_force_fiber_flv()

            # compute tendon velocity
            vmt_projected = fiber_velocity[m,:] / msel.cosalpha # important divide by cos alpha because constant width
            tendon_velocity[m, :] = self.vmt[m, :] - vmt_projected

         # other outcomes
        fiber_power = -fiber_velocity * fiber_force_flv
        tendon_power = -tendon_velocity * self.solution['tendon_force']
        muscle_power = -self.vmt * self.solution['tendon_force']
        pas_fiber_power = -fiber_velocity * passive_force

        self.solution['active_fiber_force'] = active_fiber_force
        self.solution['tendon_length'] = tendon_length
        self.solution['passive_force'] = passive_force
        self.solution['fiber_length'] = fiber_length
        self.solution['fiber_velocity'] = fiber_velocity
        self.solution['pennation_angle'] = pennation_angle
        self.solution['tendon_velocity'] = tendon_velocity
        self.solution['fiber_power'] = fiber_power
        self.solution['tendon_power'] = tendon_power
        self.solution['muscle_power'] = muscle_power
        self.solution['passive_fiber_power'] = pas_fiber_power
        self.solution['fiber_force_flv'] = fiber_force_flv

        # power ideal actuator
        self.solution["power_actuator"] = self.solution["tau_ideal"]*self.ikdot
        self.solution["joint_power"] = self.solution['id'] * self.ikdot
        self.solution["ik"] = self.ik
        self.solution["ikdot"] = self.ikdot

        if objective_function == 'min_pos_fiber_power':
            self.solution['s_pos_power'] = sol.value(s_pos_power)
            self.solution['s_neg_power'] = sol.value(s_neg_power)
        if opt_var_info == 'tendon_stiffness':
            self.solution['tendon_stiffness'] = sol.value(tendon_stiffness)

        if bool_write_solution:
            self.write_results()


        return self.solution

    def trapezoidal_intergrator(self,x, x1, xd, xd1, dt):
        error = (x1 - x) - (0.5 * dt * (xd + xd1))
        return error

    def activation_dynamics_degroote2016(self, e, a, tact, tdeact, b):
        d1 = 1. / (tact * (0.5 + 1.5 * a))
        d2 = (0.5 + 1.5 * a)/ tdeact
        f = 0.5 * np.tanh(b * (e - a))
        dadt = (d1* (f + 0.5) + d2 * (-f + 0.5))* (e - a)
        return(dadt)

    def get_muscle_properties(self):
        # I want to change this in the future. Probably best to keep all muscle properties
        # in the DeGroote muscle objects. This makes it easier to adapt muscle properties
        # using set functions.
        # get muscle properties
        if self.muscles_selected is None :
            self.identify_muscles()
        nmuscles = len(self.muscles_selected)
        lm_opt = np.zeros([nmuscles])
        fisom_opt = np.zeros([nmuscles])
        tendon_slack = np.zeros([nmuscles])
        alpha_opt = np.zeros([nmuscles])
        for m in range(nmuscles):
            muscle = self.model.getMuscles().get(self.muscles_selected[m])
            lm_opt[m] = muscle.getOptimalFiberLength()
            fisom_opt[m] = muscle.getMaxIsometricForce()
            tendon_slack[m] = muscle.getTendonSlackLength()
            alpha_opt[m] = muscle.getPennationAngleAtOptimalFiberLength()

        return(lm_opt, fisom_opt, tendon_slack, alpha_opt)

    def static_optimization(self, t, lmt, vmt, dm, id):
        # compute muscle fiber lengths assuming rigid tendon
        N = lmt.shape[1]
        ndof = len(self.dofs)
        nmuscles = len(self.muscles_selected)

        # optimization variables
        opti = ca.Opti()
        a = opti.variable(nmuscles, N)
        topt = opti.variable(ndof, N)

        # bounds
        opti.subject_to(a[:]>0)
        opti.subject_to(a[:]<1)

        # muscle forces
        tau_joint_muscles = ca.MX.zeros(ndof,N)
        lm_tilde_mat = ca.MX.zeros(nmuscles, N)
        vm_tilde_mat = ca.MX.zeros(nmuscles, N)
        for m in range(len(self.muscles_selected)):
            msel = self.degroote_muscles[m]
            lmt_sel = lmt[m,:]
            vmt_sel = vmt[m,:]
            a_sel = a[m,:].T
            Fmuscle = msel.compute_muscle_force_rigid_tendon(lmt[m,:], vmt[m,:], a[m,:].T)
            lm_tilde_mat[m, :] = msel.get_norm_fiber_length()
            vm_tilde_mat[m, :] = msel.get_norm_fiber_velocity()
            for dof in range(ndof):
                dm_sel = dm[m, :, dof]
                tau_joint_muscles[dof, :] = tau_joint_muscles[dof, :] + (Fmuscle * dm_sel).T

        # constraint ID torque equals sum of muscle torques
        opti.subject_to(tau_joint_muscles + topt == id)

        # objective function
        opti.minimize(ca.sumsqr(topt)/ndof/N + ca.sumsqr(a)/N/nmuscles)

        # solve optimization problem
        p_opts = {}
        s_opts = {"max_iter": 1000, "tol": 1e-5, "linear_solver": "mumps",
                    "nlp_scaling_method": "gradient-based"}
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        # unpack solution
        self.sol_static_opt = {"a": sol.value(a),
                               "topt": sol.value(topt),
                               "lm_tilde": sol.value(lm_tilde_mat),
                               "t": t,
                               "vm_tilde": sol.value(vm_tilde_mat)}
        return(self.sol_static_opt)

    # function to call static optimization from outside
    def run_static_optimization(self, dt = 0.01, tstart = None, tend = None):
        # interpolate inputs
        self.interpolate_inputs(dt=dt, tstart=tstart, tend=tend)

        # init degroote muscles
        self.init_muscle_model(bool_overwrite=False)

        # create casadi functions
        muscle_dyn_func = self.casadi_func_muscle_dyn()

        # static optimization
        self.static_optimization(self.t, self.lmt, self.vmt, self.dm, self.id)
        return self.sol_static_opt

    #------------------------
    # create casadi functions
    #------------------------

    def casadi_func_muscle_dyn(self):

        nmuscles = len(self.muscles_selected)
        ndof = len(self.dofs)

        # create symbolic variables for inputs
        a = ca.MX.sym('a', nmuscles)
        lm_tilde = ca.MX.sym('lm_tilde', nmuscles)
        vm_tilde = ca.MX.sym('vm_tilde', nmuscles)
        lmt = ca.MX.sym('lmt', nmuscles)
        dm = ca.MX.sym('dm', nmuscles,ndof)

        # pre-allocate outputs
        muscle_dyn_constr = ca.MX.zeros(nmuscles)
        Ftendon = ca.MX.zeros(nmuscles)
        tau_joint_muscles = ca.MX.zeros(ndof)
        active_fiber_force = ca.MX.zeros(nmuscles)

        # loop over muscles
        for m in range(nmuscles):
            # set muscle state
            msel = self.degroote_muscles[m]
            msel.set_activation(a[m])
            msel.set_norm_fiber_length(lm_tilde[m])
            msel.set_norm_fiber_length_dot(vm_tilde[m])
            msel.set_muscle_tendon_length(lmt[m])
            muscle_dyn_constr[m] = msel.compute_hill_equilibrium()
            # compute joint torques
            Fmuscle = msel.get_tendon_force()
            active_fiber_force[m] = msel.get_active_fiber_force()
            for dof in range(ndof):
                tau_joint_muscles[dof] = tau_joint_muscles[dof] + Fmuscle * dm[m, dof]
            Ftendon[m] = Fmuscle

        # create casadi function
        muscle_dyn_func = ca.Function('muscle_dyn_func', [a, lm_tilde, vm_tilde, lmt, dm],
                                      [muscle_dyn_constr, tau_joint_muscles, Ftendon, active_fiber_force],
                                      ['a', 'lm_tilde','vm_tilde', 'lmt', 'dm'],
                                      ['muscle_dyn_constr', 'joint_torque_muscles', 'Ftendon',
                                       'active_fiber_force'])


        return muscle_dyn_func

    def casadi_func_muscle_dyn_opt_tendon_stiffness(self):
        nmuscles = len(self.muscles_selected)
        ndof = len(self.dofs)

        # create symbolic variables for inputs
        a = ca.MX.sym('a', nmuscles)
        lm_tilde = ca.MX.sym('lm_tilde', nmuscles)
        vm_tilde = ca.MX.sym('vm_tilde', nmuscles)
        lmt = ca.MX.sym('lmt', nmuscles)
        dm = ca.MX.sym('dm', nmuscles,ndof)
        tendon_stiffness = ca.MX.sym('tendon_stiffness', nmuscles)

        # pre-allocate outputs
        muscle_dyn_constr = ca.MX.zeros(nmuscles)
        Ftendon = ca.MX.zeros(nmuscles)
        tau_joint_muscles = ca.MX.zeros(ndof)
        active_fiber_force = ca.MX.zeros(nmuscles)

        # loop over muscles
        for m in range(nmuscles):
            # get muscle object
            msel = self.degroote_muscles[m]
            # adapt tendon stiffness
            msel.set_tendon_stiffness(tendon_stiffness[m])
            # set muscle state
            msel.set_activation(a[m])
            msel.set_norm_fiber_length(lm_tilde[m])
            msel.set_norm_fiber_length_dot(vm_tilde[m])
            msel.set_muscle_tendon_length(lmt[m])
            muscle_dyn_constr[m] = msel.compute_hill_equilibrium()
            # compute joint torques
            Fmuscle = msel.get_tendon_force()
            active_fiber_force[m] = msel.get_active_fiber_force()
            for dof in range(ndof):
                tau_joint_muscles[dof] = tau_joint_muscles[dof] + Fmuscle * dm[m, dof]
            Ftendon[m] = Fmuscle

        # create casadi function
        muscle_dyn_func = ca.Function('muscle_dyn_func', [a, lm_tilde, vm_tilde, lmt, dm, tendon_stiffness],
                                      [muscle_dyn_constr, tau_joint_muscles, Ftendon, active_fiber_force],
                                      ['a', 'lm_tilde','vm_tilde', 'lmt', 'dm', 'kT'],
                                      ['muscle_dyn_constr', 'joint_torque_muscles', 'Ftendon',
                                       'active_fiber_force'])


        return muscle_dyn_func

    #------------------------
    # Debug functions
    #------------------------

    def debug_lmt(self):
        # we want to check here if lm_tilde is reasonable given the lmt and dm values
        # we can do this by plotting the muscle-tendon length and moment arms for a muscle
        self.get_muscle_properties()
        nmuscles = len(self.muscles_selected)
        lmt_dat = self.lmt_dat[0]
        lmt = np.zeros([nmuscles, len(lmt_dat.time)])
        for i in range(0, nmuscles):
            lmt[i, :] = lmt_dat[lmt_dat.columns[i+1]]


        plt.figure()
        ctm = -1
        for m in self.muscles_selected:
            ctm = ctm + 1
            dl = (lmt[ctm,:] - self.tendon_slack[ctm])/self.lm_opt[ctm]
            plt.plot(lmt_dat.time,dl, label = m)
        plt.xlabel('time [s]')
        plt.ylabel('lm_tilde rigid tendon')
        plt.legend()
        #plt.show()
        print('test')

    #-------------------------
    # get and set functions
    #-------------------------

    def set_dofs(self, dofs):
        # set dofs for analysis
        self.dofs =  dofs

    def set_muscles(self, muscles_selected):
        # manually select muscles
        self.muscles_selected = muscles_selected
        # update muscle objects
        self.init_muscle_model(bool_overwrite=True)

    def set_tendon_stiffness(self, muscle_name, kT):
        # first find this muscle
        isel = self.muscles_selected.index(muscle_name)
        msel = self.degroote_muscles[isel]
        msel.set_tendon_stiffness(kT)

    def set_maximal_isometric_force(self, muscle_name, Fmax):
        # first find this muscle
        isel = self.muscles_selected.index(muscle_name)
        msel = self.degroote_muscles[isel]
        msel.set_max_isometric_force(Fmax)

    # set lmt data -- select only specific muscles and moment arms in this file
    def set_lmt_dat(self, lmt_datfile):
        print('test')

    # set dm data -- select only specific muscles and moment arms in this file
    def set_dm_dat(self, dm_datfile):
        print('test')

    # function to print results to a file
    def set_out_path(self, out_path):
        self.out_path = out_path
    def set_filename(self, filename):
        # check if filename ends with .pkl if not add this
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        self.filename = filename


    def write_results(self, filename = None, filepath = None):
        # Save
        # is filename and file path is none derive it from inputs
        if filepath is None:
            if self.out_path is None:
                # self.out_path is subfolder temp in current directory
                self.out_path = os.path.join(os.getcwd(), 'temp')
        else:
            self.out_path = filepath
        # check for filename
        if filename is None:
            if self.filename is None:
                self.filename = self.my_subject.filenames[0] + "_mrs.pkl"
        else:
            self.filename = filename

        # to do: check if file exists, if this is the case add subscript _1 (or _2 if this exists, etc)
        if self.out_path is not None:
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
            # check if filename exists and if needed add subscript
            filename = os.path.join(self.out_path, self.filename)
            if os.path.exists(filename):
                base, ext = os.path.splitext(filename)
                counter = 1
                new_filename = f"{base}_{counter}{ext}"
                while os.path.exists(new_filename):
                    counter += 1
                    new_filename = f"{base}_{counter}{ext}"
                filename = new_filename

            with open(filename, "wb") as f:
                pickle.dump(self.solution, f)
        else:
            print('No output path specified')

    def default_plot(self):
        if self.solution is not None:
            # muscle activations
            plt.figure()
            plt.plot(self.solution['t'], self.solution['a'].T)
            plt.xlabel('time [s]')
            plt.ylabel('activation')
            plt.title('Muscle activations')
            plt.legend(self.muscles_selected)

            # joint moments
            plt.figure()
            # loop over dofs
            if len(self.dofs) == 1:
                plt.plot(self.solution['t'], self.solution['tau_ideal'].T, label='ideal actuator')
                plt.plot(self.solution['t'], self.solution['muscle_torque'].T, label='muscle torque')
                plt.plot(self.solution['t'], self.solution['id'].T, label='inverse dynamics')
            else:
                for dof in range(len(self.dofs)):
                    plt.subplot(len(self.dofs), 1, dof+1)
                    plt.plot(self.solution['t'], self.solution['tau_ideal'][dof,:].T, label='ideal actuator')
                    plt.plot(self.solution['t'], self.solution['muscle_torque'][dof,:].T, label='muscle torque')
                    plt.plot(self.solution['t'], self.solution['id'][dof,:].T, label='inverse dynamics')
                    plt.xlabel('time [s]')
                    plt.ylabel('torque [Nm]')
                    plt.title(self.dofs[dof] + ' torque')
            plt.legend()

            # muscle forces
            plt.figure()
            plt.plot(self.solution['t'], self.solution['tendon_force'].T)
            plt.xlabel('time [s]')
            plt.ylabel('force [ N]')
            plt.title('tendon force')
            plt.legend(self.muscles_selected)

    def plot_static_opt_results(self):
        # pot activation and torques of static optimization results
        if self.sol_static_opt is not None:
            plt.figure()
            plt.plot(self.sol_static_opt['t'], self.sol_static_opt['a'].T)
            plt.xlabel('time [s]')
            plt.ylabel('activation')
            plt.title('Muscle activations')
            plt.legend(self.muscles_selected)

            plt.figure()
            # loop over dofs
            if len(self.dofs) == 1:
                plt.plot(self.sol_static_opt['t'], self.sol_static_opt['topt'].T, label='ideal actuator')
                plt.plot(self.solution['t'], self.solution['id'].T, label='inverse dynamics')
            else:
                for dof in range(len(self.dofs)):
                    plt.subplot(len(self.dofs), 1, dof + 1)
                    plt.plot(self.sol_static_opt['t'], self.sol_static_opt['topt'][dof, :].T, label='ideal actuator')
                    plt.plot(self.solution['t'], self.solution['id'][dof, :].T, label='inverse dynamics')
                    plt.xlabel('time [s]')
                    plt.ylabel('torque [Nm]')
                    plt.title(self.dofs[dof] + ' torque')
            plt.legend()



# child of the MRS solver that ignores the force-length-velocity properties of the muscle fibers
class ideal_muscles_actuated(muscle_redundancy_solver):
    # init function is the same as in the redundancy solver
    def __init__(self, modelfile, ikfile, idfile, dofs,
                 muscles_selected = None, outpath = None):
        super().__init__(modelfile, ikfile, idfile, dofs, muscles_selected, outpath)
        self.version = 'ignore_FLV'

    # we need to overwrite the muscle dynamics function
    def formulate_solve_ocp(self, dt = 0.01, tstart = None, tend = None, bool_static_opt = True,
                            objective_function = 'min_act', bool_write_solution = True,
                            opt_var_info = 'default'):
        # formulate optimal control problem
        # this is the first version. In the future we want to speed-up computations bu using casadi functions
        # options for objective function are:
        # - min_act: minimize muscle activations
        # - min_pos_fiber_power: minimize positive fiber power
        # - min_fiber_power_sq: minimize squared fiber power

        # init degroote muscles
        self.init_muscle_model(bool_overwrite=False)

        # create casadi functions [dit moeten we ook aanpassen]
        if opt_var_info == 'default':
            muscle_dyn_func = self.casadi_func_muscle_dyn()
        elif opt_var_info == 'tendon_stiffness':
            muscle_dyn_func = self.casadi_func_muscle_dyn_opt_tendon_stiffness()

        # get lmt dat if needed
        if self.lmt_dat is None:
            self.compute_lmt_dm()

        # interpolate inputs
        self.interpolate_inputs(dt=dt, tstart=tstart, tend=tend)

        # model info
        nmuscles = len(self.muscles_selected)

        # unpack some variables
        N = self.N
        ndof = len(self.dofs)

        # optimization variables
        opti = ca.Opti()
        a = opti.variable(nmuscles, N)
        tau_ideal_optvar = opti.variable(ndof, N)  # ideal joint torque
        tau_ideal = tau_ideal_optvar  # scaling factor
        Ltendon_dot = opti.variable(nmuscles, N)

        if objective_function == 'min_pos_fiber_power':
            s_pos_power = opti.variable(nmuscles, N) * 100
            s_neg_power = opti.variable(nmuscles, N) * 100
            opti.subject_to(s_pos_power[:] >= 0)
            opti.subject_to(s_neg_power[:] >= 0)

        if opt_var_info == 'tendon_stiffness':
            tendon_stiffness = opti.variable(nmuscles)


        # lower bounds on optimization variables
        opti.subject_to(0.01 < a[:])
        if opt_var_info == 'tendon_stiffness':
            opti.subject_to(tendon_stiffness[:] > 20)


        # upper bounds on optimization variables
        opti.subject_to(a[:] < 1)
        if opt_var_info == 'tendon_stiffness':
            opti.subject_to(tendon_stiffness[:] < 50)

        # initial guess (in future based on static optimization solution)
        opti.set_initial(a[:], 0.1)
        if opt_var_info == 'tendon_stiffness':
            opti.set_initial(tendon_stiffness, 35)

        # there are no dynamics [we only need to solve for muscle fiber force at each time step]
        # I will ignore pennation angle in this simulation
        # muscle dynamics as a constraint (with some additional outputs for post-processing)
        muscle_torque = ca.MX(ndof, N)
        Ftendon = ca.MX(nmuscles, N)
        Ltendon = ca.MX(nmuscles, N)
        for k in range(0, N):
            if opt_var_info == 'default':
                muscle_torque[:, k], Ftendon[:, k], Ltendon[:, k] = (
                    muscle_dyn_func(a[:, k], self.lmt[:, k], self.dm[:, k, :]))
            elif opt_var_info == 'tendon_stiffness':
                muscle_torque[:, k], Ftendon[:, k], Ltendon[:, k] = (
                    muscle_dyn_func(a[:, k], self.lmt[:, k], self.dm[:, k, :], tendon_stiffness))

        moment_constr = (muscle_torque + tau_ideal)- self.id
        opti.subject_to(moment_constr == 0)

        # compute tendon velocity with same numerical method as in direct collocation problem
        x_mx = Ltendon
        xd_mx = Ltendon_dot
        int_error = self.trapezoidal_intergrator(x_mx[:, 0:-1], x_mx[:, 1:], xd_mx[:, 0:-1], xd_mx[:, 1:], dt)
        opti.subject_to(int_error == 0)

        # objective function
        if objective_function == 'min_act':
            J = (10*ca.sumsqr(a) / N / nmuscles +
                 0.1 * ca.sumsqr(tau_ideal_optvar) / N / ndof +
                 0.001 * ca.sumsqr(Ltendon_dot) / N / nmuscles)
        elif objective_function == 'min_pos_fiber_power':
            # compute power of the actuator
            velocity_actuator = self.vmt - Ltendon_dot
            power_actuator = -velocity_actuator * Ftendon
            # constraint with helper variables
            opti.subject_to(power_actuator/100 == (s_pos_power/100 - s_neg_power/100))
            # objective function
            J = (ca.sumsqr(s_pos_power/100) / N / nmuscles +
                 1 * ca.sumsqr(a) / N / nmuscles +
                 0.1 * ca.sumsqr(tau_ideal_optvar) / N / ndof +
                 0.001 * ca.sumsqr(Ltendon_dot) / N / nmuscles)
        elif objective_function == 'min_fiber_power_squared':
            # compute power of the actuator
            velocity_actuator = self.vmt - Ltendon_dot
            power_actuator = -velocity_actuator * Ftendon
            J = (ca.sumsqr(power_actuator/10) / N / nmuscles +
                 0.1 * ca.sumsqr(a) / N / nmuscles +
                 0.1 * ca.sumsqr(tau_ideal_optvar) / N / ndof +
                 0.001 * ca.sumsqr(Ltendon_dot) / N / nmuscles)
        else:
            print('objective function not recognized')
        opti.minimize(J)

        p_opts = {"expand": True}
        # p_opts = []
        s_opts = {"max_iter": 1000, "tol": 1e-5, "linear_solver": "mumps",
                  "nlp_scaling_method": "gradient-based"}
        # if len(a[:]) > 500*50:
        #     s_opts["hessian_approximation"] = "limited-memory"

        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        self.solution = {"t": self.t,
                         "a": sol.value(a),
                         "moment_arm": self.dm,
                         "lmt": self.lmt,
                         "id": self.id,
                         "tendon_force": sol.value(Ftendon),
                         "tendon_length": sol.value(Ltendon),
                         "tendon_velocity": sol.value(Ltendon_dot),
                         "tau_ideal": sol.value(tau_ideal),
                         "muscle_torque": sol.value(muscle_torque),
                         "J": sol.value(J),
                         "dofs": self.dofs,
                         "muscles": self.muscles_selected,
                         "vmt": self.vmt,
                         "objective_function": objective_function}
        # post processing
        fiber_velocity = self.vmt - self.solution['tendon_velocity']
        fiber_power = -fiber_velocity * self.solution['tendon_force']
        tendon_power = -self.solution['tendon_velocity'] * self.solution['tendon_force']
        muscle_power = -self.vmt * self.solution['tendon_force']
        self.solution['fiber_velocity'] = fiber_velocity
        self.solution['fiber_power'] = fiber_power
        self.solution['tendon_power'] = tendon_power
        self.solution['muscle_power'] = muscle_power
        self.solution["power_actuator"] = self.solution["tau_ideal"] * self.ikdot
        self.solution["joint_power"] = self.solution['id'] * self.ikdot
        self.solution["ik"] = self.ik
        self.solution["ikdot"] = self.ikdot

        if objective_function == 'min_pos_fiber_power':
            self.solution['s_pos_power'] = sol.value(s_pos_power)
            self.solution['s_neg_power'] = sol.value(s_neg_power)

        if opt_var_info == 'tendon_stiffness':
            self.solution['tendon_stiffness'] = sol.value(tendon_stiffness)

        if bool_write_solution:
            self.write_results()

        # return solution
        return self.solution

    def casadi_func_muscle_dyn(self):

        nmuscles = len(self.muscles_selected)
        ndof = len(self.dofs)

        # create symbolic variables for inputs
        a = ca.MX.sym('a', nmuscles)
        lmt = ca.MX.sym('lmt', nmuscles)
        dm = ca.MX.sym('dm', nmuscles, ndof)

        # pre-allocate outputs
        Ftendon = ca.MX.zeros(nmuscles)
        tau_joint_muscles = ca.MX.zeros(ndof)
        Ltendon = ca.MX.zeros(nmuscles)
        # loop over muscles
        for m in range(nmuscles):
            # set muscle state
            msel = self.degroote_muscles[m]
            # compute force ideal actuator
            Fmuscle = a[m] * msel.maximal_isometric_force
            for dof in range(ndof):
                tau_joint_muscles[dof] = tau_joint_muscles[dof] + Fmuscle * dm[m, dof]
            Ftendon[m] = Fmuscle
            # compute tendon length at this velocity
            Ftendon_norm = Fmuscle / msel.maximal_isometric_force
            Ltendon_norm = msel.inverse_force_length_tendon(Ftendon_norm, msel.kT, msel.tendon_shift)
            Ltendon[m] = Ltendon_norm * msel.tendon_slack_length

        # create casadi function
        muscle_dyn_func = ca.Function('muscle_dyn_func', [a, lmt, dm],
                                      [tau_joint_muscles, Ftendon, Ltendon],
                                      ['a', 'lmt', 'dm'],
                                      ['joint_torque_muscles', 'Ftendon', 'Ltendon'])
        return(muscle_dyn_func)

    def casadi_func_muscle_dyn_opt_tendon_stiffness(self):

        nmuscles = len(self.muscles_selected)
        ndof = len(self.dofs)

        # create symbolic variables for inputs
        a = ca.MX.sym('a', nmuscles)
        lmt = ca.MX.sym('lmt', nmuscles)
        dm = ca.MX.sym('dm', nmuscles, ndof)
        tendon_stiffness = ca.MX.sym('tendon_stiffness', nmuscles)

        # pre-allocate outputs
        Ftendon = ca.MX.zeros(nmuscles)
        tau_joint_muscles = ca.MX.zeros(ndof)
        Ltendon = ca.MX.zeros(nmuscles)
        # loop over muscles
        for m in range(nmuscles):
            # get muscle object
            msel = self.degroote_muscles[m]
            # set tendon stiffness
            msel.set_tendon_stiffness(tendon_stiffness[m])
            # compute force ideal actuator
            Fmuscle = a[m] * msel.maximal_isometric_force
            for dof in range(ndof):
                tau_joint_muscles[dof] = tau_joint_muscles[dof] + Fmuscle * dm[m, dof]
            Ftendon[m] = Fmuscle
            # compute tendon length at this velocity
            Ftendon_norm = Fmuscle / msel.maximal_isometric_force
            Ltendon_norm = msel.inverse_force_length_tendon(Ftendon_norm, msel.kT, msel.tendon_shift)
            Ltendon[m] = Ltendon_norm * msel.tendon_slack_length

        # create casadi function
        muscle_dyn_func = ca.Function('muscle_dyn_func', [a, lmt, dm, tendon_stiffness],
                                      [tau_joint_muscles, Ftendon, Ltendon],
                                      ['a', 'lmt', 'dm', 'kT'],
                                      ['joint_torque_muscles', 'Ftendon', 'Ltendon'])
        return(muscle_dyn_func)



# default function to run an extensive analysis of the mrs simulation results
def analyse_mrs_results(sim_results, tstart = None, tend = None):
    # check if sim_results is path to a pickle datafile. If this is the case load the datafile
    if isinstance(sim_results, str):
        if os.path.exists(sim_results):
            with open(sim_results, "rb") as f:
                sim_results = pickle.load(f)

    # start and end time (make sure that it is within sim time)
    if tstart is None:
        tstart = sim_results['t'][0]
    else:
        if tstart < sim_results['t'][0]:
            tstart = sim_results['t'][0]
    if tend is None:
        tend = sim_results['t'][-1]
    else:
        if tend > sim_results['t'][-1]:
            tend = sim_results['t'][-1]

    # selected indices in time window
    time = sim_results['t']
    inds = np.where((time >= tstart) & (time <= tend))[0]

    # compute work done by muscle fibers in this time window
    [fiberwork, fiberwork_pos, fiberwork_neg] =\
        get_pos_neg_netto_work(sim_results['t'], sim_results['fiber_power'], tstart = tstart, tend = tend)

    # compute mechanical work done by tendon
    [tendonwork, tendonwork_pos, tendonwork_neg] =\
        get_pos_neg_netto_work(sim_results['t'], sim_results['tendon_power'], tstart = tstart, tend = tend)

    # compute mechanical work done by muscle
    [musclework, musclework_pos, musclework_neg] =\
        get_pos_neg_netto_work(sim_results['t'], sim_results['muscle_power'], tstart = tstart, tend = tend)

    # compute activation squared in this time window
    asq = sim_results['a']**2
    asq_mean = np.mean(asq[:, inds], axis = 1)




    # store outputs
    results = {"fiberwork": fiberwork,
               "fiberwork_pos": fiberwork_pos,
               "fiberwork_neg": fiberwork_neg,
               "tendonwork": tendonwork,
               "tendonwork_pos": tendonwork_pos,
               "tendonwork_neg": tendonwork_neg,
               "musclework": musclework,
               "musclework_pos": musclework_pos,
               "musclework_neg": musclework_neg,
               "asq_mean": asq_mean}



    return(results)

def get_pos_neg_netto_work(time, power, tstart = None, tend = None):

    # start and end time (make sure that it is within sim time)
    if tstart is None:
        tstart = time [0]
    else:
        if tstart < time[0]:
            tstart = time [0]
    if tend is None:
        tend = time[-1]
    else:
        if tend > time[-1]:
            tend = time[-1]

    # net work
    inds = np.where((time >= tstart) & (time <= tend))[0]
    work  = np.trapz(power[:,inds], time[inds], axis = 1)

    # positive work
    power_pos = power.copy()
    power_pos[power_pos<0] = 0
    work_pos = np.trapz(power_pos[:, inds], time[inds], axis=1)

    # negative work
    power_neg = power.copy()
    power_neg[power_neg > 0] = 0
    work_neg = np.trapz(power_neg[:, inds], time[inds], axis=1)

    # return work
    return(work, work_pos, work_neg)




































