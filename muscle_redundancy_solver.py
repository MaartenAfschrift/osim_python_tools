# Tools for muscle redundancy solver in python
#---------------------------------------------

from osim_utilities import osim_subject
import casadi as ca
import opensim as osim
from osim_utilities import readMotionFile
from general_utilities import lowPassFilterDataFrame
import numpy as np
from pathlib import Path
from degroote2016_muscle_model import DeGrooteMuscle

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
    def __init__(self, modelfile, ikfile, idfile):

        # inputs
        self.modelfile = modelfile
        self.ikfile = ikfile
        self.idfile = idfile

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
        self.muscles_selected = None

        # list of variables that we need here
        # self.muscles (default all ?) # set by user or identify from dofs
        # self.dofs (default all muscle actuated) # set by user or identify from model
        # self.version ('default', 'ignore_FLV') # ignore_FLV = ignore force-length-velocity fiber
        # self.opt_vars ('default', 'free_kT') # free_kT = free tendon stiffness in each muscle
        # self.objective ('default', 'min_PFce') # min_PFce = minimize positive work muscle fibers


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
        return(self.muscles_selected)

    def compute_lmt_dm(self):
        # use osim_subject class to do this
        self.my_subject.compute_lmt(selected_muscles = self.muscles_selected)
        self.my_subject.compute_dM(selected_muscles = self.muscles_selected,
                                   selected_dofs = self.dofs)
        self.lmt_dat = self.my_subject.lmt_dat
        self.dm_dat = self.my_subject.dm_dat


    def filter_inputs(self, order = 2, cutoff_frequency = 6):
        ntrials = len(self.iddat)
        for itrial in range(ntrials):
            # filter moment
            self.iddat[itrial] = lowPassFilterDataFrame(self.iddat[itrial], cutoff_frequency, order)
            # filter muscle-tendon-lengths
            self.lmt_dat[itrial] = lowPassFilterDataFrame(self.lmt_dat[itrial], cutoff_frequency, order)
            # filter moment arms
            self.dm_dat[itrial] = lowPassFilterDataFrame(self.dm_dat[itrial], cutoff_frequency, order)

    def init_muscle_model(self):
        # init degroote2016 muscle model
        nmuscles = len(self.muscles_selected)
        self.degroote_muscles = []
        for m in range(nmuscles):
            self.degroote_muscles.append(DeGrooteMuscle(self.fisom_opt[m], self.lm_opt[m],
                                                        self.tendon_slack[m], self.alpha_opt[m],
                                                        10, 35))


    def formulate_solve_ocp(self, dt = 0.01, t0 = None, tend = None, bool_static_opt = True):
        # formulate optimal control problem
        # this is the first version. In the future we want to speed-up computations bu using casadi functions

        # init degroote muscles
        self.init_muscle_model()

        # create casadi functions
        muscle_dyn_func = self.casadi_func_muscle_dyn()

        # first version for a single trial
        itrial = 0
        iddat = self.iddat[itrial]
        lmt_dat = self.lmt_dat[itrial]
        dm_dat = self.dm_dat[itrial]

        # Create a discrete time axis
        if t0 is None:
            t0 = iddat.time.iloc[0]
        else:
            if t0 < iddat.time.iloc[0]:
                t0 = iddat.time.iloc[0]

        if tend is None:
            tend = iddat.time.iloc[-1]
        else:
            if tend > iddat.time.iloc[-1]:
                tend = iddat.time.iloc[-1]
        t = np.arange(t0, tend, dt)
        N = len(t)

        # interpolate moment arms and muscle-tendon lengths to the time axis (t)
        # for moment arms create a matrix with moment arms for each dofs [nmuscles, N, ndof]
        nmuscles = len(self.muscles_selected)
        ndof = len(self.dofs)
        dm = np.zeros([nmuscles, N, ndof])
        for i in range(0, ndof):
            muscle_inds = self.my_subject.dofs_dm[self.dofs[i]]
            for j in muscle_inds:
                dm_name = self.muscles_selected[j] + '_' + self.dofs[i]
                dm[j, :, i] = np.interp(t, dm_dat.time, dm_dat[dm_name])
        lmt = np.zeros([nmuscles, N])
        vmt = np.zeros([nmuscles, N])
        for i in range(0, nmuscles):
            lmt[i, :] = np.interp(t, lmt_dat.time, lmt_dat[lmt_dat.columns[i+1]])
            lmt_dot = np.gradient(lmt_dat[lmt_dat.columns[i + 1]], lmt_dat.time)
            vmt[i, :] = np.interp(t, lmt_dat.time, lmt_dot)

        # compute time derivative lmt

        # interpolate inverse dynamic moment
        id = np.zeros([ndof, N])
        for i in range(0, ndof):
            id[i, :] = np.interp(t, iddat.time, iddat[self.dofs[i] + '_moment'])

        # static optimization to get initial guess
        if bool_static_opt:
            self.static_optimization(t, lmt, vmt, dm, id)



        # model info
        nmuscles = len(self.muscles_selected)

        # optimization variables
        opti = ca.Opti()
        e = opti.variable(nmuscles, N)
        a = opti.variable(nmuscles, N)
        lm_tilde = opti.variable(nmuscles, N) # muscle fiber length / opt length
        vm_tilde = opti.variable(nmuscles, N) # time derivative of lm_tilde
        tau_ideal_optvar = opti.variable(ndof, N) # ideal joint torque
        tau_ideal = tau_ideal_optvar # scaling factor

        # lower bounds on optimization variables
        opti.subject_to(0 < e[:])
        opti.subject_to(0 < a[:])
        opti.subject_to(0.2 < lm_tilde[:])
        opti.subject_to(-10 < vm_tilde[:])

        # upper bounds on optimization variables
        opti.subject_to(e[:] < 1)
        opti.subject_to(a[:] < 1)
        opti.subject_to(lm_tilde[:] < 1.7)
        opti.subject_to(vm_tilde[:] < 10)

        # initial guess (in future based on static optimization solution)
        opti.set_initial(e[:], 0.1)
        opti.set_initial(a[:], 0.1)
        opti.set_initial(lm_tilde[:], 1)
        opti.set_initial(vm_tilde[:], 0)

        # activation dynamics
        tact = 0.015
        tdeact = 0.06
        #b = 0.1
        b = 0.01
        dadt_mx = ca.MX(nmuscles, N)
        for k in range(0, N):
            dadt_mx[:,k] = self.activation_dynamics_degroote2016(e[:, k], a[:, k], tact, tdeact, b)

        # trapezoidal integration
        x_mx = ca.vertcat(a, lm_tilde)
        xd_mx = ca.vertcat(dadt_mx, vm_tilde)
        int_error = self.trapezoidal_intergrator(x_mx[:, 0:-1], x_mx[:, 1:], xd_mx[:, 0:-1], xd_mx[:, 1:], dt)
        opti.subject_to(int_error == 0)

        # muscle dynamics as a constraint
        muscle_dyn_constr = ca.MX(nmuscles, N)
        muscle_torque = ca.MX(ndof, N)
        Ftendon = ca.MX(nmuscles, N)
        for k in range(0, N):
            muscle_dyn_constr[:, k], muscle_torque[:, k], Ftendon[:,k] = (
                muscle_dyn_func(a[:, k], lm_tilde[:, k],vm_tilde[:, k], lmt[:, k], dm[:, k, :]))
        moment_constr = (muscle_torque + tau_ideal)- id
        # add constraints
        opti.subject_to(muscle_dyn_constr == 0)
        opti.subject_to(moment_constr == 0)

        # objective function
        J = (ca.sumsqr(e)/N/nmuscles +
             ca.sumsqr(a)/N/nmuscles +
             0.1*ca.sumsqr(tau_ideal_optvar)/N/ndof +
             0.01 * ca.sumsqr(vm_tilde)/N/nmuscles)
        opti.minimize(J*10)

        p_opts = {"expand": True}
        #p_opts = {}
        s_opts = {"max_iter": 1000, "tol": 1e-5, "linear_solver": "mumps",
                  "nlp_scaling_method": "gradient-based"}
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()


        self.solution = {"t": t,
                         "e": sol.value(e),
                         "a": sol.value(a),
                         "lm_tilde": sol.value(lm_tilde),
                         "vm_tilde": sol.value(vm_tilde),
                         "muscle_dyn_constr": sol.value(muscle_dyn_constr),
                         "muscle_torque": sol.value(muscle_torque),
                         "tau_ideal": sol.value(tau_ideal),
                         "Ftendon": sol.value(Ftendon),
                         "moment_arm": dm,
                         "lmt": lmt,
                         "J": sol.value(J),
                         "id": id}
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
        self.lm_opt = lm_opt
        self.fisom_opt = fisom_opt
        self.tendon_slack = tendon_slack
        self.alpha_opt = alpha_opt

    # get and set functions
    def set_dofs(self, dofs):
        # set dofs for analysis
        self.dofs =  dofs

    def set_muscles(self, muscles_selected):
        # manually select muscles
        self.muscles_selected = muscles_selected

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
        opti.subject_to(topt[:]>-10)
        opti.subject_to(topt[:]<10)

        # muscle forces
        tau_joint_muscles = ca.MX.zeros(ndof,N)
        lm_tilde_mat = ca.MX.zeros(nmuscles, N)
        for m in range(len(self.muscles_selected)):
            msel = self.degroote_muscles[m]
            lmt_sel = lmt[m,:]
            vmt_sel = vmt[m,:]
            a_sel = a[m,:].T
            Fmuscle = msel.compute_muscle_force_rigid_tendon(lmt[m,:], vmt[m,:], a[m,:].T)
            lm_tilde_mat[m, :]= msel.get_norm_fiber_length()
            for dof in range(ndof):
                dm_sel = dm[m, :, dof]
                tau_joint_muscles[dof, :] = tau_joint_muscles[dof, :] + (Fmuscle * dm_sel).T

        # constraint ID torque equals sum of muscle torques
        opti.subject_to(tau_joint_muscles + topt == id)

        # objective function
        opti.minimize(ca.sumsqr(topt) + ca.sumsqr(a))

        # solve optimization problem
        p_opts = {}
        s_opts = {"max_iter": 1000, "tol": 1e-5, "linear_solver": "mumps",
                    "nlp_scaling_method": "gradient-based"}
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        #unpack solution
        self.sol_static_opt = {"a": sol.value(a),
                               "topt": sol.value(topt),
                               "lm_tilde": sol.value(lm_tilde_mat),
                               "t": t}









        print('todo')

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
        joint_torque_muscles = ca.MX.zeros(ndof)
        Ftendon = ca.MX.zeros(nmuscles)

        # loop over muscles
        tau_joint_muscles = ca.MX.zeros(ndof)
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
            for dof in range(ndof):
                tau_joint_muscles[dof] = tau_joint_muscles[dof] + Fmuscle * dm[m, dof]
            Ftendon[m] = Fmuscle
        # constraint ID torque equals sum of muscle torques
        for dof in range(ndof):
            joint_torque_muscles[dof] = tau_joint_muscles[dof]

        # create casadi function
        muscle_dyn_func = ca.Function('muscle_dyn_func', [a, lm_tilde, vm_tilde, lmt, dm],
                                      [muscle_dyn_constr, joint_torque_muscles, Ftendon],
                                      ['a', 'lm_tilde','vm_tilde', 'lmt', 'dm'],
                                      ['muscle_dyn_constr', 'joint_torque_muscles', 'Ftendon'])

        return muscle_dyn_func

    #------------------------
    # Debug functions
    #------------------------

    def debug_lmt(self):
        # we want to check here if lm_tilde is reasonable given the lmt and dm values
        # we can do this by plotting the muscle-tendon length and moment arms for a muscle
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






























