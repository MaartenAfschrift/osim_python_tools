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
        self.vmax = np.zeros([len(self.muscles_selected)])*10
        self.Atendon = np.zeros([len(self.muscles_selected)])*35
        self.get_muscle_properties()
        return(self.muscles_selected)

    def compute_lmt_dm(self):
        # use class to do this
        # ToDo: add utility to compute lmt and dM for subset of muscles
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
        nmuscles = len(self.muscles_selected)
        self.degroote_muscles = []
        for m in range(nmuscles):
            self.degroote_muscles.append(DeGrooteMuscle(self.fisom_opt[m], self.lm_opt[m],
                                                        self.tendon_slack[m], self.alpha_opt[m],
                                                        self.vmax[m], self.Atendon[m]))


    def formulate_solve_ocp(self, dt = 0.01, t0 = None, tend = None):
        # formulate optimal control problem
        # this is the first version. In the future we want to speed-up computations bu using casadi functions

        # init degroote muscles
        self.init_muscle_model()

        # first version for a single trial
        itrial = []
        iddat = self.iddat[itrial]
        lmt_dat = self.lmt_dat[itrial]
        dm_dat = self.dm_dat[itrial]


        # Create a discrete time axis
        if t0 is None:
            t0 = iddat.time.iloc[0]
        if tend is None:
            tend = iddat.time.iloc[-1]
        t = np.arange(t0, tend, dt)
        N = len(t)

        # ToDo: stopped here on 15/02/2025
        # interpolate moment arms and muscle-tendon lengths to the time axis (t)
        # and identify dof spanned and muscle name for dm data
        # selecting the correct dof and muscle for each column is not so easy to solve. This should probably
        # be done when computing the muscle moment arm data
        # solved: this is already done in:self.my_subject.dofs_dm
        # note: first loop is over selected dofs and inner loop is over muscles

        # we probably want a list with dms for each dof (with zeros for muscles that do not span the dof)
        dm = np.zeros([len(dm_dat.columns)-1, N])
        for i in range(1, len(dm_dat.columns)):
            colheader = dm_dat.columns[i]
            dm[i, :] = np.interp(t, dm_dat.time, dm_dat[dm_dat.columns[i]])
        lmt = np.zeros([len(lmt_dat.columns)-1, N])
        for i in range(1, len(lmt_dat.columns)):
            lmt[i, :] = np.interp(t, lmt_dat.time, lmt_dat[lmt_dat.columns[i]])



        # model info
        nmuscles = len(self.muscles_selected)

        # optimization variables
        opti = ca.Opti()
        e = opti.variable(nmuscles, N)
        a = opti.variable(nmuscles, N)
        lm_tilde = opti.variable(nmuscles, N) # muscle fiber length / opt length
        vm_tilde = opti.variable(nmuscles, N) # time derivative of lm_tilde

        # bounds on optimization variables
        opti.subject_to(0 <= e <= 1)
        opti.subject_to(0 <= a <= 1)
        opti.subject_to(0.2 <= lm_tilde <= 1.8)
        opti.subject_to(-10 <= vm_tilde <= 10)

        # initial guess

        # activation dynamics
        tact = 0.015
        tdeact = 0.06
        b = 0.1
        dadt_mx = ca.MX(nmuscles, N)
        for k in range(1, N):
            dadt_mx[:,k] = self.activation_dynamics_degroote2016(e[:, k], a[:, k], tact, tdeact, b)

        # trapezoidal integration
        x_mx = ca.vertcat(a, lm_tilde)
        xd_mx = ca.vertcat(dadt_mx, vm_tilde)
        int_error = self.trapezoidal_intergrator(x_mx[:, 0:-1], x_mx[:, 1:], xd_mx[:, 0:-1], xd_mx[:, 1:], dt)
        opti.subject_to(int_error == 0)

        # muscle dynamics as a constraint
        muscle_dyn_constr = ca.MX(nmuscles, N)
        for k in range(1, N):
            for m in range(nmuscles):
                # set muscle state
                msel = self.degroote_muscles[m]
                msel.set_activation(a[m, k])
                msel.set_norm_fiber_length(lm_tilde[m, k])
                msel.set_norm_fiber_velocity(vm_tilde[m, k])
                msel.set_muscle_tendon_length(lmt[m, k])
                muscle_dyn_constr[m, k] = msel.compute_hill_equilibrium(lm_tilde[m, k], vm_tilde[m, k], a[m, k])



    def trapezoidal_intergrator(x, x1, xd, xd1, dt):
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



























