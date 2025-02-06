# Tools for muscle redundancy solver in python
#---------------------------------------------

from osim_utilities import osim_subject
import casadi as ca
import opensim as osim
from osim_utilities import readMotionFile
import numpy as np
from pathlib import Path

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
        return(self.muscles_selected)

    def compute_lmt_dm(self):
        # use class to do this
        # ToDo: add utility to compute lmt and dM for subset of muscles
        self.my_subject.compute_lmt(selected_muscles = None)
        self.my_subject.compute_dM()

    # get and set functions
    def set_dofs(self, dofs):
        # set dofs for analysis
        self.dofs =  dofs
        print('todo')

    def set_muscles(self, muscles):
        # manually select muscles
        self.muscles = muscles
        print('todo')

















