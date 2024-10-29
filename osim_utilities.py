
import os
from pathlib import Path
import opensim as osim
import numpy as np
import pandas as pd
from inverse_dynamics import InverseDynamics
from inverse_kinematics import InverseKinematics
from kinematic_analyses import bodykinematics
import difflib

def readMotionFile(filename):
    """ Reads OpenSim .mot and .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = np.full((nr, len(labels)), np.nan)
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        if len(d) == nc:
            data[i-1,:] = d

    file_id.close()
    dat = pd.DataFrame(data, columns = labels)

    return dat

# general class to do batch processing in opensim
# This class contains all information for a subject.

# ToDo:
# I should have a proper way to update the directories
# and the set of filenames in this directory.
# current approach is to:
#       - store the filesnames in self.filenames
#       - update directories when reading files or doing an analysis

# Would be nice to read everything as pandas Dataframe instead
# of a numpy ndarray. This dataformat is much easier to manipulate
# this is done in the readmotion file function now

class osim_subject:
    def __init__(self, modelfile, maindir = []):

        # some default settings
        if len(maindir) == 0:
            # assumes that main datadir one folder up the tree from the modelfile path
            self.maindir = Path(modelfile).parents[1]
        else:
            self.maindir = Path(maindir)

        # variables class
        self.modelpath = modelfile # path to (scaled) opensim model
        self.trcfiles = [] # path to motion capture files
        self.ikfiles = [] # path to ikfiles
        self.nfiles = [] # number of files
        self.filenames = [] # names of the files
        self.general_id_settings = [] # generic id settings
        self.extload_settings = [] # generic external load settings
        self.ext_loads_dir = self.maindir.joinpath('Loads')
        self.ik_directory = self.maindir.joinpath('IK')
        self.id_directory = self.maindir.joinpath('ID')
        self.bodykin_folder = self.maindir.joinpath('BK')
        self.moment_arm_folder = self.maindir.joinpath('moment_arm')
        self.lmt_folder = self.maindir.joinpath('lmt')
        self.extload_files = []
        self.id_dat = [] # inverse dynamics data
        self.bk_pos = [] # bodykinematics data -- position
        self.bk_vel = [] # bodykinematics data -- velocity
        self.ikdat = []
        self.ikdat = []
        self.marker_dir = None # directory with trc files
        self.folder_grfdat = None
        self.matched_grfs = None # list with matching grfdat names for the motion files

        # open model (this makes it difficult to save this with pickle, so we open and close is all the time again)
        #self.model = osim.Model(self.modelpath)
        #self.model.initSystem()

        # read some default things from the model
        [self.modelmass, self.bodymass, self.bodynames]= self.getmodelmass()
        self.get_n_muscles()
        self.get_model_coordinates()


    # read from opensim model
    #------------------------

    # get number of muscles in model
    def get_n_muscles(self):
        # Number of muscles.
        model = osim.Model(self.modelpath)
        model.initSystem()
        self.nMuscles = 0
        self.muscle_names = []
        forceSet = model.getForceSet()
        for i in range(forceSet.getSize()):
            c_force_elt = forceSet.get(i)
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1
                self.muscle_names.append(c_force_elt.getName())

    # get coordinates in the model
    def get_model_coordinates(self):
        model = osim.Model(self.modelpath)
        model.initSystem()
        coord_set = model.getCoordinateSet()
        self.coord_names = []
        self.ncoord = 0
        for i in range(0, coord_set.getSize()):
            self.coord_names.append(coord_set.get(i).getName())
            self.ncoord = self.ncoord + 1

    def getmodelmass(self):
        # read the opensim model
        model = osim.Model(self.modelpath)
        model.initSystem()
        nbodies = model.getBodySet().getSize()
        m_bodies = np.full([nbodies], np.nan)
        bodynames = []
        bodyset = model.get_BodySet()
        for i in range(0, nbodies):
            bodynames.append(bodyset.get(i).getName())
            m_bodies[i] = bodyset.get(i).getMass()
        m_tot = np.nansum(m_bodies)
        return(m_tot, m_bodies, bodynames)

    # ----------------------------
    #           read files
    #----------------------------

    def set_trcfiles_fromfolder(self, trc_folder):
        # find all .trc files in a folder
        trc_files = []
        for file in os.listdir(trc_folder):
            # Check if the file ends with .mot
            if file.endswith('.trc'):
                trc_files.append(os.path.join(trc_folder, file))
        # convert to Path objects
        if not isinstance(trc_files[0], Path):
            trc_files = [Path(i) for i in trc_files]

        # add to variable
        self.trcfiles = trc_files
        self.marker_dir = trc_folder

        # get number of files
        self.nfiles = len(self.trcfiles)

        # get filenames based on ik files
        self.filenames = []
        for itrial in self.trcfiles:
            self.filenames.append(itrial.stem)

    # read ik file as a timeseries
    def read_ik_as_timeseries(self, ikfile):
        # read ik file as a time series table
        table = osim.TimeSeriesTable(str(ikfile))
        tableProcessor = osim.TableProcessor(table)
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
        time = np.asarray(table.getIndependentColumn())

        # convert to radians
        model = osim.Model(self.modelpath)
        model.initSystem()
        table = tableProcessor.processAndConvertToRadians(model)
        columnLabels = list(table.getColumnLabels())
        return(columnLabels, table)

    # reads all ik files from a folder
    def set_ikfiles_fromfolder(self, ikfolder):
        # you can use this functionality if you already solved IK
        # (for example using opencap)

        # find all .mot trials in the IK folder
        ik_files = []
        for file in os.listdir(ikfolder):
            # Check if the file ends with .mot
            if file.endswith('.mot'):
                ik_files.append(os.path.join(ikfolder, file))
        # convert to Path objects
        if not isinstance(ik_files[0], Path):
            ik_files = [Path(i) for i in ik_files]

        # add to variable
        self.ikfiles = ik_files
        self.ik_directory = ikfolder

        # get number of files
        self.nfiles = len(self.ikfiles)

        # get filenames based on ik files
        self.filenames = []
        for itrial in self.ikfiles:
            self.filenames.append(itrial.stem)

    # determines all files connected to the subject from input argument ikfiles
    def set_ikfiles(self, ikfiles):
        # set the names of the IK files
        self.ikfiles = ikfiles

        # convert to a list if needed
        if not isinstance(ikfiles, list):
            self.ikfiles = [ikfiles]
        else:
            self.ikfiles = ikfiles

        # convert to Path objects
        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]

        # get number of files
        self.nfiles = len(self.ikfiles)

        # set some default directories based on ikfiles
        self.ik_directory  = self.ikfiles[0].parents[0]
        self.id_directory = self.ikfiles[0].parents[1].joinpath('ID')

        # get filenames based on ik files
        self.filenames = []
        for itrial in self.ikfiles:
            self.filenames.append(itrial.stem)

    def read_ikfiles(self):
        # read ik files
        self.ikdat = []
        for itrial in range(0, self.nfiles):
            ik_file = Path(os.path.join(self.ik_directory, self.filenames[itrial] + '.mot'))
            ik_data = self.read_ik(ik_file)
            self.ikdat.append(ik_data)
        # also update list with ik files
        self.set_ikfiles_fromfolder(self.ik_directory)

    def read_ik(self, ik_file):
        if ik_file.exists():
            ik_data = readMotionFile(ik_file)
        else:
            ik_data = []
            print('could find read file ', ik_file)
        return(ik_data)

    # ----- bodykinematics -------
    def read_bkfiles(self):
        # read bodykinematics files
        self.bk_pos = []
        self.bk_vel = []
        for itrial in range(0, self.nfiles):
            [bk_pos, bk_vel] = self.read_bodykin(self.bodykin_folder, self.filenames[itrial])
            self.bk_pos.append(bk_pos)
            self.bk_vel.append(bk_vel)

    def read_bodykin(self, bkfolder, trial_stem):
        # path to position and velocity file
        bk_pos_file = Path(os.path.join(bkfolder, trial_stem + '_BodyKinematics_pos_global.sto'))
        bk_vel_file = Path(os.path.join(bkfolder, trial_stem + '_BodyKinematics_vel_global.sto'))
        if bk_pos_file.exists():
            bk_pos = readMotionFile(bk_pos_file)
        else:
            bk_pos = []
        if bk_vel_file.exists():
            bk_vel = readMotionFile(bk_vel_file)
        else:
            bk_vel = []
            print('could find read file ', bk_vel_file)
        return(bk_pos, bk_vel)

    def read_idfiles(self):
        self.id_dat = []
        for itrial in range(0, self.nfiles):
                # path to position and velocity file
                id_file = Path(os.path.join(self.id_directory, self.filenames[itrial] + '.sto'))
                if id_file.exists():
                    id_dat = readMotionFile(id_file)
                else:
                    id_dat = []
                    print('could find read file ', id_file)
                self.id_dat.append(id_dat)


    # ----------------------------
    #           Computations
    #------------------------------

    def get_LMT_ifile(self, ifile, tstart = None, tend = None):

        # get time vector and number of frames.
        t = self.ikdat[ifile].time
        nfr = len(t)

        # init state of model
        model = osim.Model(self.modelpath)
        model.initSystem()
        s = model.initSystem()
        state_vars = model.getStateVariableValues(s)
        force_set = model.getMuscles()
        state_vars.setToZero()
        model.setStateVariableValues(s, state_vars)
        model.realizePosition(s)
        forceSet = model.getForceSet()

        # read ik as time_series
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[ifile])
        Qs = table.getMatrix().to_numpy()

        # get state variable names
        stateVariableNames = model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]

        # select frames in time window
        if tstart is None:
            tstart = t.iloc[0]
        if tend is None:
            tend = t.iloc[-1]
        indices = np.where((t >= tstart) & (t <= tend))[0]
        # pre allocate output
        lMT = np.zeros((len(indices), self.nMuscles))

        # loop over all frames
        cti = 0
        for i in indices:
            # set state from current frame
            for j in range(0, len(columnLabels)):
                index = stateVariableNamesStr.index(columnLabels[j])
                state_vars.set(index, Qs[i,j])
            model.setStateVariableValues(s, state_vars)
            model.realizePosition(s)
            # loop over muscles to get muscle-tendon length
            for m in range(0, self.nMuscles):
                muscle_m = forceSet.get(self.muscle_names[m])
                muscle_m = osim.Muscle.safeDownCast(muscle_m)
                lMT[cti,m] = muscle_m.getLength(s)
            # update counter
            cti = cti +1

        # return lmt as a dataframe
        data = np.concatenate(
            (np.expand_dims(t[indices], axis=1), lMT), axis=1)
        columns = ['time'] + self.muscle_names
        muscle_tendon_lengths = pd.DataFrame(data=data, columns=columns)

        return muscle_tendon_lengths

    def get_dM_ifile(self, ifile):

        # get time vector and number of frames.
        t = self.ikdat[ifile].time
        nfr = len(t)

        # pre allocate output
        dM = np.zeros((nfr, self.nMuscles * self.ncoord))

        # init state of model
        model = osim.Model(self.modelpath)
        s = model.initSystem()
        state_vars = model.getStateVariableValues(s)
        force_set = model.getMuscles()
        state_vars.setToZero()
        model.setStateVariableValues(s, state_vars)
        model.realizePosition(s)

        # read ik as time_series
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[ifile])
        Qs = table.getMatrix().to_numpy()

        # get state variable names
        stateVariableNames = model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]

        # loop over all frames
        self.dM_names = []
        for i in range(0, nfr):
            # set state from current frame
            for j in range(0, len(columnLabels)):
                index = stateVariableNamesStr.index(columnLabels[j])
                state_vars.set(index, Qs[i, j])
            model.setStateVariableValues(s, state_vars)
            model.realizePosition(s)
            # loop over muscles to get muscle-tendon length
            for m in range(0, self.nMuscles):
                muscle_m = self.forceSet.get(self.muscle_names[m])
                muscle_m = osim.Muscle.safeDownCast(muscle_m)
                for j in range(0, self.ncoord):
                    # this step is computationally very expensive
                    dM[i, m + j * self.nMuscles] = muscle_m.computeMomentArm(s, model.getCoordinateSet().get(j))
                    if i == 0:
                        self.dM_names.append(muscle_m.getName() + '_' + model.getCoordinateSet().get(j).getName())
            # print current stage per 500 processed frames
            if np.remainder(i, 100) == 0:
                print('computing moment arms: frame ' , i , '/' ,  nfr)

        # return lmt as a dataframe
        data = np.concatenate(
            (np.expand_dims(t, axis=1), dM), axis=1)
        columns = ['time'] + self.dM_names
        moment_arms = pd.DataFrame(data=data, columns=columns)

        return moment_arms

    def get_dM_ifile_fast(self, ifile, tstart = None, tend = None):

        # get time vector and number of frames.
        t = self.ikdat[ifile].time
        nfr = len(t)

        # init state of model
        model = osim.Model(self.modelpath)
        s = model.initSystem()
        state_vars = model.getStateVariableValues(s)
        force_set = model.getMuscles()
        state_vars.setToZero()
        model.setStateVariableValues(s, state_vars)
        model.realizePosition(s)
        forceset = model.getForceSet()

        # read ik as time_series
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[ifile])
        Qs = table.getMatrix().to_numpy()

        # get state variable names
        stateVariableNames = model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]

        # get all headers
        self.dM_names = []
        for j in range(0, self.ncoord):
            for m in range(0, self.nMuscles):
                muscle_m = forceset.get(self.muscle_names[m])
                self.dM_names.append(muscle_m.getName() + '_' + model.getCoordinateSet().get(j).getName())

        # select frames in time window
        if tstart is None:
            tstart = t.iloc[0]
        if tend is None:
            tend = t.iloc[-1]
        indices = np.where((t >= tstart) & (t <= tend))[0]
        # pre allocate output
        dM = np.zeros((len(indices), self.nMuscles * self.ncoord))

        # loop over all frames
        cti = 0
        for i in indices:
            # set state from current frame
            for j in range(0, len(columnLabels)):
                index = stateVariableNamesStr.index(columnLabels[j])
                state_vars.set(index, Qs[i, j])
            model.setStateVariableValues(s, state_vars)
            model.realizePosition(s)
            # get relevant muscles for this coordinate
            for j in range(0, self.ncoord):
                # get relevant coordinates for this muscle
                muscles_sel = self.dofs_dm[columnLabels[j]]
                # loop over muscles to get the moment arm
                for m in muscles_sel:
                    muscle_m = forceset.get(self.muscle_names[m])
                    muscle_m = osim.Muscle.safeDownCast(muscle_m)
                    # this step is computationally very expensive
                    dM[cti, m + j * self.nMuscles] = muscle_m.computeMomentArm(s, model.getCoordinateSet().get(j))
            # print current stage per 500 processed frames
            if np.remainder(cti, 100) == 0:
                print('computing moment arms: frame ' , i , '/' ,  nfr)
            # update counter
            cti = cti+1

        # return moment arms as a dataframe
        data = np.concatenate(
            (np.expand_dims(t[indices], axis=1), dM), axis=1)
        columns = ['time'] + self.dM_names
        moment_arms = pd.DataFrame(data=data, columns=columns)

        return moment_arms

    # computing moment arms for a model is very slow because we always compute the moment arms of a particular muscle
    # for all dofs (e.g. we evaluate moment arm of soleus round shoulder joint). We can reduce computation time by first
    # identifying the relevant dofs for each muscle
    def identify_relevant_dofs_dM(self):
        # init state of model
        model = osim.Model(self.modelpath)
        s = model.initSystem()
        state_vars = model.getStateVariableValues(s)
        force_set = model.getMuscles()
        state_vars.setToZero()
        model.setStateVariableValues(s, state_vars)
        model.realizePosition(s)
        forceset = model.getForceSet()

        # get state variable names
        stateVariableNames = model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]

        # random states
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[0])
        #Qs = np.random.rand(50, len(columnLabels)) - 0.5

        # loop over dofs
        self.dofs_dm = {}
        for i in range(0, len(columnLabels)):
            xvar = np.array([-0.5, 0, 0.5])         # vary states
            index = stateVariableNamesStr.index(columnLabels[i])
            lmt = np.zeros((len(xvar), self.nMuscles))
            for j in range(0, len(xvar)):
                state_vars.set(index, xvar[j])
                model.setStateVariableValues(s, state_vars)
                model.realizePosition(s)
                # loop over muscles to get muscle-tendon length
                for m in range(0, self.nMuscles):
                    muscle_m = forceset.get(self.muscle_names[m])
                    muscle_m = osim.Muscle.safeDownCast(muscle_m)
                    lmt[j, m] = muscle_m.getLength(s)
            # evaluate if moment arms changes for this change in dof
            d_lmt = np.min(lmt, axis = 0) - np.max(lmt, axis = 0)
            index_muscles = np.where(np.abs(d_lmt)>0.0001)
            self.dofs_dm[columnLabels[i]] = index_muscles[0]

    def compute_lmt(self, boolprint = True, tstart = None, tend = None):
        # read all ik file if needed
        if len(self.ikdat) == 0:
            self.read_ikfiles()

        # compute for each ik file the muscle tendon lengths
        self.lmt_dat = []
        for ifile in range(0, self.nfiles):
            # get muscle tendon lengths
            lmt = self.get_LMT_ifile(ifile, tstart = tstart, tend= tend)

            # print to a csv file -- lmt
            outpath =  self.lmt_folder
            if not os.path.exists(outpath ):
                os.makedirs(outpath)
            trialname = f'{self.ikfiles[ifile].stem}'
            outfile = os.path.join(outpath, trialname + '_lmt.csv')
            if boolprint:
                lmt.to_csv(outfile)
            self.lmt_dat.append(lmt)

    def compute_dM(self, boolprint = True, fastversion = True,
                   tstart = None, tend = None):
        # read all ik file if needed
        if len(self.ikdat) == 0:
            self.read_ikfiles()

        # find relevant dofs for each muscle in fast version
        if fastversion:
            self.identify_relevant_dofs_dM()

        # compute for each ik file the muscle tendon lengths
        self.dm_dat = []
        for ifile in range(0, self.nfiles):
            # get moment arms
            if fastversion:
                moment_arm = self.get_dM_ifile_fast(ifile, tstart = tstart, tend = tend)
            else:
                moment_arm = self.get_dM_ifile(ifile)

            # print to a csv file -- lmt
            outpath =  self.moment_arm_folder
            if not os.path.exists(outpath ):
                os.makedirs(outpath)
            trialname = f'{self.ikfiles[ifile].stem}'
            outfile = os.path.join(outpath, trialname + '_dM.csv')
            if boolprint:
                moment_arm.to_csv(outfile)
            self.dm_dat.append(moment_arm)

    # inverse kinematics using api
    def compute_inverse_kinematics(self, boolRead = True, overwrite = False):
        # solves inverse dynamics on all trc files
        output_settings = os.path.join(self.ik_directory, 'settings')
        if not os.path.isdir(output_settings):
            os.makedirs(output_settings)
        # solve inverse dynamics for this trial
        InverseKinematics(model_input=self.modelpath,
                        xml_input=self.general_ik_settings,
                        xml_output=output_settings,
                        trc_files=self.trcfiles,
                        mot_output=self.ik_directory,
                        overwrite = overwrite)
        if boolRead:
            self.read_ikfiles()

    # generate external loads files
    # here we want to link each grf file to the correct motion. We assume that the filename is similar.
    def match_marker_grf_files(self, overwrite = False):
        self.matched_grfs = []
        # the grf files are in self.folder_grfdat
        if self.folder_grfdat is not None:
            # try to find the grf files
            #   1. get all files with .sto extension in grf folder
            #grf_files = glob.glob(os.path.join(self.folder_grfdat, ".sto"))
            # find all .trc files in a folder
            grf_files = []
            for file in os.listdir(self.folder_grfdat):
                # Check if the file ends with .mot
                if file.endswith('.mot'):
                    grf_files.append(os.path.join(self.folder_grfdat, file))
            # convert to Path objects
            if len(grf_files)>0:
                if not isinstance(grf_files[0], Path):
                    grf_files= [Path(i) for i in grf_files]

            # add to variable
            self.grf_files = grf_files
            grf_filenames = []
            for itrial in self.grf_files:
                grf_filenames.append(itrial.stem)

            if len(grf_files) != self.nfiles:
                print('warning found ', len(grf_filenames), ' grf files and ', self.nfiles, ' mocap files')
            #   2. search for each filename the most likely candidate
            for file in self.filenames:
                # search for all files with the same name
                most_similar = difflib.get_close_matches(file, grf_filenames, n=1)
                if most_similar:
                    index_of_most_similar = grf_filenames.index(most_similar[0])
                    grf_file_sel = grf_files[index_of_most_similar]
                    self.matched_grfs.append(grf_file_sel)
                else:
                    print("no grf file found for " + file)
                    self.matched_grfs.append('no grf file found')
        else:
            print('please use myobj.set_folder_grfdat before creating external loads files')

    # inverse dynamics using api
    def compute_inverse_dynamics(self, boolRead= True):
        # computes inverse dynamics for all ik files
        # id output settings
        output_settings = os.path.join(self.id_directory, 'settings')
        if not os.path.isdir(output_settings):
            os.makedirs(output_settings)
        # check if we add external loads
        if self.folder_grfdat is not None:
            self.match_marker_grf_files() #  creates a list with matching
        # solve inverse dynamics for this trial
        InverseDynamics(model_input=self.modelpath,
                        xml_input=self.general_id_settings,
                        xml_output=output_settings,
                        mot_files=self.ikfiles,
                        sto_output=self.id_directory,
                        xml_forces=self.extload_settings,
                        forces_dir=self.folder_grfdat,
                        force_names= self.matched_grfs)
        # all idfiles assigned to this
        if boolRead:
            self.read_idfiles()

    # body kinematics using api
    def compute_bodykin(self, boolRead = True, overwrite = False):
        # function to compute bodykinematics
        bkdir = (f'{self.bodykin_folder.resolve()}')
        if not os.path.isdir(bkdir):
            os.makedirs(bkdir)
        bodykinematics(self.modelpath, bkdir, self.ikfiles, overwrite = overwrite)
        # read the bk files
        if boolRead:
            self.read_bkfiles()

    # compute kinetic energy bodies
    def compute_Ekin_bodies(self):
        print('start computation kinetic energy')
        # computes kinetic energy of all rigid bodies
        bk_header = self.bk_header
        model = osim.Model(self.modelpath)
        model.initSystem()
        nbodies = model.getBodySet().getSize()
        bodyset = model.getBodySet()
        Ekin_trials =[]
        for ifile in range(0, self.nfiles):
            bk_pos = self.bk_pos[ifile]
            bk_vel = self.bk_vel[ifile]
            if len(bk_pos)>0 and len(bk_vel)>0:
                # I like to work with pandas tables
                df_pos = pd.DataFrame(bk_pos, columns=bk_header)
                df_vel = pd.DataFrame(bk_vel, columns=bk_header)
                nfr = len(df_pos.time)
                Ekin = np.full([nfr, nbodies], np.nan)
                for i in range(0, nbodies):

                    # get inertia tensor of opensim body in local coordinate system
                    I_body = osim_body_I(bodyset.get(i).getInertia())
                    m = bodyset.get(i).getMass()

                    # compute angular momentum at each frame
                    bodyName = bodyset.get(i).getName()
                    nfr = len(df_pos.time)
                    fi_dot = np.zeros([nfr, 3])
                    fi_dot[:, 0] = df_vel[bodyName + '_Ox']
                    fi_dot[:, 1] = df_vel[bodyName + '_Oy']
                    fi_dot[:, 2] = df_vel[bodyName + '_Oz']

                    fi = np.zeros([nfr, 3])
                    fi[:, 0] = df_pos[bodyName + '_Ox']
                    fi[:, 1] = df_pos[bodyName + '_Oy']
                    fi[:, 2] = df_pos[bodyName + '_Oz']

                    r_dot = np.zeros([nfr, 3])
                    r_dot[:, 0] = df_vel[bodyName + '_X']
                    r_dot[:, 1] = df_vel[bodyName + '_Y']
                    r_dot[:, 2] = df_vel[bodyName + '_Z']

                    # inertia in world coordinate system
                    for t in range(0, nfr):
                        T_Body = transform(fi[t, 0], fi[t, 1], fi[t, 2])
                        I_world = T_Body.T * I_body * T_Body

                        # rotational kinetic energy
                        Ek_rot = 0.5 * np.dot(np.dot(fi_dot[t, :].T, I_world), fi_dot[t, :])

                        # translational kinetic energy
                        Ek_transl = 0.5 * m * np.dot(r_dot[t, :].T, r_dot[t, :])

                        # total kinetic energy
                        Ekin[t, i] = Ek_rot + Ek_transl
            else:
                Ekin = []
            Ekin_trials.append(Ekin)
            print('... file ' + str(ifile+1) + '/' + str(self.nfiles))
        return(Ekin_trials)

    def compute_linear_impulse_bodies(self):
        # computes the linear impulse of all bodies
        print('started with computation linear impulse of all bodies')
        model = osim.Model(self.modelpath)
        model.initSystem()
        nbodies = model.getBodySet().getSize()
        bodyset = model.getBodySet()
        impulse_trials =[]
        for ifile in range(0, self.nfiles):
            bk_pos = self.bk_pos[ifile]
            bk_vel = self.bk_vel[ifile]
            if len(bk_pos) > 0 and len(bk_vel) > 0:
                # I like to work with pandas tables
                nfr = len(bk_vel.time)
                linear_impulse = np.full([nfr, nbodies, 3], np.nan)
                for i in range(0, nbodies):
                    m = bodyset.get(i).getMass()
                    bodyName = bodyset.get(i).getName()
                    r_dot = np.zeros([nfr, 3])
                    r_dot[:, 0] = bk_vel[bodyName + '_X']
                    r_dot[:, 1] = bk_vel[bodyName + '_Y']
                    r_dot[:, 2] = bk_vel[bodyName + '_Z']
                    linear_impulse[:,i,:] = m * r_dot
            else:
                linear_impulse = []
            impulse_trials.append(linear_impulse)
            print('... file ' + str(ifile + 1) + '/' + str(self.nfiles))
        return(impulse_trials)


    # Getters
    def get_modelpath(self):
        return self.modelpath

    def get_trcfiles(self):
        return self.trcfiles

    def get_ikfiles(self):
        return self.ikfiles

    def get_nfiles(self):
        return self.nfiles

    def get_filenames(self):
        return self.filenames

    def get_general_id_settings(self):
        return self.general_id_settings

    def get_extload_settings(self):
        return self.extload_settings

    def get_ext_loads_dir(self):
        return self.ext_loads_dir

    def get_ik_directory(self):
        return self.ik_directory

    def get_id_directory(self):
        return self.id_directory

    def get_bodykin_folder(self):
        return self.bodykin_folder

    def get_extload_files(self):
        return self.extload_files

    def get_id_dat(self):
        return self.id_dat

    def get_bk_pos(self):
        return self.bk_pos

    def get_bk_vel(self):
        return self.bk_vel

    def get_ikdat(self):
        return self.ikdat

    def get_marker_dir(self):
        return self.marker_dir

    # Setters
    def set_modelpath(self, modelpath):
        self.modelpath = modelpath

    def set_trcfiles(self, trcfiles):
        self.trcfiles = trcfiles

    def set_nfiles(self, nfiles):
        self.nfiles = nfiles

    def set_filenames(self, filenames):
        self.filenames = filenames

    def set_bodykin_folder(self, bodykin_folder):
        if not isinstance(bodykin_folder, Path):
            bodykin_folder = Path(bodykin_folder)
        self.bodykin_folder = bodykin_folder

    def set_extload_files(self, extload_files):
        self.extload_files = extload_files

    def set_id_dat(self, id_dat):
        self.id_dat = id_dat

    def set_bk_pos(self, bk_pos):
        self.bk_pos = bk_pos

    def set_bk_vel(self, bk_vel):
        self.bk_vel = bk_vel

    def set_ikdat(self, ikdat):
        self.ikdat = ikdat

    def set_marker_dir(self, marker_dir):
        self.marker_dir = marker_dir

    def set_folder_grfdat(self, folder):
        self.folder_grfdat = folder

    def set_momentarm_folder(self, folder):
        self.moment_arm_folder = folder

    def set_lmt_folder(self, folder):
        self.lmt_folder = folder


    # set functions (not from chatgpt)
    #---------------
    def set_general_id_settings(self, general_id_settings):
        self.general_id_settings = general_id_settings

    def set_generic_external_loads(self, general_loads_settings):
        self.extload_settings = general_loads_settings

    def set_id_directory(self, id_directory):
        if not isinstance(id_directory, Path):
            id_directory = Path(id_directory)
        if not os.path.isdir(id_directory):
            os.makedirs(id_directory)
        self.id_directory = id_directory

    def set_ext_loads_dir(self, ext_loads_dir):
        if not isinstance(ext_loads_dir, Path):
            ext_loads_dir = Path(ext_loads_dir)
        self.ext_loads_dir = ext_loads_dir

    def set_general_ik_settings(self, general_ik_settings):
        self.general_ik_settings = general_ik_settings

    def set_ik_directory(self, ik_directory):
        if not isinstance(ik_directory, Path):
            ik_directory = Path(ik_directory)
        if not os.path.isdir(ik_directory):
            os.makedirs(ik_directory)
        self.ik_directory = ik_directory

    def set_marker_directory(self, marker_dir):
        if not isinstance(marker_dir, Path):
            marker_dir = Path(marker_dir)
        self.marker_dir = marker_dir

    # specific functions for project at Ajax:
    #------------------------------------------

    def create_extloads_soccer(self, dt_hit = 0.1, mbal = 0.450):
        # this approach is based on the assumption that there is a conservation of linear
        # impulse of the left and right foot
        impulse_trials = self.compute_linear_impulse_bodies()
        self.extload_files = []

        for itrial in range(0, self.nfiles):
            # convert to pandas structure
            df_ik = self.ikdat[itrial]
            df_pos = self.bk_pos[itrial]
            df_vel = self.bk_vel[itrial]

            # assumption that persons hits the ball at max velocity of the foot
            thit = df_vel.time.iloc[np.argmax(df_vel.calcn_r_X)]

            # get linear impulse of the foot
            p_footR = impulse_trials[itrial][:, self.bodynames.index('calcn_r'), 0]

            # compute release velocity ball based on assumption conservation of linear impulse
            it0 = np.where(df_ik.time >= thit)[0][0]
            if df_ik.time.iloc[-1] > (thit + dt_hit):
                itend = np.where(df_ik.time >= (thit + dt_hit))[0][0]
                v_ball_post = (p_footR[it0] - p_footR[itend]) / mbal
            else:
                v_ball_post = 0
                print('Problem with computing ball velocity ', Path(self.ikfiles[itrial]).stem)
                itend = it0

            # create external loads file for soccer kick -- right leg
            nfr = len(df_ik.time)
            dat_ballFoot = np.zeros([nfr, 10])
            dat_ballFoot[:, 0] = df_ik.time
            dat_ballFoot[range(it0, itend), 1] = -(p_footR[it0] - p_footR[itend]) / dt_hit
            dat_ballFoot[:, 4] = df_pos.calcn_r_X
            dat_ballFoot[:, 5] = df_pos.calcn_r_Y
            dat_ballFoot[:, 6] = df_pos.calcn_r_Z
            dat_headers = ['time', 'ball_force_vx', 'ball_force_vy', 'ball_force_vz', 'ball_force_px', 'ball_force_py',
                           'ball_force_pz', 'ground_torque_x', 'ground_torque_y', 'ground_torque_z']
            forcesfilename = os.path.join(self.ext_loads_dir, Path(self.ikfiles[itrial]).stem + '.sto')
            WriteMotionFile(dat_ballFoot, dat_headers, forcesfilename)
            self.extload_files.append(forcesfilename)

    # various functions for specific projects
    def print_file_timinghit_ball(self, outfile, dt_hit = 0.1):
        # Create the initial data
        df = pd.DataFrame(columns=['trialname', 'time_hit_ball'])
        for itrial in range(0, self.nfiles):
            # convert to pandas structure
            df_ik = self.ikdat[itrial]
            df_pos = self.bk_pos[itrial]
            df_vel = self.bk_vel[itrial]

            # assumption that persons hits the ball at max velocity of the foot
            thit = df_vel.time.iloc[np.argmax(df_vel.calcn_r_X)]

            # add to pandas dataframe
            new_row = pd.DataFrame({'trialname': [self.ikfiles[itrial].stem], 'time_hit_ball': [thit]})
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(outfile)
        print('printend timing hitting ball ', outfile)



def transform(Rx, Ry, Rz, bool_deg_input = True):
    """
    Compute transformation matrix for x-y'-z'' intrinsic Euler rotations
    (the OpenSim convention).

    Parameters:
    Rx (float): Rotation around the x-axis in degrees.
    Ry (float): Rotation around the y-axis in degrees.
    Rz (float): Rotation around the z-axis in degrees.

    Returns:
    numpy.ndarray: The 3x3 transformation matrix.
    """
    # Convert degrees to radians
    if bool_deg_input:
        Rx = np.radians(Rx)
        Ry = np.radians(Ry)
        Rz = np.radians(Rz)

    # Compute transformation matrix elements
    R11 = np.cos(Ry) * np.cos(Rz)
    R12 = -np.cos(Ry) * np.sin(Rz)
    R13 = np.sin(Ry)
    R21 = np.cos(Rx) * np.sin(Rz) + np.sin(Rx) * np.sin(Ry) * np.cos(Rz)
    R22 = np.cos(Rx) * np.cos(Rz) - np.sin(Rx) * np.sin(Ry) * np.sin(Rz)
    R23 = -np.sin(Rx) * np.cos(Ry)
    R31 = np.sin(Rx) * np.sin(Rz) - np.cos(Rx) * np.sin(Ry) * np.cos(Rz)
    R32 = np.sin(Rx) * np.cos(Rz) + np.cos(Rx) * np.sin(Ry) * np.sin(Rz)
    R33 = np.cos(Rx) * np.cos(Ry)

    # Create the transformation matrix
    R = np.array([[R11, R12, R13],
                  [R21, R22, R23],
                  [R31, R32, R33]])

    return R

def osim_body_I(Inertia):
    # returns Inertia tensor as 3x3 nummpy ndraay based on an opensim Inertia object
    I_osim_Mom = Inertia.getMoments()
    I_osim_Prod = Inertia.getProducts()
    I_body = np.zeros([3,3])
    I_body[0, 0] = I_osim_Mom.get(0)
    I_body[1, 1] = I_osim_Mom.get(1)
    I_body[2, 2] = I_osim_Mom.get(0)
    I_body[1, 0]= I_osim_Prod.get(0)
    I_body[0, 1]= I_osim_Prod.get(0)
    I_body[0, 2]= I_osim_Prod.get(1)
    I_body[2, 0]= I_osim_Prod.get(1)
    I_body[2, 1]= I_osim_Prod.get(2)
    I_body[1, 2]= I_osim_Prod.get(2)
    return(I_body)

def WriteMotionFile(data_matrix, colnames, filename):
    datarows, datacols = data_matrix.shape
    time = data_matrix[:, 0]
    range_values = [time[0], time[-1]]

    if len(colnames) != datacols:
        raise ValueError(f'Number of column names ({len(colnames)}) does not match the number of columns in the data ({datacols})')

    # Open the file for writing
    try:
        with open(filename, 'w') as fid:
            # Write MOT file header
            fid.write(f'{filename}\nnRows={datarows}\nnColumns={datacols}\n\n')
            fid.write(f'name {filename}\ndatacolumns {datacols}\ndatarows {datarows}\nrange {range_values[0]} {range_values[1]}\nendheader\n')

            # Write column names
            cols = '\t'.join(colnames) + '\n'
            fid.write(cols)

            # Write data
            for i in range(datarows):
                row = '\t'.join([f'{value:20.10f}' for value in data_matrix[i, :]]) + '\n'
                fid.write(row)

    except IOError:
        print(f'\nERROR: {filename} could not be opened for writing...\n')


