import opensim as osim
from pathlib import Path
import os
import numpy as np
import pandas as pd
from general_utilities import readMotionFile

class bodykinematics:
    def __init__(self, modelfile, outputdir, ikfiles, overwrite = False):
        self.modelpath = modelfile
        self.outputdir = outputdir
        self.ikfiles = ikfiles
        self.overwrite = overwrite

        if not isinstance(ikfiles, list):
            self.ikfiles = [ikfiles]
        else:
            self.ikfiles = ikfiles

        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]

        # run bodykinematics on all trials
        for itrial in self.ikfiles:
            self.run_bk(itrial, bool_overwrite = self.overwrite)

    def run_bk(self, trial, bool_overwrite = False):

        # output path and file
        filename = trial.stem
        output_bk_pos_file = os.path.join(self.outputdir, filename + '_BodyKinematics_pos_global.sto')

        if (not (os.path.exists(output_bk_pos_file))) | bool_overwrite:

            # run body kinematics analysis
            model = osim.Model(self.modelpath)
            bk = osim.BodyKinematics()

            # get start and end of IK file
            motion = osim.Storage(f'{trial.resolve()}')
            tstart = motion.getFirstTime()
            tend = motion.getLastTime()

            # general settings bodykinematics
            bk.setStartTime(tstart)
            bk.setEndTime(tend)
            bk.setOn(True)
            bk.setStepInterval(1)
            bk.setInDegrees(True)

            # add analysis to the model
            model.addAnalysis(bk)
            model.initSystem()

            # create an analysis tool
            tool = osim.AnalyzeTool(model)
            tool.setLoadModelAndInput(True)
            tool.setResultsDir(self.outputdir)
            tool.setInitialTime(tstart)
            tool.setFinalTime(tend)
            tool.setName(filename)

            # run the analysis
            tool.setCoordinatesFileName((f'{trial.resolve()}'))
            tool.run()
        else:
            print(output_bk_pos_file + ' does already exist, skipping bodykin analysis')

    def read_results(self):
        print('ToDo')

    def plot_results(self):
        print('ToDo')

# compute muscle tendon lengths through the api
class lmt_api():
    def __init__(self, modelfile, ikfiles, outputdir,
                 overwrite = False,
                 tstart = None,
                 tend = None,
                 ikdat = None):
        self.modelpath = modelfile
        # read opensim model
        self.model = osim.Model(self.modelpath)
        self.model.initSystem()
        self.outputdir = outputdir
        self.ikfiles = ikfiles
        self.overwrite = overwrite
        self.ikdat = ikdat
        self.get_n_muscles()
        self.get_model_coordinates()

        if not isinstance(ikfiles, list):
            self.ikfiles = [ikfiles]
        else:
            self.ikfiles = ikfiles

        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]

        self.nfiles = len(self.ikfiles)



    def read_ik(self, ik_file):
        if ik_file.exists():
            ik_data = readMotionFile(ik_file)
        else:
            ik_data = []
            print('could find read file ', ik_file)
        return(ik_data)

    def compute_lmt(self, tstart=None, tend=None):

        # read ik files
        if self.ikdat is None:
            self.ikdat = []
            for ikfile in self.ikfiles:
                ik_data = self.read_ik(ikfile)
                self.ikdat.append(ik_data)

        # compute for each ik file the muscle tendon lengths
        lmt_dat = []
        for ifile in range(0, self.nfiles):
            # get muscle tendon lengths
            lmt = self.get_LMT_ifile(ifile, tstart=tstart, tend=tend)

            # print to a csv file -- lmt
            outpath = self.outputdir
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            trialname = f'{self.ikfiles[ifile].stem}'
            outfile = os.path.join(outpath, trialname + '_lmt.csv')
            lmt.to_csv(outfile)
            lmt_dat.append(lmt)
            #
        return(lmt_dat)


    def get_LMT_ifile(self, ifile, tstart = None, tend = None):

        # get time vector and number of frames.
        t = self.ikdat[ifile].time
        nfr = len(t)

        # init state of model
        s = self.model.initSystem()
        state_vars = self.model.getStateVariableValues(s)
        force_set = self.model.getMuscles()
        state_vars.setToZero()
        self.model.setStateVariableValues(s, state_vars)
        self.model.realizePosition(s)
        forceSet = self.model.getForceSet()

        # read ik as time_series
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[ifile])
        Qs = table.getMatrix().to_numpy()

        # get state variable names
        stateVariableNames = self.model.getStateVariableNames()
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
        print('start computation lmt ', ifile)
        for i in indices:
            # set state from current frame
            for j in range(0, len(columnLabels)):
                index = stateVariableNamesStr.index(columnLabels[j])
                state_vars.set(index, Qs[i,j])
            self.model.setStateVariableValues(s, state_vars)
            self.model.realizePosition(s)
            # loop over muscles to get muscle-tendon length
            for m in range(0, self.nMuscles):
                muscle_m = forceSet.get(self.muscle_names[m])
                muscle_m = osim.Muscle.safeDownCast(muscle_m)
                lMT[cti,m] = muscle_m.getLength(s)
                # print current stage per 500 processed frames
            if np.remainder(cti, 200) == 0:
                print(' computing lmt: frame ', i-indices[0], '/', len(indices))
            # update counter
            cti = cti +1

        # return lmt as a dataframe
        data = np.concatenate(
            (np.expand_dims(t[indices], axis=1), lMT), axis=1)
        columns = ['time'] + self.muscle_names
        muscle_tendon_lengths = pd.DataFrame(data=data, columns=columns)

        return muscle_tendon_lengths

    # read ik file as a timeseries
    def read_ik_as_timeseries(self, ikfile):
        # read ik file as a time series table
        table = osim.TimeSeriesTable(str(ikfile))
        tableProcessor = osim.TableProcessor(table)
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
        time = np.asarray(table.getIndependentColumn())

        # convert to radians
        table = tableProcessor.processAndConvertToRadians(self.model)
        columnLabels = list(table.getColumnLabels())
        return(columnLabels, table)


    # get number of muscles in model
    def get_n_muscles(self):
        # Number of muscles.

        self.nMuscles = 0
        self.muscle_names = []
        forceSet = self.model.getForceSet()
        for i in range(forceSet.getSize()):
            c_force_elt = forceSet.get(i)
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1
                self.muscle_names.append(c_force_elt.getName())

    # get coordinates in the model
    def get_model_coordinates(self):
        self.model.initSystem()
        coord_set = self.model.getCoordinateSet()
        self.coord_names = []
        self.ncoord = 0
        for i in range(0, coord_set.getSize()):
            self.coord_names.append(coord_set.get(i).getName())
            self.ncoord = self.ncoord + 1

# compute muscle moment arms through the api
class moment_arm_api():
    def __init__(self, modelfile, ikfiles, outputdir,
                 overwrite = False,
                 tstart = None,
                 tend = None,
                 ikdat = None):
        self.modelpath = modelfile
        self.model = osim.Model(self.modelpath)
        self.model.initSystem()
        self.outputdir = outputdir
        self.ikfiles = ikfiles
        self.overwrite = overwrite
        self.ikdat = ikdat
        self.get_n_muscles()
        self.get_model_coordinates()
        self.dofs_dm = None
        self.dm_dat = None

        if not isinstance(ikfiles, list):
            self.ikfiles = [ikfiles]
        else:
            self.ikfiles = ikfiles

        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]

        self.nfiles = len(self.ikfiles)

    # computing moment arms for a model is very slow because we always compute
    # the moment arms of a particular musclefor all dofs (e.g. we evaluate moment
    # arm of soleus round shoulder joint). We can reduce computation time by first
    # identifying the relevant dofs for each muscle
    def identify_relevant_dofs_dM(self):
        # init state of model
        s = self.model.initSystem()
        state_vars = self.model.getStateVariableValues(s)
        force_set = self.model.getMuscles()
        state_vars.setToZero()
        self.model.setStateVariableValues(s, state_vars)
        self.model.realizePosition(s)
        forceset = self.model.getForceSet()

        # get state variable names
        stateVariableNames = self.model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]

        # random states
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[0])

        # loop over dofs
        self.dofs_dm = {}
        for i in range(0, len(columnLabels)):
            xvar = np.array([-0.5, 0, 0.5])  # vary states
            index = stateVariableNamesStr.index(columnLabels[i])
            lmt = np.zeros((len(xvar), self.nMuscles))
            for j in range(0, len(xvar)):
                state_vars.set(index, xvar[j])
                self.model.setStateVariableValues(s, state_vars)
                self.model.realizePosition(s)
                # loop over muscles to get muscle-tendon length
                for m in range(0, self.nMuscles):
                    muscle_m = forceset.get(self.muscle_names[m])
                    muscle_m = osim.Muscle.safeDownCast(muscle_m)
                    lmt[j, m] = muscle_m.getLength(s)
            # evaluate if moment arms changes for this change in dof
            d_lmt = np.min(lmt, axis=0) - np.max(lmt, axis=0)
            index_muscles = np.where(np.abs(d_lmt) > 0.0001)
            self.dofs_dm[columnLabels[i]] = index_muscles[0]

    # compute moment arms for all muscles and dofs
    def get_dm_ifile(self, ifile, tstart = None, tend = None):

        # get time vector and number of frames.
        t = self.ikdat[ifile].time
        nfr = len(t)

        # init state of model
        s = self.model.initSystem()
        state_vars = self.model.getStateVariableValues(s)
        force_set = self.model.getMuscles()
        state_vars.setToZero()
        self.model.setStateVariableValues(s, state_vars)
        self.model.realizePosition(s)
        forceset = self.model.getForceSet()

        # read ik as time_series
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[ifile])
        qs = table.getMatrix().to_numpy()

        # get state variable names
        state_variable_names = self.model.getStateVariableNames()
        state_variable_names_str = [
            state_variable_names.get(i) for i in range(
                state_variable_names.getSize())]

        # get all headers
        self.dM_names = []
        for j in range(0, self.ncoord):
            for m in range(0, self.nMuscles):
                muscle_m = forceset.get(self.muscle_names[m])
                self.dM_names.append(muscle_m.getName() + '_' +
                                     self.model.getCoordinateSet().get(j).getName())

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
                index = state_variable_names_str.index(columnLabels[j])
                state_vars.set(index, qs[i, j])
            self.model.setStateVariableValues(s, state_vars)
            self.model.realizePosition(s)
            # get relevant muscles for this coordinate
            for j in range(0, self.ncoord):
                # get relevant coordinates for this muscle
                muscles_sel = self.dofs_dm[columnLabels[j]]
                # loop over muscles to get the moment arm
                for m in muscles_sel:
                    muscle_m = forceset.get(self.muscle_names[m])
                    muscle_m = osim.Muscle.safeDownCast(muscle_m)
                    # compute moment arms for given state
                    # this step is computationally very expensive
                    dM[cti, m + j * self.nMuscles] =\
                        muscle_m.computeMomentArm(s, self.model.getCoordinateSet().get(j))
            # print current stage per 500 processed frames
            if np.remainder(cti, 500) == 0:
                print('computing moment arms: frame ' , i-indices[0] , '/' ,  len(indices))
            # update counter
            cti = cti+1

        # return moment arms as a dataframe
        data = np.concatenate(
            (np.expand_dims(t[indices], axis=1), dM), axis=1)
        columns = ['time'] + self.dM_names
        moment_arms = pd.DataFrame(data=data, columns=columns)

        return moment_arms

    def compute_dm(self, boolprint = True, tstart = None, tend = None):

        # find relevant dofs for each muscle in fast version
        if self.dofs_dm is None:
            self.identify_relevant_dofs_dM()

        # read ik files
        if self.ikdat is None:
            self.ikdat = []
            for ikfile in self.ikfiles:
                ik_data = self.read_ik(ikfile)
                self.ikdat.append(ik_data)


        # compute for each ik file the muscle tendon lengths
        self.dm_dat = []
        for ifile in range(0, self.nfiles):
            # get moment arms
            moment_arm = self.get_dm_ifile(ifile, tstart = tstart, tend = tend)

            # print to a csv file -- lmt
            outpath =  self.outputdir
            if not os.path.exists(outpath ):
                os.makedirs(outpath)
            trialname = f'{self.ikfiles[ifile].stem}'
            outfile = os.path.join(outpath, trialname + '_dM.csv')
            if boolprint:
                moment_arm.to_csv(outfile)
            self.dm_dat.append(moment_arm)
        # return dm datt
        return(self.dm_dat)


    # some generic opensim functions to read properties from the model
    #-------------------------------------------------------------------
    # read ik file as a timeseries
    def read_ik_as_timeseries(self, ikfile):
        # read ik file as a time series table
        table = osim.TimeSeriesTable(str(ikfile))
        tableProcessor = osim.TableProcessor(table)
        tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
        time = np.asarray(table.getIndependentColumn())

        # convert to radians
        self.model.initSystem()
        table = tableProcessor.processAndConvertToRadians(self.model)
        columnLabels = list(table.getColumnLabels())
        return (columnLabels, table)

    # get number of muscles in model
    def get_n_muscles(self):
        # Number of muscles.

        self.nMuscles = 0
        self.muscle_names = []
        forceSet = self.model.getForceSet()
        for i in range(forceSet.getSize()):
            c_force_elt = forceSet.get(i)
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1
                self.muscle_names.append(c_force_elt.getName())

    # get coordinates in the model
    def get_model_coordinates(self):
        self.model.initSystem()
        coord_set = self.model.getCoordinateSet()
        self.coord_names = []
        self.ncoord = 0
        for i in range(0, coord_set.getSize()):
            self.coord_names.append(coord_set.get(i).getName())
            self.ncoord = self.ncoord + 1

    def read_ik(self, ik_file):
        if ik_file.exists():
            ik_data = readMotionFile(ik_file)
        else:
            ik_data = []
            print('could find read file ', ik_file)
        return(ik_data)

# create class to compute whole body angular momentum from kinematics data






