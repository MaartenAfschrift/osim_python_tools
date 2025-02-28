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

    def compute_lmt(self, tstart=None, tend=None, selected_muscles = None):

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
            lmt = self.get_LMT_ifile(ifile, tstart=tstart, tend=tend, selected_muscles = selected_muscles)

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


    def get_LMT_ifile(self, ifile, tstart = None, tend = None, selected_muscles = None):

        # get time vector and number of frames.
        t = self.ikdat[ifile].time
        nfr = len(t)

        # test if we want to compute for all muscles or only a subset
        if selected_muscles is None:
            nmuscles = self.nMuscles
            muscle_names = self.muscle_names
        else:
            nmuscles = len(selected_muscles)
            muscle_names = selected_muscles

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
        else:
            if tstart < t.iloc[0]:
                tstart = t.iloc[0]

        if tend is None:
            tend = t.iloc[-1]
            if tend > t.iloc[-1]:
                tend = t.iloc[-1]
        indices = np.where((t >= tstart) & (t <= tend))[0]
        # pre allocate output
        lMT = np.zeros((len(indices), nmuscles))

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
            for m in range(0, nmuscles):
                muscle_m = forceSet.get(muscle_names[m])
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
        columns = ['time'] + muscle_names
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

    def identify_relevant_dofs_dM(self, selected_dofs = None, selected_muscles = None):
        # selected muscles
        if selected_muscles is None:
            selected_muscles = self.muscle_names
        nmuscles = len(selected_muscles)

        # find index of selected muscles
        index_muscles = np.zeros(nmuscles, dtype = int)
        ct = 0
        for m in selected_muscles:
            index_muscles[ct] = int(self.muscle_names.index(m))
            ct =ct + 1

        # init state of model
        s = self.model.initSystem()
        forceset = self.model.getMuscles()

        # test if we have to loop over all dofs or only a subset
        if selected_dofs is None:
            selected_dofs = self.coord_names

        # test if we want to compute for all muscles or only a subset
        if selected_muscles is None:
            muscle_names = self.muscle_names
        else:
            muscle_names = selected_muscles

        # pre allocate self.dofs_dm
        self.dofs_dm = {}
        for dof in selected_dofs:
            xvar = np.array([-0.5, 0, 0.5])  # vary states
            lmt = np.zeros((len(xvar), len(muscle_names)))
            for j in range(0, len(xvar)):
                self.model.getCoordinateSet().get(dof).setValue(s, xvar[j])
                self.model.realizePosition(s)
                # loop over muscles to get muscle-tendon length
                for m in range(0, nmuscles):
                    muscle_m = forceset.get(selected_muscles[m])
                    muscle_m = osim.Muscle.safeDownCast(muscle_m)
                    lmt[j, ctm] = muscle_m.getLength(s)
                    ctm = ctm + 1
            # evaluate if moment arms changes for this change in dof
            d_lmt = np.min(lmt, axis=0) - np.max(lmt, axis=0)
            index_sel = np.where(np.abs(d_lmt) > 0.0001)
            if len(index_sel[0]) > 0:
                self.dofs_dm[columnLabels[i]] = index_muscles[index_sel[0]]
            else:
                self.dofs_dm[columnLabels[i]] = []

    # compute moment arms for all muscles and dofs
    def get_dm_ifile(self, ifile, tstart = None, tend = None, limitfilesize = True,
                    selected_muscles = None, selected_dofs = None):            

        # get time vector and number of frames.
        t = self.ikdat[ifile].time
        nfr = len(t)
        ikdat = self.ikdat[ifile]

        # init state of model
        s = self.model.initSystem()
        forceset = self.model.getMuscles()
        state_vars = self.model.getStateVariableValues(s)

        # test if we have to loop over all dofs or only a subset
        if selected_dofs is None:
            selected_dofs = self.coord_names

        # test if we want to compute for all muscles or only a subset
        if selected_muscles is None:
            nmuscles = self.nMuscles
            muscle_names = self.muscle_names
        else:
            nmuscles = len(selected_muscles)
            muscle_names = selected_muscles

        # get all headers
        self.dM_names = []
        if limitfilesize:
            ct_dof = -1
            for dof in self.dofs_dm:
                ct_dof = ct_dof + 1
                if len(self.dofs_dm[dof])>0:
                    for m in self.dofs_dm[dof]:
                        muscle_m = forceset.get(self.muscle_names[m])
                        self.dM_names.append(muscle_m.getName() + '_' +
                                             self.model.getCoordinateSet().get(ct_dof).getName())
        else:
            for j in range(0, self.ncoord):
                for m in range(0, self.nMuscles):
                    muscle_m = forceset.get(self.muscle_names[m])
                    self.dM_names.append(muscle_m.getName() + '_' +
                                         self.model.getCoordinateSet().get(j).getName())

        # get state variable names
        stateVariableNames = self.model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]

        # read the ik file as a time series
        [columnLabels, table] = self.read_ik_as_timeseries(self.ikfiles[0])
        Qs = table.getMatrix().to_numpy()

        # select frames in time window
        if tstart is None:
            tstart = t.iloc[0]
        else:
            if tstart < t.iloc[0]:
                tstart = t.iloc[0]

        if tend is None:
            tend = t.iloc[-1]
        else:
            if tend > t.iloc[-1]:
                tend = t.iloc[-1]

        indices = np.where((t >= tstart) & (t <= tend))[0]
        
        # pre allocate output [all these zeros needed ?, for not relevant muscles as well ?]
        if limitfilesize:
            dM = np.zeros((len(indices), len(self.dM_names)))
        else:
            dM = np.zeros((len(indices), self.nMuscles * self.ncoord))

        # loop over all frames
        cti = 0
        for i in indices:
            # set state from current frame
            ct_col = 0
            for j in range(0, len(columnLabels)):
                index = stateVariableNamesStr.index(columnLabels[j])
                state_vars.set(index, Qs[i,j])
            self.model.setStateVariableValues(s, state_vars)
            self.model.realizePosition(s)
            # get relevant muscles for this coordinate
            ctdof = -1
            for dof in selected_dofs:
                ctdof = ctdof + 1
                # get relevant muscles for this dof
                muscles_sel = self.dofs_dm[dof]
                # loop over all relevant muscles to get the moment arm
                for im in muscles_sel:
                    muscle = muscle_names[im]
                    muscle_m = forceset.get(muscle)
                    muscle_m = osim.Muscle.safeDownCast(muscle_m)
                    # compute moment arms for given state
                    # this step is computationally very expensive
                    if limitfilesize:
                        dM[cti, ct_col] = muscle_m.computeMomentArm(s, self.model.getCoordinateSet().get(j))
                        ct_col = ct_col + 1
                    else:
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


    def compute_dm(self, boolprint = True, tstart = None, tend = None, selected_muscles = None,
                   limitfilesize = True, selected_dofs = None):

        # find relevant dofs for each muscle in fast version
        if selected_muscles is None:
            selected_muscles = self.muscle_names

        if self.dofs_dm is None:
            self.identify_relevant_dofs_dM(selected_dofs = selected_dofs,
                                           selected_muscles = selected_muscles)

        # read ik files
        if self.ikdat is None:
            self.ikdat = []
            for ikfile in self.ikfiles:
                ik_data = self.read_ik(ikfile)
                self.ikdat.append(ik_data)

        self.dm_dat = []
        for ifile in range(0, self.nfiles):
            # get moment arms
            moment_arm = self.get_dm_ifile(ifile, tstart = tstart, tend = tend,
                                           selected_muscles = selected_muscles,
                                          limitfilesize= limitfilesize)
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






