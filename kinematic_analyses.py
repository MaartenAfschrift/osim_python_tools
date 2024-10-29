import opensim as osim
from pathlib import Path
import os
import numpy as np
import pandas as pd

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
        self.outputdir = outputdir
        self.ikfiles = ikfiles
        self.overwrite = overwrite
        self.ikdat = None
        self.get_n_muscles()
        self.get_model_coordinates()

        if not isinstance(ikfiles, list):
            self.ikfiles = [ikfiles]
        else:
            self.ikfiles = ikfiles

        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]

    def read_ik(self, ik_file):
        if ik_file.exists():
            ik_data = ReadMotionFile(ik_file)
        else:
            ik_data = []
            print('could find read file ', ik_file)
        return(ik_data)

    def compute_lmt(self, tstart=None, tend=None):

        # read ik files
        if self.ikdat is None:
            self.ikdat = []
            for ikfile in self.ikfiles:
                ik_data = self.read_ik(str(ikfile))
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



