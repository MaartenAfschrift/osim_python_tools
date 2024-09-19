import opensim as osim
from pathlib import Path
import os

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








