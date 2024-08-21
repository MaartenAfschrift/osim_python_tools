import opensim as osim
from pathlib import Path

class bodykinematics:
    def __init__(self, modelfile, outputdir, ikfiles):
        self.modelpath = modelfile
        self.outputdir = outputdir
        self.ikfiles = ikfiles

        if not isinstance(ikfiles, list):
            self.ikfiles = [ikfiles]
        else:
            self.ikfiles = ikfiles

        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]

        # run bodykinematics on all trials
        for itrial in self.ikfiles:
            self.run_bk(itrial)



    def run_bk(self, trial):
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
        filename = trial.stem
        tool.setName(filename)

        # run the analysis
        tool.setCoordinatesFileName((f'{trial.resolve()}'))
        tool.run()

    def read_results(self):
        print('ToDo')

    def plot_results(self):
        print('ToDo')








