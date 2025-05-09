"""
Inverse dynamic class in pyosim
"""
from pathlib import Path
import opensim as osim
import os

class InverseDynamics:
    """
    Inverse dynamic in pyosim

    Parameters
    ----------
    model_input : str
        Path to the osim model
    xml_input : str
        Path to the generic id xml
    xml_output : str
        Output path of the id xml
    xml_forces : str
        Path to the generic forces sensor xml
    forces_dir : str
        Path of the directory containing the forces files (`.sto`)
    mot_files : str, Path, list
        Path or list of path to the directory containing the motion files (`.mot`)
    sto_output : Path, str
        Output directory
    prefix : str, optional
        Optional prefix to put in front of the output filename (typically model name)
    low_pass : int, optional
        Cutoff frequency for an optional low pass filter on coordinates
    multi : bool, optional
        Launch InverseDynamics in multiprocessing if True
    force_names: list with name of the force files (in case the filename of your forces file is
        not the same as the mot_files)

    # Examples
    # --------
    # >>> from pyosim import Conf
    # >>> from pyosim import InverseDynamics
    # >>> from pathlib import Path
    # >>>
    # >>> PROJECT_PATH = Path('../Misc/project_sample')
    # >>> TEMPLATES_PATH = PROJECT_PATH / '_templates'
    # >>>
    # >>> participant = 'dapo'
    # >>> model = 'wu'
    # >>>
    # >>> trials = [ifile for ifile in (PROJECT_PATH / participant / '1_inverse_kinematic').glob('*.mot')]
    # >>> conf = Conf(project_path=PROJECT_PATH)
    # >>> onsets = conf.get_conf_field(participant, ['onset'])
    # >>>
    # >>> idyn = InverseDynamics(
    # >>>     model_input=f"{PROJECT_PATH / participant / '_models' / model}_scaled_markers.osim",
    # >>>     xml_input=f'{TEMPLATES_PATH / model}_ik.xml',
    # >>>     xml_output=f"{PROJECT_PATH / participant / '_xml' / model}_ik.xml",
    # >>>     xml_forces=f'{TEMPLATES_PATH}/forces_sensor.xml',
    # >>>     forces_dir=f"{PROJECT_PATH / participant / '0_forces'}",
    # >>>     mot_files=trials,
    # >>>     sto_output=f"{(PROJECT_PATH / participant / '2_inverse_dynamic').resolve()}",
    # >>>     prefix=model,
    # >>>     low_pass=10
    # >>> )
    # """

    def __init__(
            self,
            model_input,
            xml_input,
            xml_output,
            mot_files,
            sto_output,
            xml_forces=None,
            forces_dir=None,
            prefix=None,
            low_pass=None,
            multi=False,
            force_names = None,
    ):
        self.model_input = model_input
        self.xml_input = xml_input
        self.xml_output = xml_output
        self.sto_output = sto_output
        self.xml_forces = xml_forces
        self.forces_dir = forces_dir
        self.low_pass = low_pass
        self.multi = multi
        self.prefix = prefix
        self.force_names = force_names

        if not isinstance(mot_files, list):
            self.mot_files = [mot_files]
        else:
            self.mot_files = mot_files

        if not isinstance(self.mot_files[0], Path):
            self.mot_files = [Path(i) for i in self.mot_files]

        self.main_loop()

    def main_loop(self):
        if self.multi:
            import os
            from multiprocessing import Pool

            indices = list(range(1, len(self.mot_files)))
            pool = Pool(os.cpu_count())
            pool.map(self.run_id_tool, self.mot_files, indices)
        else:
            ct = 0
            for itrial in self.mot_files:
                self.run_id_tool(itrial,ct)
                ct = ct+1

    def run_id_tool(self, trial, index_ftrial):
        print(f'\t{trial.stem}')

        # initialize inverse dynamic tool from setup file
        model = osim.Model(self.model_input)
        id_tool = osim.InverseDynamicsTool(self.xml_input)
        id_tool.setModel(model)

        # get starting and ending time
        motion = osim.Storage(f'{trial.resolve()}')
        start = motion.getFirstTime()
        end = motion.getLastTime()

        # inverse dynamics tool
        id_tool.setStartTime(start)
        id_tool.setEndTime(end)
        id_tool.setCoordinatesFileName(f'{trial.resolve()}')

        if self.low_pass:
            id_tool.setLowpassCutoffFrequency(self.low_pass)

        # set name of input (mot) file and output (sto)
        filename = f'{trial.stem}'
        id_tool.setName(filename)
        id_tool.setOutputGenForceFileName(f"{filename}.sto")
        id_tool.setResultsDir(f'{self.sto_output}')

        # external loads file
        if self.forces_dir:
            loads = osim.ExternalLoads(self.xml_forces, True)
            if self.force_names is None: # assumes equal names or with prefix
                if self.prefix:
                    loads.setDataFileName(
                        f"{Path(self.forces_dir, trial.stem.replace(f'{self.prefix}_', '')).resolve()}.sto"
                    )
                else:
                    loads.setDataFileName(
                        f"{Path(self.forces_dir, trial.stem).resolve()}.sto"
                    )
            else: # assumes different names (setting force_names)
                loads.setDataFileName(str(self.force_names[index_ftrial]))

            temp_xml = Path(os.path.join(self.forces_dir,trial.stem + '_extloads.xml'))
            loads.printToXML(f'{temp_xml.resolve()}')  # temporary xml file
            id_tool.setExternalLoadsFileName(f'{temp_xml}')

        # run inverse dynamics
        id_tool.run()

        # print id tool settings ? [ToDo]

        #if self.forces_dir:
        #    temp_xml.unlink()  # delete temporary xml file
