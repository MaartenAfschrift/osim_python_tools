# test muscle redundancy solver on data of Tim van der Zee

import os
from general_utilities import readMotionFile, WriteMotionFile, lowPassFilterDataFrame
from muscle_redundancy_solver import muscle_redundancy_solver, muscle_redundancy_solver_exo
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# ok dit werkt goed, eens testen of we niet een andere processing pass moeten gebruiken.
# Nu ziet inverse dynamica data er heel heel ruizig uit. Andere optie is om dit te filteren

bool_default_mrs = True
bool_exo_mrs = True

# to do: add controller die % inverse dynamica assistentie geeft

# path to opensim model
gen_osim_model = 'C:/Users/mat950/Documents/Software/general_tools/python_toolkit/osim_tools_python/data/subject1.osim'

datapath = 'C:/Users/mat950/OneDrive - Vrije Universiteit Amsterdam/Onderzoek/Data/VanDerZee'
osim_model_path = os.path.join(datapath,'osim_models','p1.osim')
ikfile_csv = os.path.join(datapath,'dynamics_processingpass','subject_1_20_IK_all.csv')
idfile_csv = os.path.join(datapath,'dynamics_processingpass','subject_1_20_ID_all.csv')
time_csv = os.path.join(datapath,'dynamics_processingpass','subject_1_20_time_all.csv')
ikfile_mot = os.path.join(datapath,'dynamics_processingpass','subject_1_20_IK_all.mot')
idfile_mot = os.path.join(datapath,'dynamics_processingpass','subject_1_20_ID_all.sto')

# conver csv files to mot and sto files
data_ik = pd.read_csv(ikfile_csv)
data_ik_np = data_ik.to_numpy()[:,1:]
indices_rot = np.r_[0:3, 6:23]
data_ik_np[:, indices_rot] = data_ik_np[:, indices_rot]*180/np.pi
ik_headers = data_ik.columns[1:].to_list()
ik_headers.insert(0, 'time')
time_np = pd.read_csv(time_csv).to_numpy()[:,1]
time_np = time_np.reshape((-1,1))
all_ikdat = np.concatenate((time_np, data_ik_np), axis=1)
WriteMotionFile(all_ikdat, ik_headers,ikfile_mot)

data_id = pd.read_csv(idfile_csv)
data_id_np = data_id.to_numpy()[:,1:]
id_headers = data_id.columns[1:].to_list()
id_headers.insert(0, 'time')
all_iddat = np.concatenate((time_np, data_id_np), axis=1)
WriteMotionFile(all_iddat, id_headers,idfile_mot)

# create muscle redundancy solver object and solve
dofs = ["ankle_angle_r"]
muscles = ['soleus_r','tib_ant_r']

if bool_default_mrs:
    my_mrs = muscle_redundancy_solver(gen_osim_model, ikfile_mot,
                                      idfile_mot, dofs,muscles)
    my_mrs.set_tendon_stiffness('soleus_r', 30)
    my_mrs.filter_inputs(cutoff_frequency=6)
    # test formulate and solve ocp
    my_mrs.formulate_solve_ocp(dt = 0.01, tstart = 2,tend = 8)
    my_mrs.default_plot()
    my_mrs.plot_static_opt_results()

if bool_exo_mrs:
    # initi exoskeleton mrs object
    my_mrs_exo = muscle_redundancy_solver_exo(gen_osim_model, ikfile_mot,
                                          idfile_mot, dofs,muscles)
    # set achilles tendon stiffness
    my_mrs_exo.set_tendon_stiffness('soleus_r', 30)
    #my_mrs_exo.set_controller_type('percentage_id')
    my_mrs_exo.set_controller_type('percentage_id_shortening')
    my_mrs_exo.set_dofs_acutated(dofs)
    my_mrs_exo.set_percentage_id_assistance(0.5)
    # filter inputs
    my_mrs_exo.filter_inputs(cutoff_frequency=6)
    # test formulate and solve ocp
    my_mrs_exo.formulate_solve_ocp(dt = 0.01, tstart = 2,tend = 8)
    my_mrs_exo.default_plot()
    my_mrs_exo.plot_exo_support()

    # test with higher assistance level
    #my_mrs_exo.set_percentage_id_assistance(0.7)
    #my_mrs_exo.formulate_solve_ocp(dt=0.01, tstart=2, tend=8)
    #my_mrs_exo.default_plot()
    #my_mrs_exo.plot_exo_support()




plt.show()

