# test batch processing
#-----------------------

# import utilities
from osim_utilities import osim_subject
import os
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import pandas as pd
import opensim as osim
from inverse_dynamics import InverseDynamics
from inverse_kinematics import InverseKinematics
from kinematic_analyses import bodykinematics
from scipy import signal
import scipy.interpolate as interpolate

# matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# test with simone her dataset
# mainpath = 'D:\Data\Berkelmans'
mainpath = "/Volumes/SIMONE/Promoveren/VU_PerturbationExperiments_2024/Data_MetingenApril2024"
datapath = os.path.join(mainpath,'SUB01')
#model_path = os.path.join(datapath, 'osim','scaled_model.osim')
model_path = os.path.join(datapath,
                          'osim',
                          'scaledmodel_subj1.osim')


# some general settings
general_ik_settings = os.path.join(mainpath,
                                   'SUB01',
                                   'osim',
                                   'iksettings_subj1.xml')

# create object for processing
subj = osim_subject(model_path)

# condition names to iterate over
condition_names = ['fast_skatebelt', 'fast_small', 'slow_skatebelt', 'slow_small']
# condition_name = 'slow_skatebelt'

for condition_name in condition_names:
    trc_folder = os.path.join(datapath, 'trc', condition_name)
    ik_folder = os.path.join(datapath, 'osim', 'ik', condition_name)

    # run inverse kinematics
    subj.set_trcfiles_fromfolder(trc_folder)
    subj.set_general_ik_settings(general_ik_settings)
    subj.set_ik_directory(ik_folder)
    subj.compute_inverse_kinematics(overwrite=False)

    # run bodykinematics
    bk_folder = os.path.join(datapath, 'osim', 'bk', condition_name)
    subj.set_bodykin_folder(bk_folder)
    subj.compute_bodykin(overwrite=False)

    # example if you want to save all info of this subject
    # note that this is just convenient but not necessary as
    # ik and bodykinematics are also saved as .mot and .sto files
    picklefolder = os.path.join(datapath, 'osim')
    if not (os.path.isdir(picklefolder)):
        os.mkdir(picklefolder)
    subj_save_file = os.path.join(picklefolder, condition_name + '_subj.pkl')
    with open(subj_save_file, 'wb') as file:
        pickle.dump(subj, file)

##
# trc_folder = os.path.join(datapath, 'trc', condition_name )
# ik_folder = os.path.join(datapath,'osim','ik', condition_name)

# some general settings
# general_ik_settings = os.path.join(mainpath,
#                                   'SUBXX',
#                                   'osim',
#                                   'iksettings_subjXX.xml')
# create object for processing
#subj = osim_subject(model_path)

# run inverse kinematics
#subj.set_trcfiles_fromfolder(trc_folder)
#subj.set_general_ik_settings(general_ik_settings)
#subj.set_ik_directory(ik_folder)
#subj.compute_inverse_kinematics(overwrite = False)

# run bodykinematics
# bk_folder = os.path.join(datapath, 'osim','bk',condition_name)
#subj.set_bodykin_folder(bk_folder)
#subj.compute_bodykin(overwrite = False)

# example if you want to save all info of this subject
# note that this is just convenient but not necessary as
# ik and bodykinematics are also saved as .mot and .sto files
#picklefolder = os.path.join(datapath,'osim')
#if not (os.path.isdir(picklefolder)):
#    os.mkdir(picklefolder)
#subj_save_file = os.path.join(picklefolder ,condition_name + '_subj.pkl')
#with open(subj_save_file, 'wb') as file:
#    pickle.dump(subj, file)