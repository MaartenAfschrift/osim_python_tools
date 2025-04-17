# test some functionality of the osim_subject class

import opensim as osim
from osim_utilities import osim_subject
from kinematic_analyses import lmt_api, moment_arm_api
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# I want to get the path to the main folder (osim_tools_python)
from pathlib import Path

# get current directory
main_dir = Path.cwd().parent
print('main_dir:', main_dir)
datapath = Path.joinpath(main_dir,'data','example_simone')
trc_folder = str(Path.joinpath(datapath,'trc'))
ik_folder = str(Path.joinpath(datapath, 'ik'))
model_path = str(Path.joinpath(datapath, 'scaledmodel_subj1.osim'))
general_ik_settings = str(Path.joinpath(datapath,
                                        'settings',
                                        'ik_settings.xml'))
bk_folder = str(Path.joinpath(datapath, 'bk'))
ik_filt_folder = str(Path.joinpath(datapath, 'ik_filt'))
bk_folder_filt = str(Path.joinpath(datapath, 'bk_filt'))

# create object for processing
subj = osim_subject(model_path)

# set trc files and compute inverse kinematics
# run inverse kinematics
subj.set_trcfiles_fromfolder(trc_folder)
subj.set_general_ik_settings(general_ik_settings)
subj.set_ik_directory(ik_folder)
subj.compute_inverse_kinematics(overwrite=False)

# ik data
ik_dat = subj.ikdat.copy()

# filter ik data
subj.filter_and_save_ik(outfolder = ik_filt_folder,
                        update_ik_directory= True,
                        cutoff=6, order = 4)
ik_dat_filt = subj.ikdat.copy()

# run bodykinematics
subj.set_bodykin_folder(bk_folder_filt)
subj.compute_bodykin(overwrite=False)
bk_dat_filt = subj.bk_vel.copy()

# run bodykinematics without filtering
subj.set_ikfiles_fromfolder(ik_folder) # not filtered ik data
subj.set_bodykin_folder(bk_folder)
subj.compute_bodykin(overwrite=False)
bk_dat = subj.bk_vel.copy()

# test filtered and unfiltered bk solution
#------------------------------------------

plt.figure()
plt.subplot(2,1,1)
plt.plot(bk_dat_filt[0].time, bk_dat_filt[0].center_of_mass_Z, label='filtered')
plt.plot(bk_dat[0].time, bk_dat[0].center_of_mass_Z, label='unfiltered')
plt.legend()
plt.title('Filtered and unfiltered center of mass Z')
plt.xlabel('time (s)')
plt.ylabel('cm Z (m)')

plt.subplot(2,1,2)
plt.plot(ik_dat_filt[0].time, ik_dat_filt[0].ankle_angle_r, label='filtered')
plt.plot(ik_dat[0].time, ik_dat[0].ankle_angle_r, label='unfiltered')
plt.legend()
plt.title('Filtered and unfiltered ankle angle R')
plt.xlabel('time (s)')
plt.ylabel('ankle angle (rad)')

