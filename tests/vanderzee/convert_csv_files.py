# convert csv files vanderzee
#---------------------------
import os
import pandas as pd
import numpy as np
from general_utilities import readMotionFile, WriteMotionFile, lowPassFilterDataFrame


datapath = 'C:/Users/mat950/OneDrive - Vrije Universiteit Amsterdam/Onderzoek/Data/VanDerZee'
osim_model_path = os.path.join(datapath,'osim_models','p1.osim')
nsubj = 10
speeds = [7,9,11,13,14,16,18,20]
folder_sel = 'kinematic_processing_pass'
for pp in range(10):
    for speed in speeds:
        ppi = pp + 1
        ikfile_csv = os.path.join(datapath, folder_sel,
                                  "".join(['subject_', str(ppi), '_', str(speed), '_IK_all.csv']))
        ikfile_csv = os.path.join(datapath,folder_sel,
                                  "".join(['subject_' ,str(ppi), '_', str(speed) ,'_IK_all.csv']))
        idfile_csv = os.path.join(datapath,folder_sel,
                                  "".join(['subject_', str(ppi), '_', str(speed) ,'_ID_all.csv']))
        time_csv = os.path.join(datapath,folder_sel,
                                "".join(['subject_', str(ppi), '_', str(speed) ,'_time_all.csv']))
        ikfile_mot = os.path.join(datapath,folder_sel,
                                  "".join(['subject_', str(ppi), '_', str(speed) ,'_IK_all.mot']))
        idfile_mot = os.path.join(datapath,folder_sel,
                                  "".join(['subject_', str(ppi), '_', str(speed) ,'_ID_all.sto']))

        if (os.path.exists(ikfile_csv) and os.path.exists(idfile_csv) and os.path.exists(time_csv) and
                not(os.path.exists(ikfile_mot)) and not(os.path.exists(idfile_mot))):
            # convert csv files to mot and sto files
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


