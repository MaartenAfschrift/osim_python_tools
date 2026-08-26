
import pandas as pd
import datetime

# read excel file from K5 system (cosmed)
def read_k5(filename):
    df = pd.read_excel(filename, sheet_name='Data')
    # extract time vector, VO2 and VCO2 data
    VCO2 = (df['VCO2'][2:-1]).to_numpy()  # in ml/min
    VO2 = (df['VO2'][2:-1]).to_numpy()  # in ml/min
    time = (df['t'][2:-1]).to_numpy()  # in ml/min
    RQ = (df['RQ'][2:-1]).to_numpy()  # respiratory quotient

    # convert time to seconds
    time_vals = [t for t in time if isinstance(t, datetime.time)]
    time_seconds = [t.hour * 3600 + t.minute * 60 + t.second for t in time_vals]
    start_time = time_seconds[0]
    time = [t - start_time for t in time_seconds]

    # compute metabolic energy with brockways equation
    P_metab = 16.58 * VO2 + 4.51 * VCO2  # this gives metabolic power in J/min ?
    P_metab = P_metab / 60  # in W

    # return as a pandas dataframe
    df_metab = pd.DataFrame({'time': time, 'VO2': VO2, 'VCO2': VCO2, 'RQ': RQ, 'P_metab': P_metab})

    # subject information
    age = df.iloc[3,1]
    height = df.iloc[4,1]
    weight = df.iloc[5, 1]
    subj_info = {'age': age, 'height': height, 'weight': weight}

    return df_metab, subj_info



# read delsys file -- function to read exported .csv files from the Trigno delsys system at the VU
def read_delsys(filename, filetype = "default"):
    # input arguments
    # filename = path to the exported .csv file
    # filename = "default", comma separated file
    #             "dutch" = ; separated file (delsys exports this type when you language setting is dutch)

    # read the headers
    if filetype == "default":
        delsysheader_labels = pd.read_csv(filename, skiprows=5, nrows=0)
        delsysheader_sensorid = pd.read_csv(filename, skiprows=3, nrows=0, na_values=' ')
    elif filetype == "dutch":
        delsysheader_labels = pd.read_csv(filename, skiprows=5, nrows=0, sep=';')
        delsysheader_sensorid = pd.read_csv(filename, skiprows=3, nrows=0, na_values=' ', sep=';')
    else:
        # assuming comma separated file
        delsysheader_labels = pd.read_csv(filename, skiprows=5, nrows=0)
        delsysheader_sensorid = pd.read_csv(filename, skiprows=3, nrows=0, na_values=' ')

    # we want to unpack this sensorid array
    cols = delsysheader_sensorid.columns
    ct = 0
    headers = []
    iTime = []
    for i in range(0, len(delsysheader_labels.columns)):
        # intentional header if len colncolname = {str} ' LoadCell_links (76998)'ame is above 5
        if (i < len(cols)):
            colname = cols[i]
        else:
            colname = ' '
        if (len(colname) > 5):
            pattern = r'[a-zA-Z_]+'
            colname_edit = re.findall(pattern, colname)
            colname_out = '_'.join(colname_edit)
            headers.append(colname_out + '_time')
            lastheader = colname_out
            iTime.append(ct)
        else:
            pattern = r'[a-zA-Z_]+'
            colname_edit = re.findall(pattern, delsysheader_labels.columns[ct])
            colname_out = '_'.join(colname_edit)
            if 'Time_Series' in colname_out:
                iTime.append(ct)
            headers.append(lastheader + '_' + colname_out)
        # update counter
        ct = ct + 1

    # read the delsys file with our headers
    if filetype == "default":
        delsysdat = pd.read_csv(filename, skiprows=7, na_values=' ', header=None, names=headers)
    else:
        delsysdat = pd.read_csv(filename, skiprows=7, na_values=' ', header=None, names=headers, sep=';',
                                decimal=',')

    # unpack the delsys information
    [emg_dat, emg_time, emg_header] = select_data_delsys(delsysdat, 'EMG_mV')
    [imu_dat, imu_time, imu_header] = select_data_delsys(delsysdat, '_ACC_')
    [loadcell_dat, loadcell_time, loadcell_header] = select_data_delsys(delsysdat, 'LoadCell_')

    # return output
    return (emg_dat, emg_time, emg_header, imu_dat, imu_time, imu_header,
            loadcell_dat, loadcell_time, loadcell_header, delsysdat)


def select_data_delsys(delsysdat, headerinfo):
    # we have to adapt some things here
    # delsys might export data without time information
    # and data can even contain different sampling frequencies
    # so we have to update code


    cols = delsysdat.columns
    boolfirst = True
    ctCol = 0
    headersout = []
    for i in range(0, len(cols)):
        colsel = cols[i]
        if (headerinfo in colsel) and not ('time' in colsel) and not ('Time' in colsel):
            # this is an emg sensor
            ctCol = ctCol + 1
            headersout.append(colsel)
            if boolfirst == True:  # first time vector is the one we will use for all signals
                tRef = delsysdat.iloc[:, i - 1].to_numpy()
                tRef = tRef[~np.isnan(tRef)]
                boolfirst = False

    if boolfirst == False:  # only sort data when we found at least 1 column with input string as header
        boolfirst = True
        dat = np.zeros([len(tRef), ctCol])
        ctCol = -1
        for i in range(0, len(cols)):
            colsel = cols[i]
            if (headerinfo in colsel) and not ('time' in colsel) and not ('Time' in colsel):
                # this is an emg sensor
                ctCol = ctCol + 1
                if boolfirst == True:
                    tRef = delsysdat.iloc[:, i - 1].to_numpy()
                    tRef = tRef[~np.isnan(tRef)]
                    boolfirst = False
                    if (tRef[0] > 0.02 or tRef[0] < 0):
                        print('error with reference time information')

                datsel = delsysdat.iloc[:, i].to_numpy()
                tsel = delsysdat.iloc[:, i - 1].to_numpy()
                datsel = datsel[~np.isnan(tsel)]
                tsel = tsel[~np.isnan(tsel)]
                dat_intrp = interp1d(tsel, datsel, fill_value=0, bounds_error=False)
                try:
                    datsel_int = dat_intrp(tRef)
                except:
                    print('interpolation error')
                    datsel_int = np.nan
                try:
                    dat[:, ctCol] = datsel_int
                except:
                    dat[:, ctCol] = 0
                    print('error')

    else:
        # no columns found with input string as header
        dat = []
        tRef = []

    return (dat, tRef, headersout)