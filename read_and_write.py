
import pandas as pd
import datetime
import re
import numpy as np
from scipy.interpolate import interp1d
import scipy as sp

# read excel file from K5 system (cosmed)
def read_k5(filename):
    # Read with header for named measurement column access
    df = pd.read_excel(filename, sheet_name='Data')
    # Read without header to access metadata embedded in the header row
    # (test date and barometric pressure are stored in the header row of the Excel file)
    df_raw = pd.read_excel(filename, sheet_name='Data', header=None)

    # --- Measurement data ---
    # Row layout (with pandas header): idx 0 = units, idx 1 = blank, idx 2+ = data
    # All 59 measurement columns available in the file
    meas_cols = [
        't', 'Rf', 'VT', 'VE', 'IV', 'VO2', 'VCO2', 'RQ', 'VE/VO2', 'VE/VCO2',
        'VO2/kg', 'METS', 'FeO2', 'FeCO2', 'FiO2', 'FiCO2', 'PeO2', 'PeCO2',
        'Phase', 'Marker', 'Amb. Temp.', 'RH Amb', 'Device Temp.', 'Analyz. Press.',
        'PB', 'EEkc', 'EEh', 'EEm', 'EEtot', 'EEkg', 'PRO', 'Fat', 'CHO',
        'PRO%', 'FAT%', 'CHO%', 'npRQ', 'Long', 'Lat', 'GPS Altitude',
        'Barom_Altitude', 'GPS Speed', 'GPS Pace', 'GPS Dist.', 'LogVE',
        't Rel', 'mark Speed', 'mark Distance', 'Battery', 'Phase time', 'BR',
        'O2 Cost', 'O2 Delay', 'CO2 Delay', 'GPS Heading', 'RH Sample', 'Cadence',
        'Satellites', 'Fixing', 'Satellites SNR',
    ]
    # Only include columns that actually exist in this file
    available_cols = [c for c in meas_cols if c in df.columns]
    df_meas = df[available_cols].iloc[2:].reset_index(drop=True).copy()

    # Convert K5 time column to seconds relative to trial start
    time_raw = df['t'].iloc[2:].reset_index(drop=True)

    def _k5_time_to_seconds(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, datetime.time):
            return (
                value.hour * 3600
                + value.minute * 60
                + value.second
                + value.microsecond / 1e6
            )
        if isinstance(value, datetime.timedelta):
            return value.total_seconds()
        td = pd.to_timedelta(str(value), errors='coerce')
        if pd.isna(td):
            return np.nan
        return td.total_seconds()

    time_seconds = time_raw.apply(_k5_time_to_seconds).to_numpy(dtype=float)
    valid_time = ~np.isnan(time_seconds)
    if not np.any(valid_time):
        raise ValueError("No valid time values found in K5 't' column.")

    start_time = time_seconds[valid_time][0]
    time_relative = time_seconds - start_time
    df_meas['t'] = time_relative
    df_meas['time_s'] = time_relative

    # Compute metabolic power with Brockway equation (result in W)
    VO2 = df['VO2'].iloc[2:].to_numpy()
    VCO2 = df['VCO2'].iloc[2:].to_numpy()
    df_meas['P_metab'] = (16.58 * VO2 + 4.51 * VCO2) / 60  # W

    # --- Subject and test metadata ---
    # Metadata column layout (pandas iloc with header):
    #   col 1  : subject values  (last name, first name, gender, age, height, weight, DOB)
    #   col 4  : test info values (test time, subject type, ..., test type, ..., test duration)
    #   col 7  : ambient/device values (ambient temp, humidity, flowmeter temp, STPD, ..., BSA, BMI, HR max)
    # Test date and barometric pressure sit in the header row → use df_raw (header=None)
    subj_info = {
        # subject demographics
        'last_name':                df.iloc[0, 1],
        'first_name':               df.iloc[1, 1],
        'gender':                   df.iloc[2, 1],
        'age':                      df.iloc[3, 1],
        'height_cm':                df.iloc[4, 1],
        'weight_kg':                df.iloc[5, 1],
        'dob':                      df.iloc[6, 1],
        # test info
        'test_date':                df_raw.iloc[0, 4],   # embedded in header row
        'test_time':                df.iloc[0, 4],
        'subject_type':             df.iloc[1, 4],
        'test_type':                df.iloc[6, 4],
        'test_duration':            df.iloc[8, 4],
        # ambient / device conditions
        'barometric_pressure_mmhg': df_raw.iloc[0, 7],  # embedded in header row
        'ambient_temperature_c':    df.iloc[0, 7],
        'ambient_humidity_pct':     df.iloc[1, 7],
        # derived body metrics
        'bsa_m2':                   df.iloc[8, 7],
        'bmi':                      df.iloc[9, 7],
        'hr_max':                   df.iloc[10, 7],
    }

    return df_meas, subj_info



# read delsys file -- function to read exported .csv files from the Trigno Delsys system
def read_delsys(filename, filetype="default", output_format="dataframe"):
    """Read an exported .csv file from the Trigno Delsys system.

    Parameters
    ----------
    filename : str
        Path to the exported .csv file.
    filetype : str
        'default'  – comma-separated (standard Delsys export).
        'dutch'    – semicolon-separated with comma decimal (Dutch locale export).
    output_format : str
        'dataframe' (default) – return three DataFrames (emg_df, imu_df, loadcell_df),
                                each with a leading 'time' column followed by signal columns.
        'legacy'             – return the original tuple
                                (emg_dat, emg_time, emg_header,
                                 imu_dat, imu_time, imu_header,
                                 loadcell_dat, loadcell_time, loadcell_header,
                                 delsysdat).

    Returns (output_format='dataframe')
    ------------------------------------
    emg_df      : DataFrame – columns ['time', *emg_headers],   EMG signals in mV
    imu_df      : DataFrame – columns ['time', *imu_headers],   ACC signals in G
    loadcell_df : DataFrame – columns ['time', *lc_headers],    load-cell signals in mV

    Returns (output_format='legacy')
    ----------------------------------
    emg_dat, emg_time, emg_header,
    imu_dat, imu_time, imu_header,
    loadcell_dat, loadcell_time, loadcell_header,
    delsysdat
    """
    sep     = ';' if filetype == "dutch" else ','
    decimal = ',' if filetype == "dutch" else '.'

    def _read_row(skip):
        """Read a single header row from the CSV into a plain list."""
        return pd.read_csv(filename, skiprows=skip, nrows=1,
                           header=None, sep=sep).iloc[0].tolist()

    # Row 3 (0-indexed): sensor names; only the first column of each sensor group is
    # populated – the rest are blank / NaN.
    # Row 5: column data types  (e.g. "EMG 1 (mV)", "ACC X Time Series (s)", …)
    sensor_row   = _read_row(3)
    col_type_row = _read_row(5)
    n_cols = len(col_type_row)

    # --- Forward-fill sensor names ------------------------------------------
    # Strip the numeric sensor ID in parentheses, then sanitise to a valid identifier.
    current_sensor = None
    sensor_per_col = []
    for raw in sensor_row:
        s = str(raw).strip()
        if s and s.lower() != 'nan':
            match = re.match(r'^(.*?)\s*\(\d+\)\s*$', s)
            name  = match.group(1).strip() if match else s
            current_sensor = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_')
        sensor_per_col.append(current_sensor)
    # Pad to n_cols in case the sensor-name row is shorter than the data rows
    while len(sensor_per_col) < n_cols:
        sensor_per_col.append(current_sensor)

    # --- Read raw data -------------------------------------------------------
    delsysdat = pd.read_csv(
        filename, skiprows=7, header=None,
        sep=sep, decimal=decimal, low_memory=False,
    )
    # Ensure we don't exceed the actual column count in the data
    n_cols = min(n_cols, delsysdat.shape[1])

    # --- Classify each column by its data type ------------------------------
    def _classify(s):
        s = str(s).strip()   # strip leading/trailing spaces from CSV values
        if 'Time Series' in s:   return 'time'
        if 'EMG'    in s and 'mV' in s: return 'emg'
        if 'Analog' in s and 'mV' in s: return 'loadcell'
        if 'ACC X'  in s:        return 'acc_x'
        if 'ACC Y'  in s:        return 'acc_y'
        if 'ACC Z'  in s:        return 'acc_z'
        return 'other'

    col_classes = [_classify(col_type_row[i]) for i in range(n_cols)]

    # --- Build channel list --------------------------------------------------
    # For every data column find its nearest preceding time column.
    channels = []
    for i in range(n_cols):
        cls = col_classes[i]
        if cls in ('emg', 'loadcell', 'acc_x', 'acc_y', 'acc_z'):
            t_col = next(
                (j for j in range(i - 1, -1, -1) if col_classes[j] == 'time'),
                None,
            )
            channels.append({
                'sensor':   sensor_per_col[i],
                'dtype':    cls,
                'time_col': t_col,
                'data_col': i,
            })

    # --- Helper: extract a single (time, data) pair --------------------------
    def _extract(time_col, data_col):
        t = pd.to_numeric(delsysdat.iloc[:, time_col], errors='coerce').to_numpy()
        d = pd.to_numeric(delsysdat.iloc[:, data_col], errors='coerce').to_numpy()
        mask = ~np.isnan(t)
        return t[mask], d[mask]

    # --- Helper: interpolate a group of channels onto a common time ----------
    def _build_output(channel_list, label_suffix=None):
        if not channel_list:
            return np.array([]), np.array([]), []
        t_ref, _ = _extract(channel_list[0]['time_col'], channel_list[0]['data_col'])
        dat      = np.zeros((len(t_ref), len(channel_list)))
        headers  = []
        for j, ch in enumerate(channel_list):
            t_ch, d_ch = _extract(ch['time_col'], ch['data_col'])
            if len(t_ch) > 1:
                f = interp1d(t_ch, d_ch, bounds_error=False, fill_value=0.0)
                dat[:, j] = f(t_ref)
            name = ch['sensor']
            headers.append(f"{name}_{label_suffix}" if label_suffix else name)
        return dat, t_ref, headers

    # --- Split channels by type and build outputs ----------------------------
    emg_channels = [ch for ch in channels if ch['dtype'] == 'emg']
    lc_channels  = [ch for ch in channels if ch['dtype'] == 'loadcell']
    acc_x_chs    = [ch for ch in channels if ch['dtype'] == 'acc_x']
    acc_y_chs    = [ch for ch in channels if ch['dtype'] == 'acc_y']
    acc_z_chs    = [ch for ch in channels if ch['dtype'] == 'acc_z']

    emg_dat,      emg_time,      emg_header      = _build_output(emg_channels)
    loadcell_dat, loadcell_time, loadcell_header = _build_output(lc_channels)

    # IMU: interleave ACC X / Y / Z columns so output is [X0,Y0,Z0, X1,Y1,Z1, …]
    imu_channels_interleaved = [
        ch
        for triplet in zip(acc_x_chs, acc_y_chs, acc_z_chs)
        for ch in triplet
    ]
    # Any unmatched axes (sensor with only partial ACC data) appended at the end
    n_matched = len(acc_x_chs) if len(acc_x_chs) == len(acc_y_chs) == len(acc_z_chs) else 0
    imu_channels_interleaved += acc_x_chs[n_matched:] + acc_y_chs[n_matched:] + acc_z_chs[n_matched:]

    if imu_channels_interleaved:
        t_ref_imu, _ = _extract(imu_channels_interleaved[0]['time_col'],
                                 imu_channels_interleaved[0]['data_col'])
        imu_dat    = np.zeros((len(t_ref_imu), len(imu_channels_interleaved)))
        imu_header = []
        suffix_map = {'acc_x': 'ACC_X', 'acc_y': 'ACC_Y', 'acc_z': 'ACC_Z'}
        for j, ch in enumerate(imu_channels_interleaved):
            t_ch, d_ch = _extract(ch['time_col'], ch['data_col'])
            if len(t_ch) > 1:
                f = interp1d(t_ch, d_ch, bounds_error=False, fill_value=0.0)
                imu_dat[:, j] = f(t_ref_imu)
            imu_header.append(f"{ch['sensor']}_{suffix_map[ch['dtype']]}")
        imu_time = t_ref_imu
    else:
        imu_dat, imu_time, imu_header = np.array([]), np.array([]), []

    if output_format == "legacy":
        return (emg_dat, emg_time, emg_header,
                imu_dat, imu_time, imu_header,
                loadcell_dat, loadcell_time, loadcell_header,
                delsysdat)

    def _to_df(dat, time, headers):
        if time.size == 0:
            return pd.DataFrame(columns=["time"] + headers)
        df = pd.DataFrame(dat, columns=headers)
        df.insert(0, "time", time)
        return df

    emg_df      = _to_df(emg_dat,      emg_time,      emg_header)
    imu_df      = _to_df(imu_dat,      imu_time,      imu_header)
    loadcell_df = _to_df(loadcell_dat, loadcell_time, loadcell_header)

    return emg_df, imu_df, loadcell_df


def read_fbwm(filename, sampling_frequency=200, headers=None):
    """Read a FBWM `.gai` file into a pandas DataFrame with a time vector.

    Parameters
    ----------
    filename : str
        Path to the tab-delimited FBWM file.
    sampling_frequency : float, default 200
        Sampling frequency in Hz.
    headers : list[str], optional
        Column names for the raw signals. When omitted, the default FBWM
        channel names are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with a leading ``time`` column and the FBWM signals.
    """
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")

    if headers is None:
        headers = [
            "front_motor",
            "trigger",
            "back_motor",
            "front_loadcell",
            "back_loadcell",
        ]

    df = pd.read_csv(filename, delimiter="\t", header=None, names=headers)
    time = np.arange(len(df), dtype=float) / float(sampling_frequency)
    df.insert(0, "time", time)
    return df


def read_qualisys(filename):
    """Read a Qualisys trajectory export in .mat format into a pandas DataFrame.

    The function is intended for QTM exports organised as
    ``qual['trial_0002']['Trajectories']['Labeled']``. The returned DataFrame has
    a ``time`` column, followed by one triple of marker columns per labelled marker,
    e.g. ``LASI_x``, ``LASI_y``, ``LASI_z``.

    Parameters
    ----------
    filename : str
        Path to the exported .mat file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per time frame and marker coordinates in columns.
    """
    qual = readmat(filename)

    # Qualisys exports can contain a single trial structure or multiple trials.
    trial = None
    if isinstance(qual, dict):
        trial_candidates = [key for key in qual if not key.startswith('__')]
        if len(trial_candidates) == 1:
            trial = qual[trial_candidates[0]]
        else:
            for key in trial_candidates:
                value = qual[key]
                if isinstance(value, dict) and 'Trajectories' in value:
                    trial = value
                    break

    if trial is None or not isinstance(trial, dict):
        raise ValueError(f"No Qualisys trial structure found in '{filename}'.")

    trajectories = trial.get('Trajectories', None)
    if trajectories is None or not isinstance(trajectories, dict):
        raise ValueError(f"No 'Trajectories' section found in '{filename}'.")

    labeled = trajectories.get('Labeled', None)
    if labeled is None or not isinstance(labeled, dict):
        raise ValueError(f"No 'Trajectories/Labeled' section found in '{filename}'.")

    labels = np.asarray(labeled.get('Labels', []), dtype=object).ravel()
    data = labeled.get('Data', None)
    if data is None:
        raise ValueError(f"No marker data found under 'Trajectories/Labeled/Data' in '{filename}'.")
    if len(labels) == 0:
        raise ValueError(f"No marker labels found in '{filename}'.")

    arr = np.asarray(data, dtype=float)
    n_markers = len(labels)

    # QTM commonly exports marker data as (n_markers, 4, n_frames) where the
    # first three elements are x/y/z and the fourth indicates valid/invalid.
    # A few exports may instead store frames x markers x 3; support both.
    if arr.ndim != 3:
        raise ValueError(
            "Unexpected Qualisys marker layout: expected a 3D array "
            f"for 'Trajectories/Labeled/Data', got shape {arr.shape}."
        )

    if arr.shape[0] == n_markers and arr.shape[1] in (3, 4):
        coords = arr[:, :3, :].transpose(2, 0, 1)
    elif arr.shape[1] == n_markers and arr.shape[2] in (3, 4):
        coords = arr[:, :, :3]
    elif arr.shape[0] in (3, 4) and arr.shape[1] == n_markers:
        coords = np.transpose(arr[:3, :, :], (2, 1, 0))
    elif arr.shape[0] == n_markers and arr.shape[2] > 1:
        coords = arr[:, :, :3]
    elif arr.shape[1] == n_markers and arr.shape[0] > 1:
        coords = arr[:, :, :3]
    else:
        raise ValueError(
            "Could not interpret the coordinate layout in QTM file. "
            f"Expected marker count {n_markers}, got data shape {arr.shape}."
        )

    n_frames = coords.shape[0]
    time = None

    # Prefer explicit time, which is available in some exports, but not all.
    for candidate in [labeled.get('Time', None), trial.get('Time', None), trajectories.get('Time', None)]:
        if candidate is not None:
            time = np.asarray(candidate, dtype=float).ravel()
            if time.size == n_frames:
                break
            time = None

    if time is None:
        frame_rate = trial.get('FrameRate', None)
        if frame_rate is not None:
            frame_rate = float(np.asarray(frame_rate).reshape(-1)[0])
        if frame_rate is not None and frame_rate > 0:
            start_frame = float(np.asarray(trial.get('StartFrame', 1)).reshape(-1)[0])
            time = (np.arange(n_frames, dtype=float) + start_frame - 1.0) / frame_rate
        else:
            time = np.arange(n_frames, dtype=float)

    time = np.asarray(time, dtype=float)
    if time.size != n_frames:
        time = np.linspace(0.0, n_frames - 1.0, n_frames)

    # Build a DataFrame with one row per frame, columns in the form
    # <markername>_x, <markername>_y, <markername>_z.
    # A 2D matrix is used here instead of a dict-of-columns to keep memory use
    # reasonable for long motion-capture trials.
    seen = {}
    columns = []
    for i, marker_name in enumerate(labels):
        name = str(marker_name).strip()
        if name == '':
            name = f'marker_{i + 1}'
        clean = re.sub(r'[^0-9a-zA-Z_]+', '_', name).strip('_')
        if clean == '':
            clean = f'marker_{i + 1}'
        if clean in seen:
            seen[clean] += 1
            clean = f'{clean}_{seen[clean]}'
        else:
            seen[clean] = 0

        columns.extend([f'{clean}_x', f'{clean}_y', f'{clean}_z'])

    flat = coords.reshape(n_frames, n_markers * 3)
    df = pd.DataFrame(flat, columns=columns)
    df.insert(0, 'time', time)
    return df


def readmat(filename):
    '''
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
       ----------
       Parameters:
       filename: String
               including path and .mat-extension of the datafile you want to load
       Returns :
       data: Array or Dictionary
             data from .mat file, can also be matlab structure which is known
             as dictionary in Python
       -------
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], sp.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _has_struct(elem):
        """Determine if elem is an array and if any array item is a struct"""
        return isinstance(elem, np.ndarray) and any(isinstance(
            e, sp.io.matlab.mio5_params.mat_struct) for e in elem)

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sp.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sp.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = sp.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)