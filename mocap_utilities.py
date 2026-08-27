import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.stats import pearsonr, linregress
from scipy.signal import butter, lfilter
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import scipy as sp
import re
from typing import Optional, Tuple

# detect heelstrike based on vertical GRF
def detect_heelstrike_toeoff(time, F, treshold, dtOffPlate=None):
    if dtOffPlate is None:
        dtOffPlate = 0.001

    # Get the index of the events
    AnalogRate = 1 / np.mean(np.diff(time))
    IndexOffPlate = np.where(F < treshold)[0]  # indexes with foot off the FP
    IndexEndSwing = np.where(np.diff(IndexOffPlate) > dtOffPlate * AnalogRate)[
        0
    ]  # minimal duration swing phase
    hs = IndexOffPlate[IndexEndSwing]  # minimal duration swing phase
    to = IndexOffPlate[
        IndexEndSwing + 1
    ]  # start swing is when foot leaves the FP for the next time

    # Get the timing of the events
    ths = time[hs]
    tto = time[to]

    return ths, tto, hs, to

# norm cycle to N datapoints
def norm_cycle(time, EventVector, data, *args):
    # size of the input data
    ndim = data.ndim
    if ndim == 1:
        n_samples = data.shape
        nc = 1
    elif ndim == 2:
        n_samples, nc = data.shape

    # treshold on duration between consecutive events
    treshold_dtStride = np.inf
    if args and len(args) >= 1:
        treshold_dtStride = args[0]

    # print warnings
    bool_print = True
    if args and len(args) >= 2:
        bool_print = args[1]

    # nPointsInt
    n_points_int = 100
    if args and len(args) >= 3:
        n_points_int = args[2]

    # select duration stance and swing phase for each step
    n_step = len(EventVector) - 1
    dataV = np.empty((100, nc, n_step)) if nc > 1 else np.empty((n_points_int, n_step))
    dataV.fill(np.nan)

    for i in range(n_step):
        dtStride = EventVector[i + 1] - EventVector[i]
        if dtStride < treshold_dtStride:
            # select indices of the selected stride
            i_sel = (time >= EventVector[i]) & (time <= EventVector[i + 1])
            n_fr = np.sum(i_sel)
            if n_fr > 0:
                # interpolate the data to 100 datapoints
                if nc == 1:
                    data_interp = interp1d(
                        np.linspace(1, n_fr, n_fr), data[i_sel], kind="linear"
                    )
                    dataV[:, i] = data_interp(np.linspace(1, n_fr, n_points_int))
                else:
                    data_interp = interp1d(
                        np.linspace(1, n_fr, n_fr),
                        data[i_sel, :],
                        axis=0,
                        kind="linear",
                    )
                    dataV[:, :, i] = data_interp(np.linspace(1, n_fr, n_points_int))
        else:
            if bool_print:
                print(
                    f"Removed a cycle from the analysis because duration of cycle was {dtStride} s"
                )

    if nc == 1:
        DatMean = np.nanmean(dataV, axis=1)
        DatSTD = np.nanstd(dataV, axis=1)
        DatMedian = np.nanmedian(dataV, axis=1)
    else:
        DatMean = np.nanmean(dataV, axis=2)
        DatSTD = np.nanstd(dataV, axis=2)
        DatMedian = np.nanmedian(dataV, axis=2)

    return dataV, DatMean, DatSTD, DatMedian

# numerical derivative
def central_difference(x, y):
    if y.ndim == 1:
        tmp = np.diff(y) / np.diff(x)
        dy_dx = np.concatenate(([tmp[0]], 0.5 * (tmp[1:] + tmp[:-1]), [tmp[-1]]))

    elif y.ndim == 2:
        nr, nc = y.shape
        BoolTranspose = False
        if nc > nr:
            x = x.T
            y = y.T
            BoolTranspose = True
        # pre allocate output
        nr, nc = y.shape
        dy_dx = np.zeros((nr, nc))
        for i in range(nc):
            tmp = np.diff(y[:, i]) / np.diff(x)
            dy_dx[:, i] = np.concatenate(
                ([tmp[0]], 0.5 * (tmp[1:] + tmp[:-1]), [tmp[-1]])
            )
        if BoolTranspose:
            dy_dx = dy_dx.T
    return dy_dx

# norm phase (betweeon t0 and tend) to N datapoints
def NormPhase(time, t0, tend, data, *args):
    nc = data.shape[1] if len(data.shape) > 1 else 1

    treshold_dtStride = np.inf
    if args and len(args) >= 1:
        treshold_dtStride = args[0]

    BoolNormx0 = False
    if args and len(args) >= 2:
        BoolNormx0 = args[1]

    nPointsInt = 100
    if args and len(args) >= 3:
        nPointsInt = args[2]

    nstep = len(t0)
    if nc == 1:
        dataV = np.empty((nPointsInt, nstep))
    else:
        dataV = np.empty((nPointsInt, nc, nstep))

    for i in range(nstep):
        tendSel = tend[tend > t0[i]]
        if len(tendSel) > 0:
            tendSel = tendSel[0]
            dtStride = tendSel - t0[i]
            if dtStride < treshold_dtStride:
                iSel = (time >= t0[i]) & (time <= tendSel)
                nfr = np.sum(iSel)
                if nfr > 0:
                    if nc == 1:
                        dsel = data[iSel]
                        if BoolNormx0:
                            dsel = dsel - dsel[0]
                        interp_func = interp1d(np.arange(nfr), dsel, kind="linear")
                        dataV[:, i] = interp_func(np.linspace(0, nfr - 1, nPointsInt))
                    else:
                        dsel = data[iSel, :]
                        if BoolNormx0:
                            dsel = dsel - dsel[0, :]
                        interp_func = interp1d(
                            np.arange(nfr), dsel, axis=0, kind="linear"
                        )
                        dataV[:, :, i] = interp_func(
                            np.linspace(0, nfr - 1, nPointsInt)
                        )

    DatMean = np.nanmean(dataV, axis=1)
    DatSTD = np.nanstd(dataV, axis=1)
    DatMedian = np.nanmedian(dataV, axis=1)

    return dataV, DatMean, DatSTD, DatMedian

# currently converted using chatGPT, not verified
# the whole thing with the nansignal is a bit weird, but seems to work
def normalizetimebase_step(t, signal, to, hs):
    # check if toe-off is not after heelstrike (we want swing phase)
    if to[0] > hs[0]:
        to = to[:-1]
        hs = hs[1:]
    # get n cycles
    min_length = min(len(to), len(hs))
    to = to[:min_length]
    hs = hs[:min_length]
    # find nans in signal
    nansignal = ~np.isnan(signal)

    Cycle = np.full((51, min_length), np.nan)
    CycleLength = np.full(min_length, -51)
    N = min_length
    if N > 1:
        for i in range(N):
            isel = (t >= to[i]) & (t <= hs[i])
            x = np.arange(np.sum(isel))
            CycleLength[i] = len(x)
            try:
                interp_func = interp1d(x, signal[isel], kind="cubic")
                Cycle[:, i] = interp_func(np.linspace(0, x[-1], 51))
            except:
                Cycle[:, i] = np.nan # not any number in the signal
        tmp = np.nanmean(Cycle[:, :N], axis=1)
        TimeGain = 51 / np.mean(CycleLength)
    elif N == 1:
        i = 0
        isel = (t >= to[i]) & (t <= hs[i])
        x = np.arange(np.sum(isel))
        CycleLength = len(x)
        try:
            interp_func = interp1d(x, signal[isel], kind="cubic")
            Cycle[:, i] = interp_func(np.linspace(0, x[-1], 51))
        except:
            Cycle[:, i] = np.nan
        TimeGain = 51 / CycleLength
    return Cycle, TimeGain

# currently converted using chatGPT, not verified
# seems to work now, but programming can be improved (also in the original matlab code)
def order_events(events, loc_mode="walk"):
    lhs = events["lhs"]
    rhs = events["rhs"]
    lto = events["lto"]
    rto = events["rto"]

    if loc_mode == "walk":
        event_list = np.concatenate((lhs, rto, rhs, lto))
        lhs_code = np.ones(len(lhs)) * 1
        rto_code = np.ones(len(rto)) * 2
        rhs_code = np.ones(len(rhs)) * 3
        lto_code = np.ones(len(lto)) * 4
        code = np.concatenate((lhs_code, rto_code, rhs_code, lto_code))

        data = np.vstack((code, event_list)).T
        data = data[data[:, 1].argsort()]

        ind_begin = np.argmax(data[:, 0] == 1)
        ind_end = np.max(np.where(data[:, 0] == 4))
        data = data[ind_begin : ind_end + 1]

        lhs_sort = data[data[:, 0] == 1, 1]
        rto_sort= data[data[:, 0] == 2, 1]
        rhs_sort = data[data[:, 0] == 3, 1]
        lto_sort = data[data[:, 0] == 4, 1]
        EventDat = np.vstack([lhs_sort, rto_sort, rhs_sort, lto_sort]).T
        events_out = pd.DataFrame(EventDat,columns=["lhs","rto","rhs","lto"])

        flag = 0
        if not (
            len(events_out["lto"]) == len(events_out["rto"])
            and len(events_out["rhs"]) == len(events_out["rto"])
            and len(events_out["lhs"]) == len(events_out["rto"])
        ):
            flag = 1
        elif (
            np.any((events_out["rto"] - events_out["lhs"]) < 0)
            or np.any((events_out["rhs"] - events_out["rto"]) < 0)
            or np.any((events_out["lto"] - events_out["rhs"]) < 0)
        ):
            flag = 2

    elif loc_mode == "run":
        event_list = np.concatenate((lhs, lto, rhs, rto))
        lhs_code = np.ones(len(lhs)) * 1
        lto_code = np.ones(len(lto)) * 2
        rhs_code = np.ones(len(rhs)) * 3
        rto_code = np.ones(len(rto)) * 4
        code = np.concatenate((lhs_code, lto_code, rhs_code, rto_code))

        data = np.vstack((code, event_list)).T
        data = data[data[:, 1].argsort()]

        ind_begin = np.argmax(data[:, 0] == 1)
        ind_end = np.max(np.where(data[:, 0] == 4))
        data = data[ind_begin : ind_end + 1]

        lhs_sort = data[data[:, 0] == 1, 1]
        rto_sort= data[data[:, 0] == 2, 1]
        rhs_sort = data[data[:, 0] == 3, 1]
        lto_sort = data[data[:, 0] == 4, 1]
        EventDat = np.vstack([lhs_sort, rto_sort, rhs_sort, lto_sort]).T
        events_out = pd.DataFrame(EventDat,columns=["lhs","rto","rhs","lto"])

        flag = 0
        if not (
            len(events_out["lto"]) == len(events_out["rto"])
            and len(events_out["rhs"]) == len(events_out["rto"])
            and len(events_out["lhs"]) == len(events_out["rto"])
        ):
            flag = 1

    return events_out, flag


def nanR2(A, B):
    """
    Calculate the coefficient of determination (R^2), root-mean-square (rms), and mean-absolute (ma) difference
    between corresponding columns of matrices A and B, considering NaN values.

    Parameters:
    A (numpy.ndarray): First input matrix.
    B (numpy.ndarray): Second input matrix.

    Returns:
    R2 (numpy.ndarray): Coefficient of determination (R^2) for each column.
    rms (numpy.ndarray): Root-Mean-Square difference for each column.
    ma (numpy.ndarray): Mean-Absolute difference for each column.
    """
    R2 = np.zeros(A.shape[1])
    rms = np.zeros(A.shape[1])
    ma = np.zeros(A.shape[1])

    for i_col in range(A.shape[1]):
        A_col = A[:, i_col]
        B_col = B[:, i_col]
        valid_indices = np.where(~np.isnan(A_col) & ~np.isnan(B_col))[0]

        if len(valid_indices) > 0:
            corr_coef = np.corrcoef(A_col[valid_indices], B_col[valid_indices])
            R2[i_col] = corr_coef[0, 1] ** 2
            rms[i_col] = np.sqrt(
                np.nanmean((A_col[valid_indices] - B_col[valid_indices]) ** 2)
            )
            ma[i_col] = np.nanmean(np.abs(A_col[valid_indices] - B_col[valid_indices]))

    return R2, rms, ma

def abs_expvar_fp(fp_act, fp_pred):
    # Calculate the absolute explained variance
    abs_expvar1 = np.sum((fp_pred - np.mean(fp_act)) ** 2)
    abs_expvar2 = np.sqrt(np.mean((fp_pred - np.mean(fp_act)) ** 2))

    return abs_expvar1, abs_expvar2

def ButterFilter_Low_NaNs(fs, dat, filter_order=2, cutoff_frequency=6):
    # interpolate data to remove nans (needed before filtering)
    if dat.ndim == 1:
        nfr = len(dat)
        non_nan_indices = np.arange(len(dat))[~np.isnan(dat)]
        nan_indices = np.arange(len(dat))[np.isnan(dat)]
        interp_func = interp1d(
            non_nan_indices,
            dat[non_nan_indices],
            kind="linear",
            fill_value="extrapolate",
        )
        dat_int = interp_func(np.arange(nfr))

    elif dat.ndim == 2:
        nfr, nc = dat.shape
        non_nan_indices = np.arange(len(dat[:, 0]))[~np.isnan(dat[:, 0])]
        nan_indices = np.arange(len(dat[:, 0]))[np.isnan(dat[:, 0])]

        # Create an interpolation function (linear interpolation in this case)
        dat_int = np.zeros((dat.shape))

        for i in range(nc):
            interp_func = interp1d(
                non_nan_indices,
                dat[non_nan_indices, i],
                kind="linear",
                fill_value="extrapolate",
            )
            dat_int[:, i] = interp_func(np.arange(nfr))

    # low pass filter COM information
    # Create a Butterworth low-pass filter
    b, a = butter(filter_order, cutoff_frequency / (fs / 2), btype="low")
    dat_filt = filtfilt(b, a, dat_int.T).T
    if dat.ndim == 1:
        dat_filt[nan_indices] = np.nan
    elif dat.ndim == 2:
        dat_filt[nan_indices,:] = np.nan
    return dat_filt

def PlotBar(x, y, *args):
    # Default properties
    Cs = [0.6, 0.6, 0.6]  # Default color
    mk = 3  # Default marker size
    h = None  # Default figure handle

    # Input color
    if args:
        Cs = args[0]
    if len(args) > 1:
        mk = args[1]
    if len(args) > 2:
        h = args[2]
        plt.figure(h)

    # Plot bar with average of individual datapoints
    plt.bar(x, np.nanmean(y), facecolor=Cs, edgecolor=Cs)

    # Plot individual datapoints on top
    n = len(y)
    xrange = 0.2
    dx = (np.arange(1, n + 1) / n * xrange - 0.5 * xrange) + x
    plt.plot(dx, y, 'o', markerfacecolor=Cs, color=[0.2, 0.2, 0.2], markersize=mk)

def get_freq_spect(emg_dat, emg_time, boolPlot = False, emg_headers = "None"):
    [nfr,ncol] = emg_dat.shape
    fs = 1/np.nanmean(np.diff(emg_time))
    for i in range(0, ncol):
        frequencies = np.fft.rfftfreq(nfr, d=1 / fs)  # Frequency bins
        fft_result = np.abs(np.fft.rfft(emg_dat[:,i]))  # Compute FFT and take the magnitude
        if i == 0:
            freq_out = np.zeros([len(frequencies), ncol])
            fft_out = np.zeros([len(frequencies), ncol])
        freq_out[:,i] = frequencies
        fft_out[:,i] = fft_result

    if boolPlot:
        plt.figure()
        for i in range(0, ncol):
            plt.subplot(4,4,i+1)
            plt.plot(freq_out[:,i], fft_out[:,i])
            if emg_headers == "None":
                plt.title('unkown')
            else:
                plt.title(emg_headers[i])

def filter_emg(emg_dat, emg_time=None, lowcut_bp = 20, highcut_bp = 400, lowcut_lp = 10):
    """Filter raw EMG signals with a bandpass, rectification, and lowpass envelope.

    Parameters
    ----------
    emg_dat : ndarray (n_samples, n_channels) or pandas DataFrame
        Raw EMG data. If a DataFrame is passed the filtered result is returned
        as a DataFrame with the same column names and index. A 'time' column
        (case-insensitive) is passed through unchanged and excluded from filtering.
    emg_time : ndarray (n_samples,), optional
        Time vector in seconds (used to compute the sampling frequency).
        Can be omitted when emg_dat is a DataFrame that contains a 'time' column.
    lowcut_bp : float
        Lower cutoff frequency of the bandpass filter in Hz (default 20).
    highcut_bp : float
        Upper cutoff frequency of the bandpass filter in Hz (default 400).
    lowcut_lp : float
        Cutoff frequency of the lowpass envelope filter in Hz (default 10).

    Returns
    -------
    emg_filt : ndarray or DataFrame (same type as emg_dat)
        Filtered EMG envelope, same shape as emg_dat.
    """
    import pandas as pd
    is_dataframe = isinstance(emg_dat, pd.DataFrame)
    if is_dataframe:
        col_names    = emg_dat.columns
        idx          = emg_dat.index
        # Identify pass-through columns (time vector) so they are not filtered
        passthrough  = [c for c in col_names if str(c).lower() == 'time']
        signal_cols  = [c for c in col_names if str(c).lower() != 'time']
        data_arr     = emg_dat[signal_cols].to_numpy()
        if emg_time is None and 'time' in col_names:
            emg_time = emg_dat['time'].to_numpy()
    else:
        data_arr    = emg_dat
        passthrough = []
        signal_cols = None

    [nfr, ncol] = data_arr.shape
    fs = 1. / np.nanmean(np.diff(emg_time))
    emg_filt = np.zeros(data_arr.shape)
    for i in range(0, ncol):
        data = data_arr[:, i]
        # Bandpass filter
        b_bp, a_bp = butter_bandpass(lowcut_bp, highcut_bp, fs)
        filtered_bp = filtfilt(b_bp, a_bp, data)

        # Take absolute value
        filtered_abs = np.abs(filtered_bp)

        # Lowpass filter
        b_lp, a_lp = butter_lowpass(lowcut_lp, fs)
        filtered_lp = filtfilt(b_lp, a_lp, filtered_abs)
        emg_filt[:, i] = filtered_lp

    if is_dataframe:
        result = pd.DataFrame(emg_filt, index=idx, columns=signal_cols)
        for col in passthrough:
            result.insert(0, col, emg_dat[col].values)
        return result
    return emg_filt

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def detect_peaks(x:np.ndarray, mph:Optional[float]=None, mpd:int=1, threshold:int=0, edge:str='rising',
                 kpsh:bool=False, valley:bool=False, show:bool=False, ax:Optional[plt.Axes]=None):
    """Detect peaks in data based on their amplitude and other features.
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0)
                           & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0)
                           & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(
            np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

# might be better to do this with a toe-marker and a heel-marker seperately
def detect_events_markerbased(heel_marker, toe_marker, sfmarker, tmarker, treadmill_velocity = 0,
                              heel_vel_threshold = 0.05, toe_vel_threshold = 0.5, bool_plot = False,
                              min_duration_stance = 0.2):
    # toe_vel_threshold : threshold on forward velocity toe-marker to detect toe-off
    # heel_vel_threshold: threshold on vertical velocity heel-marker to detect heelstrike

    # detect gait events based on marker coordinates and some conditions statements
    # assumes that x is walking direction (forward positive) and z is vertical (upward positive)

    # detect if frames are on rows or columns
    [nr, nc] = heel_marker.shape
    if nc>nr:
        # transpose input data
        heel_marker = heel_marker.T
        toe_marker = toe_marker.T

    # low pass filter the marker coordinates
    heel_filt = ButterFilter_Low_NaNs(sfmarker, heel_marker, filter_order=2, cutoff_frequency=4)
    toe_filt = ButterFilter_Low_NaNs(sfmarker, toe_marker, filter_order=2, cutoff_frequency=4)

    # identify heelstrike as:
    # - low position of left heel marker
    # - heel marker velocity in walking direction is +/- equal to treadmill speed for the next n ms [half duration assumed stance phase ?]
    # - vertical heel marker velocity is small for the next n ms [half duration assumed stance phase ?]
    # -
    heel_filt_dot = central_difference(tmarker, heel_filt)
    heel_filt_dot[:, 0] = heel_filt_dot[:, 0] + treadmill_velocity
    toe_filt_dot = central_difference(tmarker, toe_filt)
    toe_filt_dot[:, 0] = toe_filt_dot[:, 0] + treadmill_velocity

    # first attempt
    # first find minimal values in vertical heelstrike velocity
    zd = heel_filt_dot[:, 2]

    # peaks in vertical velocity of marker
    max_downward_vel, _ = sp.signal.find_peaks(-zd, distance=0.4 * sfmarker, height=0.5)
    nheelstrikes = len(max_downward_vel)
    # find first time vertical velocity heel marker is above -0.05 m/s
    ct = 0
    i_heelstrikes = np.zeros(nheelstrikes, dtype=int)
    for i in (max_downward_vel):
        potential_strikes = (zd > -heel_vel_threshold) & (tmarker > tmarker[i])
        if any(potential_strikes):
            i_heelstrikes[ct] = np.argmax(potential_strikes)
        ct = ct + 1
    i_heelstrikes = np.delete(i_heelstrikes, range(ct-1, nheelstrikes), axis=0)
    t_heelstrike = tmarker[i_heelstrikes]

    # define toe-off at first point were forward velocity of marker exceeds a threshold again
    #
    ct = 0
    i_toeoff = np.zeros(nheelstrikes, dtype=int)
    xd = toe_filt_dot[:, 0]
    for i in (i_heelstrikes):
        potential_toeoff = (xd > toe_vel_threshold) & (tmarker > tmarker[i]+min_duration_stance)
        if any(potential_toeoff):
            i_toeoff[ct] = np.argmax(potential_toeoff)
            ct = ct+1
    i_toeoff = np.delete(i_toeoff, range(ct - 1, nheelstrikes), axis=0)
    t_toeoff = tmarker[i_toeoff]

    # plot figure if wanted
    if bool_plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(tmarker, heel_filt_dot[:, 2])
        plt.vlines(t_heelstrike,-1,1,'b')
        plt.vlines(t_toeoff,-1,1, 'r')
        plt.ylabel('heel marker vertical velocity')
        plt.xlabel('time [s]')

        plt.subplot(2, 1, 2)
        plt.plot(tmarker, toe_filt_dot[:, 0])
        plt.vlines(t_heelstrike,-1,1, 'b')
        plt.vlines(t_toeoff,-1,1, 'r')
        plt.ylabel('toe marker forward velocity')
        plt.xlabel('time [s]')
        plt.show()




    # return timing of the gait events
    return(t_heelstrike, t_toeoff)