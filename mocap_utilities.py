import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.stats import pearsonr, linregress
from scipy.signal import butter, lfilter
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import scipy as sp
import re

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


# class to create data table, fit regression model and visualise datafit
class RegressionModel:
    def __init__(self, nrows):
        # init the data matrices
        self.x1 = np.empty(nrows)
        self.x1.fill(np.nan)
        self.x2 = np.empty(nrows)
        self.x2.fill(np.nan)
        self.y = np.empty(nrows)
        self.y.fill(np.nan)
        self.rowcounter = 0
        self.gains = []
        self.intercept = []

        # identifier (id) for datapoint. you can use this to select specific datapoints in regression analysis
        self.id = np.empty(nrows)
        self.id.fill(np.nan)
        self.indexsel = np.zeros(nrows, dtype=bool)

        # model fit information
        self.model = LinearRegression()
        self.r2 = np.nan
        self.y_predicted = np.empty(nrows)

    def adddatapoint(self, x1, x2, y, id):
        self.x1[self.rowcounter] = x1
        self.x2[self.rowcounter] = x2
        self.y[self.rowcounter] = y
        self.id[self.rowcounter] = id
        self.rowcounter = self.rowcounter + 1

    def getindexmatchid(self, idsel):
        # get indices with matching ids
        indexsel = np.zeros(len(self.y), dtype=bool)
        if isinstance(idsel, int):
            indexsel = self.id == idsel
        else:
            for i in idsel:
                isel = self.id == i
                if np.any(isel):
                    indexsel[isel] = True
        return indexsel

    def fitmodel(self, idsel, boolplot=False, colors=[(0.6, 0.6, 0.6)],
                 xlabel1 = "COM position [m]", xlabel2 = "COM velocity [m/s]",
                 ylabel = "Ankle moment [Nm]", min_samples = 10):
        # get indices with matching ids
        self.indexsel = self.getindexmatchid(idsel)
        if np.any(self.indexsel) and np.sum(self.indexsel)>min_samples:
            # fit the linear model
            X = np.column_stack((self.x1[self.indexsel], self.x2[self.indexsel]))
            self.model.fit(X, self.y[self.indexsel])
            self.y_predicted = self.model.predict(X)
            self.r2 = r2_score(self.y[self.indexsel], self.y_predicted)
            self.gains = self.model.coef_
            self.intercept = self.model.intercept_
            # plot model if wanted
            if boolplot:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
                if isinstance(idsel, int):
                    ax1.scatter(
                        self.x1[self.indexsel],
                        self.y[self.indexsel],
                        marker="o",
                        color=colors,
                    )
                    ax1.set_xlabel(xlabel1)
                    ax1.set_ylabel(ylabel)
                    ax2.scatter(
                        self.x2[self.indexsel],
                        self.y[self.indexsel],
                        marker="o",
                        color=colors,
                    )
                    ax2.set_xlabel(xlabel2)
                    ax2.set_title("ID-end " + str(idsel))
                    ax3.scatter(
                        self.y_predicted,
                        self.y[self.indexsel],
                        marker="o",
                        color=colors,
                    )
                    ax3.set_xlabel("Regression [Nm]")
                    yMin = np.min([np.min(self.y_predicted), np.min(self.y[self.indexsel])])
                    yMax = np.max([np.max(self.y_predicted), np.max(self.y[self.indexsel])])
                    ax3.plot([yMin,yMax],[yMin,yMax], color=(0.1, 0.1, 0.1), linewidth=1)
                    ax3.set_title("Rsq " + str(np.round(self.r2, 2)))
                else:
                    ctColor = 0
                    for i in idsel:
                        isel = self.id == i
                        if np.any(isel):
                            ax1.scatter(
                                self.x1[isel],
                                self.y[isel],
                                marker="o",
                                color=colors[ctColor],
                            )
                            ax1.set_xlabel(xlabel1)
                            ax1.set_ylabel(ylabel)
                            ax2.scatter(
                                self.x2[isel],
                                self.y[isel],
                                marker="o",
                                color=colors[ctColor],
                            )
                            ax2.set_xlabel(xlabel2)
                            if len(idsel) < 5:
                                ax2.set_title("ID-end " + str(idsel))
                            # predict values with current inputs
                            X = np.column_stack((self.x1[isel], self.x2[isel]))
                            y_pred = self.model.predict(X)
                            ax3.scatter(
                                y_pred, self.y[isel], marker="o", color=colors[ctColor]
                            )
                            ax3.set_xlabel("Regression [Nm]")
                            ax3.set_title("Rsq " + str(np.round(self.r2, 2)))
                        if len(colors) > 1:
                            ctColor = ctColor + 1

                plt.tight_layout()
                # plt.show()

    def writetocsv(self, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # write the header
            header = ["com", "comd", "T", "id"]
            writer.writerow(header)

            # write the data
            for i in range(self.rowcounter - 1):
                data = [self.x1[i], self.x2[i], self.y[i], self.id[i]]
                writer.writerow(data)

    def importfromcsv(self, filename):
        # read the data file and update variables
        df = pd.read_csv(filename)
        self.x1 = np.array(df.com)
        self.x2 = np.array(df.comd)
        self.y = np.array(df["T"])
        self.id = np.array(df.id)
        self.rowcounter = len(self.x1)

        # still have to debug this one (from chapgpt :))


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


# function to run regression between COM state and foot placement
def FootPlacement_COMstate(
    t,
    COM,
    FootL,
    FootR,
    Event,
    centerdata=0,
    removeorigin=True,
    pred_samples=range(0, 50),
    order=2,
    boolPlot=True,
    cond="SteadyState_normal",
):
    # input arguments:

    # numerical derivative of COM motion
    COMd = central_difference(t, COM)
    COMdd = central_difference(t, COMd)

    # order the gait events
    [events, flag] = order_events(Event, "walk")

    # check for errors in order gait events (currently no method to solve errors automatically)
    if flag != 0:
        raise ValueError("There is something wrong with your events, returning")

    # convert gait events to numpy array
    lhs = np.array(events["lhs"])
    rhs = np.array(events["rhs"])
    lto = np.array(events["lto"])
    rto = np.array(events["rto"])

    # method used by Moira van Leeuwen to exclude outliers ?
    st_L = np.diff(lhs)
    st_R = np.diff(rhs)
    ignore_L = np.where(
        (st_L > (np.median(st_L) + 0.3 * np.median(st_L)))
        | (st_L < (np.median(st_L) - 0.3 * np.median(st_L)))
    )
    ignore_R = np.where(
        (st_R > (np.median(st_R) + 0.3 * np.median(st_R)))
        | (st_R < (np.median(st_R) - 0.3 * np.median(st_R)))
    )
    st_L = np.delete(st_L, ignore_L)

    # normalise data between heelstrike and toe-off (to 51 datapoints)
    # you have to read this part carefully (not very intuitive).
    # for the left you have to read this as follows.
    #   - COM_L information contains COM motion during right single stance phase
    #   - foot_L contains left foot position during the first left stance phase
    #   (after the right single stance phase). We use this to determine the foot
    #   placement location
    #   - origin_L contains the locatin of the right foot during the right single
    #   stance phase (this is the origin of the coordinate system if you set the flag
    #   remove origin)
    COM_L, _ = normalizetimebase_step(t, COM, lto[:-1], lhs[1:])
    COM_L_vel, _ = normalizetimebase_step(t, COMd, lto[:-1], lhs[1:])
    COM_L_acc, _ = normalizetimebase_step(t, COMdd, lto[:-1], lhs[1:])
    foot_L, _ = normalizetimebase_step(t, FootL, rto[1:], rhs[1:])
    origin_L, _ = normalizetimebase_step(t, FootR, lto[:-1], lhs[1:])

    COM_R, _ = normalizetimebase_step(t, COM, rto[:-1], rhs[:-1])
    COM_R_vel, _ = normalizetimebase_step(t, COMd, rto[:-1], rhs[:-1])
    COM_R_acc, _ = normalizetimebase_step(t, COMdd, rto[:-1], rhs[:-1])
    foot_R, _ = normalizetimebase_step(t, FootR, lto[:-1], lhs[1:])
    origin_R, _ = normalizetimebase_step(t, FootL, rto[:-1], rhs[:-1])

    # remove the strides that are too short or too long
    COM_L[:, ignore_L] = []
    COM_L_vel[:, ignore_L] = []
    COM_L_acc[:, ignore_L] = []
    foot_L[:, ignore_L] = []
    origin_L[:, ignore_L] = []

    COM_R[:, ignore_R] = []
    COM_R_vel[:, ignore_R] = []
    COM_R_acc[:, ignore_R] = []
    foot_R[:, ignore_R] = []
    origin_R[:, ignore_R] = []

    # Extract data during midstance to set origin and to get coordinate of foot placement
    foot_L = foot_L[24, :]
    origin_L = origin_L[24, :]
    foot_R = foot_R[24, :]
    origin_R = origin_R[24, :]
    foot_L = foot_L.T
    origin_L = origin_L.T

    if removeorigin:
        foot_L = foot_L - origin_L  # Subtract origin, which is the other foot
        foot_R = foot_R - origin_R

    if centerdata:
        foot_L = foot_L - np.nanmean(
            foot_L
        )  # use deviations from average foot placement location
        foot_R = foot_R - np.nanmean(
            foot_R
        )  # use deviations form average foot placement location

    # pre allocate Jacobian matrices and statistics
    nPred = len(pred_samples)
    Rsq_vect = np.full((nPred, 1), np.nan)
    Coeff_vect = np.full((nPred, order), np.nan)
    Intervect_vect = np.full((nPred, 1), np.nan)
    Rsq_vect_R = np.full((nPred, 1), np.nan)
    Coeff_vect_R = np.full((nPred, order), np.nan)
    Intervect_vect_R = np.full((nPred, 1), np.nan)
    Rsq_vect_L = np.full((nPred, 1), np.nan)
    Coeff_vect_L = np.full((nPred, order), np.nan)
    Intervect_vect_L = np.full((nPred, 1), np.nan)
    # L_jac = np.zeros((nPred, order))
    # R_jac = np.zeros((nPred, order))
    cti = 0

    # loop over all pred_samples (i.e. predict foot placement based on COM information at pred_sample i)
    for i_pred_sample in pred_samples:
        COM_L_sample = COM_L[i_pred_sample,:].T
        COM_L_vel_sample = COM_L_vel[i_pred_sample,:].T
        COM_L_acc_sample = COM_L_acc[i_pred_sample,:].T

        COM_R_sample = COM_R[i_pred_sample,:].T
        COM_R_vel_sample = COM_R_vel[i_pred_sample,:].T
        COM_R_acc_sample = COM_R_acc[i_pred_sample,:].T

        # predictors
        pred_Lstance = np.column_stack(
            (COM_L_sample, COM_L_vel_sample, COM_L_acc_sample)
        )
        pred_Rstance = np.column_stack(
            (COM_R_sample, COM_R_vel_sample, COM_R_acc_sample)
        )
        # remove origin and means.
        if removeorigin:
            pred_Lstance[:, 0] = pred_Lstance[:, 0] - origin_L
            pred_Rstance[:, 0] = pred_Rstance[:, 0] - origin_R

        if centerdata:
            pred_Lstance = pred_Lstance - np.tile(
                np.nanmean(pred_Lstance, axis=0), (pred_Lstance.shape[0], 1) # controleren of dit juist is TODO
            )
            pred_Rstance = pred_Rstance - np.tile(
                np.nanmean(pred_Rstance, axis=0), (pred_Rstance.shape[0], 1)
            )

        # linear regression for right leg
        foot_R_sample = foot_R
        tmp = np.column_stack((foot_R_sample, pred_Rstance))
        ind_R = np.array(range(tmp.shape[0]))
        nan_mask = np.isnan(np.sum(tmp, axis=1))
        foot_R_sample = foot_R_sample[~nan_mask]
        pred_Rstance = pred_Rstance[~nan_mask, :]
        ind_R = ind_R[~nan_mask]

        X = pred_Rstance[:, :order]
        model = LinearRegression()
        if np.any(ind_R):
            model.fit(X, foot_R_sample)
            y_pred = model.predict(X)
            Rsq_vect_R[cti] = r2_score(foot_R_sample, y_pred)
            Coeff_vect_R[cti, :] = model.coef_
            Intervect_vect_R[cti] = model.intercept_

        else:
            Rsq_vect_R[cti] = np.nan
            Coeff_vect_R[cti, :] = np.nan
            Intervect_vect_R[cti] = np.nan

        # linear regression for left leg
        foot_L_sample = foot_L
        tmp = np.column_stack((foot_L_sample, pred_Lstance))
        ind_L = np.array(range(tmp.shape[0]))
        nan_mask = np.isnan(np.sum(tmp, axis=1))
        foot_L_sample = foot_L_sample[~nan_mask]
        pred_Lstance = pred_Lstance[~nan_mask, :]
        ind_L = ind_L[~nan_mask]

        X = pred_Lstance[:, :order]
        model = LinearRegression()
        if np.any(ind_L):
            model.fit(X, foot_L_sample)
            y_pred = model.predict(X)
            Rsq_vect_L[cti] = r2_score(foot_L_sample, y_pred)
            Coeff_vect_L[cti, :] = model.coef_
            Intervect_vect_L[cti] = model.intercept_

        else:
            Rsq_vect_L[cti] = np.nan
            Coeff_vect_L[cti, :] = np.nan
            Intervect_vect_L[cti] = np.nan

        # linear regression for left and right leg combined
        foot_sample = np.concatenate((foot_L, foot_R))
        pred_stance = np.concatenate((pred_Lstance[:, 0:order], pred_Rstance[:, 0:order]))
        tmp = np.column_stack((foot_sample, pred_stance))
        ind_LR = np.array(range(tmp.shape[0]))
        nan_mask = np.isnan(np.sum(tmp, axis=1))
        foot_sample = foot_sample[~nan_mask]
        pred_stance = pred_stance[~nan_mask, :]
        ind_LR = ind_LR[~nan_mask]

        X = pred_stance
        model = LinearRegression()
        if np.any(ind_LR):
            model.fit(X, foot_sample)
            y_pred = model.predict(X)
            Rsq_vect[cti] = r2_score(foot_sample, y_pred)
            Coeff_vect[cti, :] = model.coef_
            Intervect_vect[cti] = model.intercept_

        else:
            Rsq_vect[cti] = np.nan
            Coeff_vect[cti, :] = np.nan
            Intervect_vect[cti] = np.nan
        cti = cti + 1

    if boolPlot:
        plt.plot(pred_samples, Rsq_vect)

    # Collect outputs - stride time
    OUT = {}
    OUT["stride_time"] = {}
    OUT["stride_time"]["data"] = np.nanmean(st_L)
    OUT["stride_time_var"] = {}
    OUT["stride_time_var"]["data"] = np.nanstd(st_L)
    OUT["stride_time"]["titel"] = "Stride time"
    OUT["stride_time"]["ylabel"] = "Stride time [s]"
    OUT["stride_time_var"]["titel"] = "Stride time variability"
    OUT["stride_time_var"]["ylabel"] = "Stride time variability [s]"

    # Collect outputs - COM information
    OUT["COM_var"] = {}
    OUT["COM_var"]["data"] = np.nanstd(COM_L, axis=1)
    OUT["COM_vel_var"] = {}
    OUT["COM_vel_var"]["data"] = np.nanstd(COM_L_vel, axis=1)
    OUT["COM_acc_var"] = {}
    OUT["COM_acc_var"]["data"] = np.nanstd(COM_L_acc, axis=1)

    OUT["COM_var"]["ylabel"] = "CoM variability [m]"
    OUT["COM_vel_var"]["ylabel"] = "CoM velocity variability [m/s]"
    OUT["COM_acc_var"]["ylabel"] = "CoM acceleration variability [m/s^2]"

    OUT["COM_var"]["titel"] = "CoM variability"
    OUT["COM_vel_var"]["titel"] = "CoM velocity variability"
    OUT["COM_acc_var"]["titel"] = "CoM acceleration variability"

    # Calculate nanmean for COM, COM_vel, COM_acc
    OUT["COM"] = {}
    OUT["COM"]["data"] = np.nanmean(COM_L, axis=1)
    OUT["COM_vel"] = {}
    OUT["COM_vel"]["data"] = np.nanmean(COM_L_vel, axis=1)
    OUT["COM_acc"] = {}
    OUT["COM_acc"]["data"] = np.nanmean(COM_L_acc, axis=1)

    OUT["COM"]["ylabel"] = "CoM position [m]"
    OUT["COM_vel"]["ylabel"] = "CoM velocity [m/s]"
    OUT["COM_acc"]["ylabel"] = "CoM acceleration [m/s^2]"

    OUT["COM"]["titel"] = "CoM position"
    OUT["COM_vel"]["titel"] = "CoM velocity"
    OUT["COM_acc"]["titel"] = "CoM acceleration"

    # output regression information
    OUT["Regression"] = {}
    OUT["Regression"]["L-Rsq"] = Rsq_vect_L
    OUT["Regression"]["L-Coeff"] = Coeff_vect_L
    OUT["Regression"]["L-Intercept"] = Intervect_vect_L

    # output regression information
    OUT["Regression"] = {}
    OUT["Regression"]["R-Rsq"] = Rsq_vect_R
    OUT["Regression"]["R-Coeff"] = Coeff_vect_R
    OUT["Regression"]["R-Intercept"] = Intervect_vect_R

    # output regression information
    OUT["Regression"] = {}
    OUT["Regression"]["Rsq"] = Rsq_vect
    OUT["Regression"]["Coeff"] = Coeff_vect
    OUT["Regression"]["Intercept"] = Intervect_vect

    return OUT


def feedback_com_xcom(
    force, com, vcom, xcom, lfoot, rfoot, maxlag, l_HeelStrike, r_HeelStrike, time
):
    # inputs
    #   - force: nfr x 2 array with horizontal ground reaction forces
    #   - com: nfr x 2 array with horizontal COM position
    #   - vcom: nfr x 2 array with horizontal COM velocity
    #   - xcom: nfr x 2 array with horizontal extrapolated center of mass position
    #   - lfoot: nfr x 2 array with horizontal position left foot
    #   - rfoot: nfr x 2 array with horizontal position right foot
    #   - maxlag: maximal delay in the system (expressed as % gait cycle)
    #   - l_Heelstrike: time of left heelstrike
    #   - r_Heelstrike: time of right heelstrike
    #   - time: time vect
    # outputs:
    #   - x

    # get the sampling frequency
    fs = 1.0 / np.nanmean(np.diff(time))

    # get numer of strides and duration stride
    n_strides = len(l_HeelStrike) - 1
    stride_f = np.nanmean(np.diff(l_HeelStrike))

    # get a time vector
    nfr = len(com)
    time = np.linspace(0, (nfr - 1) / fs, nfr)

    # pre-allocate variables
    corr_phase_xcom = np.zeros((50, 2, 2))
    gain_phase_xcom = np.zeros((50, 2, 2))
    lag_xcom = np.zeros((2, 2))
    corr_phase_com = np.zeros((50, 2, 2))
    gain_phase_com = np.zeros((50, 2, 2))
    gain_phase_vcom = np.zeros((50, 2, 2))
    lag_com = np.zeros((2, 2))

    for dim in range(2):  # horizontal dims (xy)
        # combine data in nfr x 6 array
        dimdata = np.vstack(
            (
                force[:, dim],
                com[:, dim],
                vcom[:, dim],
                xcom[:, dim],
                lfoot[:, dim],
                rfoot[:, dim],
            )
        ).T
        for ft in range(2):  # left or right heelstrikes
            ndata2 = np.zeros((100, n_strides - 2, 6))  # pre allocate array
            for var in range(6):
                if ft == 0:
                    tmp, _, _, _ = norm_cycle(time, l_HeelStrike, dimdata[:, var])
                else:
                    tmp, _, _, _ = norm_cycle(time, r_HeelStrike, dimdata[:, var])
                ndata2[:, :, var] = tmp[:, :-2]
            # com and xcom position w.r.t stance foot position
            ndata2[:, :, 1] = ndata2[:, :, 1] - np.mean(
                ndata2[9:40, :, 4 + ft], axis=0
            )  # com pos wrt stance foot
            ndata2[:, :, 3] = ndata2[:, :, 3] - np.mean(
                ndata2[9:40, :, 4 + ft], axis=0
            )  # xcom pos wrt stance foot

            # for var in range(6):
            #     plt.figure()
            #     plt.plot(ndata2[:,:,var])
            # plt.show()

            # pre-allocate output matrices
            corr_force_xcom = np.zeros((50, maxlag))
            gain_force_xcom = np.zeros((50, maxlag))
            corr_force_com = np.zeros((50, maxlag))
            gain_force_com = np.zeros((50, maxlag))
            gain_force_vcom = np.zeros((50, maxlag))

            # loop over lags in system (% gait cycle)
            for j in range(maxlag):
                for i in range(50, 100):
                    mf = ndata2[i, :, 0]  # GRF
                    mx = ndata2[i - j - 1, :, 3]  # xCOM
                    mc = ndata2[i - j- 1, :, 1]  # COM
                    mv = ndata2[i - j- 1, :, 2]  # COM velocity

                    # remove nans
                    iNan = np.logical_or(np.isnan(mf), np.isnan(mx))
                    mx = np.delete(mx, iNan)
                    mf = np.delete(mf, iNan)
                    mc = np.delete(mc, iNan)
                    mv = np.delete(mv, iNan)

                    # pearson correlation
                    corr_force_xcom[i - 50, j], _ = pearsonr(mf, mx)
                    p_temp = np.polyfit(mx, mf, 1)
                    gain_force_xcom[i - 50, j] = p_temp[0]
                    RegModel = LinearRegression()
                    X = np.column_stack((mc, mv))
                    RegModel.fit(X, mf)
                    y_predicted = RegModel.predict(X)
                    corr_force_com[i - 50, j] = np.sqrt(r2_score(mf, y_predicted))
                    gain_force_com[i - 50, j] = RegModel.coef_[0]
                    gain_force_vcom[i - 50, j] = RegModel.coef_[1]


            test = corr_force_xcom.copy()
            test[test > 0] = np.nan
            ztest = np.tanh(np.nanmean(np.arctanh(test), axis=0))
            lag_xcom[dim, ft] = np.argmin(ztest)
            corr_phase_xcom[:, dim, ft] = corr_force_xcom[:, int(lag_xcom[dim, ft])]
            gain_phase_xcom[:, dim, ft] = gain_force_xcom[:, int(lag_xcom[dim, ft])]

            beta_vel = gain_force_vcom[:, :]
            beta_pos = gain_force_com[:, :]
            beta_weighted = beta_vel * (2 * np.pi * stride_f) + beta_pos
            lag_com[dim, ft] = np.argmin(np.min(beta_weighted, axis=1))
            corr_phase_com[:, dim, ft] = corr_force_com[:, int(lag_com[dim, ft])]
            gain_phase_com[:, dim, ft] = gain_force_com[:, int(lag_com[dim, ft])]
            gain_phase_vcom[:, dim, ft] = gain_force_vcom[:, int(lag_com[dim, ft])]

    return (
        corr_phase_xcom,
        gain_phase_xcom,
        lag_xcom,
        corr_phase_com,
        gain_phase_com,
        gain_phase_vcom,
        lag_com,
    )


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
                if boolfirst == True:  # first time vector is the one we will use for all signals
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

def filter_emg(emg_dat, emg_time, lowcut_bp = 20, highcut_bp = 400, lowcut_lp = 10):
    [nfr, ncol] = emg_dat.shape
    fs = 1. / np.nanmean(np.diff(emg_time))
    emg_filt = np.zeros(emg_dat.shape)
    for i in range(0, ncol):
        data = emg_dat[:, i]
        # Bandpass filter

        b_bp, a_bp = butter_bandpass(lowcut_bp, highcut_bp, fs)
        filtered_bp = filtfilt(b_bp, a_bp, data)

        # Take absolute value
        filtered_abs = np.abs(filtered_bp)

        # Lowpass filter
        b_lp, a_lp = butter_lowpass(lowcut_lp, fs)
        filtered_lp = filtfilt(b_lp, a_lp, filtered_abs)
        emg_filt[:,i] = filtered_lp
    return(emg_filt)

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

# class to create data table.
# I like it to store all my outcomes (of single measurements) in a datatable. This class gives me the ability to
# compute average values for a specific id (e.g. average of all measurements in a single subject and so on). I also
# gives the option to export this to a csv file and some general plot functions.
#       under development !
class datatable:
    def __init__(self, nrows, ncols, header = 'none'):
        # init the data matrices
        self.dat = np.empty([nrows, ncols])
        self.header = header
        self.rowcounter = 0

        # identifier (id) for datapoint. you can use this to select specific datapoints from the table
        self.id = np.empty(nrows)
        self.id.fill(np.nan)
        self.indexsel = np.zeros(nrows, dtype=bool)

    def add_datapoint(self,x, id):
        self.dat[self.rowcounter, range(0,len(x))] = x
        self.id[self.rowcounter] = id
        self.rowcounter = self.rowcounter + 1

    def get_indexmatchid(self, idsel):
        # get indices with matching ids
        indexsel = np.zeros(len(self.id), dtype=bool)
        if isinstance(idsel, int):
            indexsel = self.id == idsel
        else:
            for i in idsel:
                isel = self.id == i
                if np.any(isel):
                    indexsel[isel] = True
        return indexsel

    def set_header(self, header):
        self.header = header

    def printcsv(self, filepath):
        # print data to a file
        print('currently not supported')

    def get_mean_std_id(self):
        # compute mean and std for all data of a unique ID
        print('currently not supported')

# ToDo: implement a gait event detection function (maybe something for simone)
# (see e-mail of Sjoerd and Sina): This should work without a GUI !



# I stole this from: https://github.com/pyCGM2/pyCGM2/blob/Master/pyCGM2/Signal/detect_peaks.py
from typing import Optional, Tuple
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