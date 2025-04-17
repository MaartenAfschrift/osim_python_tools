
from scipy import signal
import numpy as np
import pandas as pd
import os
import sys

def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    fs = 1 / np.round(np.mean(np.diff(time)), 16)
    wn = lowpass_cutoff_frequency / (fs / 2)
    sos = signal.butter(order / 2, wn, btype='low', output='sos')
    dataFilt = signal.sosfiltfilt(sos, data, axis=0)

    return dataFilt

def lowPassFilterDataFrame(data, lowpass_cutoff_frequency, order=4):
    time = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    dataFilt = lowPassFilter(time, data, lowpass_cutoff_frequency, order)
    dataFilt = pd.DataFrame(dataFilt, columns=data.columns)
    # return dataframe with time vector
    return pd.concat([time, dataFilt], axis=1)

def readMotionFile(filename):
    """ Reads OpenSim .mot and .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = np.full((nr, len(labels)), np.nan)
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        if len(d) == nc:
            data[i-1,:] = d

    file_id.close()
    dat = pd.DataFrame(data, columns = labels)

    return dat


def WriteMotionFile(data_matrix, colnames, filename):


    datarows, datacols = data_matrix.shape
    time = data_matrix[:, 0]
    range_values = [time[0], time[-1]]

    if len(colnames) != datacols:
        raise ValueError(f'Number of column names ({len(colnames)}) does not match the number of columns in the data ({datacols})')

    # Open the file for writing
    try:
        with open(filename, 'w') as fid:
            # Write MOT file header
            fid.write(f'{filename}\nnRows={datarows}\nnColumns={datacols}\n\n')
            # ToDo: maybe check if data_matrix is in degrees or radians
            fid.write(
                f'name {filename}\ndatacolumns {datacols}\ndatarows {datarows}\nrange {range_values[0]} {range_values[1]}\ninDegrees=yes\nendheader\n')
            # Write column names
            cols = '\t'.join(colnames) + '\n'
            fid.write(cols)

            # Write data
            for i in range(datarows):
                row = '\t'.join([f'{value:20.10f}' for value in data_matrix[i, :]]) + '\n'
                fid.write(row)

    except IOError:
        print(f'\nERROR: {filename} could not be opened for writing...\n')

class diary:
    def __init__(self, filename="output_log.txt"):
        self.filename = filename
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.file = None

    def on(self):
        """Start logging output to the file."""
        if self.file is None:
            self.file = open(self.filename, "w")
            sys.stdout = self.file
            sys.stderr = self.file
            print("Diary ON - Logging started")

    def off(self):
        """Stop logging and restore normal output."""
        if self.file:
            print("Diary OFF - Logging stopped")  # This goes into the file
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.file.close()
            self.file = None
            print("Diary OFF - Logging stopped")  # This prints to the screen