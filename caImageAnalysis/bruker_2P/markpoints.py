from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

from .bruker_utils import round_microseconds


class MarkPoints:
    '''Bruker MarkPoints'''
    def __init__(self, path, log_path, cycle=None):
        '''
        path: path to the MarkPoints xml files
        log: path to the experiment xml file
        '''
        self.xml_path = path
        self.cycle = cycle

        with open(self.xml_path, 'r') as file:
            xml = file.read()

        Bs_data = BeautifulSoup(xml)

        self.uncaging_laser = Bs_data.pvmarkpointelement['uncaginglaser']
        self.uncaging_laser_power = float(Bs_data.pvmarkpointelement['uncaginglaserpower'])

        self.iterations = int(Bs_data.pvmarkpointserieselements['iterations'])  # number of iterations of the pulse train in an experiment
        self.iteration_delay = float(Bs_data.pvmarkpointserieselements['iterationdelay'])
        self.repetitions = int(Bs_data.pvmarkpointelement['repetitions'])  # number of pulse repetitions in a single iteration
        self.pulse_duration = float(Bs_data.pvgalvopointelement['duration'])  # in ms
        self.initial_delay = float(Bs_data.pvgalvopointelement['initialdelay'])  # delay at the beginning of a pulse train (ms)
        self.interpoint_delay = float(Bs_data.pvgalvopointelement['interpointdelay'])  # delay between two pulses (ms)
        self.spiral_revolutions = float(Bs_data.pvgalvopointelement['spiralrevolutions'])  # number of spiral revolutions

        points = Bs_data.find_all('point')
        self.points = dict()
        for point in points:
            self.points[int(point['index'])] = {
                'spiral_height': float(point['spiralheight']),
                'spiral_size_in_microns': float(point['spiralsizeinmicrons']),
                'spiral_width': float(point['spiralwidth']),
                'x': float(point['x']),
                'y': float(point['y'])
            }
            
        self.n_pulses = self.repetitions * self.iterations  # total number of pulses duing the waveform
        self.get_output_dt(log_path)
    
    def get_output_dt(self, log_path):
        '''Returns the datetime of successful outputs'''
        with open(log_path, 'r') as file:
            log = file.read()

        Bs_log_data = BeautifulSoup(log)

        sequences = Bs_log_data.find_all('sequence')

        if self.cycle is not None:
            for seq in sequences:
                if seq['cycle'] == str(self.cycle):
                    start_time = seq['time']  # start time of the markpoints

        else:
            for seq in sequences:
                if seq.markpoints:
                    start_time = seq['time']  # start time of the markpoints

        start_time = round_microseconds(start_time)
        dt_start_time = dt.strptime(start_time, '%H:%M:%S.%f')

        y, _ = self.get_waveform()
        self.output_start_ts = np.where(np.diff(y) == self.uncaging_laser_power)  # start times of pulses (ms)
        # add 1 to each value to change them from Python range (first time as zero) to the imaging range (first time as 1)
        self.output_start_ts = self.output_start_ts + np.repeat(1, len(self.output_start_ts))

        self.output_stop_ts = np.where(np.diff(y) == -self.uncaging_laser_power)  # stop times of pulses (ms)
        # add 1 to each value to change them from Python range (first time as zero) to the imaging range (first time as 1)
        self.output_stop_ts = self.output_stop_ts + np.repeat(1, len(self.output_stop_ts))

        self.output_start_dts = list()
        for t in self.output_start_ts[0]:
            dt_output_ts = dt_start_time + timedelta(milliseconds=int(t))  # times of output starts as datetime
            self.output_start_dts.append(dt_output_ts.time())

        self.output_stop_dts = list()
        for t in self.output_stop_ts[0]:
            dt_output_ts = dt_start_time + timedelta(milliseconds=int(t))  # times of output stops as datetime
            self.output_stop_dts.append(dt_output_ts.time())

    def align_pulses_to_frametimes(self, frametimes):
        '''Creates a 'pulse' column in a frametimes dataframe, and adds the pulse number'''
        if 'pulse' not in frametimes.columns:
            frametimes['pulse'] = 0
            last_pulse = 0
        else:
            # if there are multiple markpoints events in the experiment, keeps track of the previous markpoints' final pulse
            last_pulse = frametimes['pulse'].unique()[-1]

        curr_pulse = 0  # pulse index for the current markpoints
        
        for i, row in frametimes.iterrows():
            try:
                if (row['time'] >= self.output_start_dts[curr_pulse]) and (row['time'] <= self.output_stop_dts[curr_pulse]):
                    frametimes.loc[i, 'pulse'] = -1  # pulse of -1 should indicate during pulse, meaning the imaging shutters should be closed
                elif (row['time'] >= self.output_start_dts[curr_pulse]) and (row['time'] > self.output_stop_dts[curr_pulse]):
                    curr_pulse += 1
                    last_pulse += 1
                    frametimes.loc[i, 'pulse'] = last_pulse

            except IndexError:
                # exception for the last pulse
                frametimes.loc[i, 'pulse'] = last_pulse

        return frametimes

    def get_pulse_train(self, plot=False):
        '''Returns the y, x values of a single pulse train'''
        if self.initial_delay < 1:
            # if the initial delay in 0.__ something, we lose a y data point in the beginning
            y = np.repeat(0, 1)
        else:
            y = np.repeat(0, self.initial_delay)

        x = np.arange(self.initial_delay)

        for pulse in range(self.repetitions):
            pulse_y = np.repeat(self.uncaging_laser_power, self.pulse_duration)
            pulse_x = np.arange(x[-1] + 1, x[-1] + 1 + self.pulse_duration)
            
            y = np.append(y, pulse_y)
            x = np.append(x, pulse_x)

            if pulse != self.repetitions-1:
                # if there is another pulse after this pulse, add the pulse spacing
                pulse_spacing_y = np.repeat(0, self.interpoint_delay)
                pulse_spacing_x = np.arange(x[-1] + 1, x[-1] + 1 + self.interpoint_delay)
                
                y = np.append(y, pulse_spacing_y)
                x = np.append(x, pulse_spacing_x)
            
            elif pulse == self.repetitions-1:
                # make sure that the voltage comes back to resting for the final pulse
                y = np.append(y, 0)
                x = np.append(x, x[-1] + 1)

        if plot:
            plt.figure(figsize=(20, 5))
            plt.plot(x, y)
            plt.xlabel('Time (ms)', fontsize=18)
            plt.ylabel('Voltage (V)', fontsize=18)

        return y, x

    def get_waveform(self, plot=False):
        '''Returns the y, x values of the entire waveform, including any repetitions of the pulse train'''
        for train in range(self.iterations):
            if train == 0:
                y, x = self.get_pulse_train()

            else:
                delay_y = np.repeat(0, self.iteration_delay)
                delay_x = np.arange(x[-1] + 1, x[-1] + 1 + self.iteration_delay)

                train_y, train_x = self.get_pulse_train()
                train_x = train_x + delay_x[-1] + 1
                y = np.concatenate((y, delay_y, train_y))
                x = np.concatenate((x, delay_x, train_x))

        if plot:
            plt.figure(figsize=(20, 5))
            plt.plot(x, y)
            plt.xlabel('Time (ms)', fontsize=18)
            plt.ylabel('Laser power', fontsize=18)

        return y, x
