from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

from .bruker_utils import round_microseconds


class VoltageOutput:
    '''Bruker Voltage Output'''
    def __init__(self, path, log_path):
        '''
        path: path to the VoltageOutput xml file
        log: path to the experiment xml file
        '''
        self.xml_path = path

        with open(self.xml_path, 'r') as file:
            xml = file.read()

        Bs_data = BeautifulSoup(xml)
        waveforms = Bs_data.find_all('waveform')

        self.waveform = [waveform for waveform in waveforms if waveform.enabled.string == 'true'][0]
        self.unit_scale_factor = float(self.waveform.unitscalefactor.string)  # scaling from waveform voltage to output voltage
        self.pulse_train = self.waveform.waveformcomponent_pulsetrain

        self.rest_potential = float(self.pulse_train.restpotential.string)  # the resting potential of the waveform (V)
        self.initial_delay = int(self.pulse_train.firstpulsedelay.string)  # delay at the beginning of a pulse train (ms)
        self.pulse_count = int(self.pulse_train.pulsecount.string)  # number of pulses within a single train
        self.pulse_width = int(self.pulse_train.pulsewidth.string)  # duration of a single pulse (ms)
        self.pulse_spacing = int(self.pulse_train.pulsespacing.string)  # duration between pulses (ms)
        self.pulse_potential_start = float(self.pulse_train.pulsepotentialstart.string)  # voltage of the first pulse (V)
        self.pulse_potential_delta = float(self.pulse_train.pulsepotentialdelta.string)  # change in voltage between consecutive pulses (V)
        self.repetitions = int(self.pulse_train.repetitions.string)  # repetitions of the entire pulse train
        self.delay_between_reps = int(self.pulse_train.delaybetweenreps.string)  # duration between pulse trains

        self.n_pulses = self.pulse_count * self.repetitions  # total number of pulses duing the waveform

        self.get_output_dt(log_path)
    
    def get_output_dt(self, log_path):
        '''Returns the datetime of successful outputs'''
        with open(log_path, 'r') as file:
            log = file.read()

        Bs_log_data = BeautifulSoup(log)

        first_line = Bs_log_data.find_all('pvscan')
        str_time = first_line[0]['date'].split(' ')[1]  # start time of the experiment

        dt_time = dt.strptime(str_time, '%H:%M:%S')

        if first_line[0]['date'].split(' ')[2] == 'PM':
            # datetime doesn't handle military time well
            military_adjustment = timedelta(hours=12)
            dt_time = dt_time + military_adjustment

        frames = Bs_log_data.find_all('frame')
        str_abstime = round_microseconds(frames[0]['relativetime'])
        dt_abstime = timedelta(seconds=int(str_abstime[:str_abstime.find('.')]),
                                        microseconds=int(str_abstime[str_abstime.find('.')+1:]))
        dt_start_time = dt_time + dt_abstime

        voltage_out_log = Bs_log_data.find_all('voltageoutput')[0]
        self.name = voltage_out_log['name']
        if 'gavage' or 'DOI' in self.name:
            # for gavage, injections occur only when voltage drops from 5V to 0V
            self.delta_V = -5  # we'll use this as change in voltage necessary for a 'successful output' 
        
        abstime = round_microseconds(voltage_out_log['relativetime'])  # start time of the voltage output
        dt_abstime = timedelta(seconds=int(abstime[:abstime.find('.')]),
                                        microseconds=int(abstime[abstime.find('.')+1:]))
        dt_start_vo_time = dt_start_time + dt_abstime  # start time of the voltage output as a datetime

        y, _ = self.get_waveform()
        self.output_ts = np.where(np.diff(y) == self.delta_V)[0] + 1  # times of successful outputs (ms)
        
        self.output_dts = list()
        for t in self.output_ts:
            dt_output_ts = dt_start_vo_time + timedelta(milliseconds=int(t))  # times of successful outputs as datetime
            self.output_dts.append(dt_output_ts.time())

    def align_pulses_to_frametimes(self, frametimes):
        '''Creates a 'pulse' column in a frametimes dataframe, and adds the pulse number'''
        frametimes['pulse'] = 0
        curr_pulse = 0
        
        for i, row in frametimes.iterrows():
            try:
                if row['time'] >= self.output_dts[curr_pulse]:
                    curr_pulse += 1
                    frametimes.loc[i, 'pulse'] = curr_pulse
                else:
                    frametimes.loc[i, 'pulse'] = curr_pulse
            except IndexError:
                # exception for the last pulse
                frametimes.loc[i, 'pulse'] = curr_pulse

        return frametimes

    def get_pulse_train(self, plot=False):
        '''Returns the y, x values of a single pulse train'''
        y = np.repeat(self.rest_potential * self.unit_scale_factor, self.initial_delay)
        x = np.arange(self.initial_delay)

        for pulse in range(self.pulse_count):
            pulse_V = (self.pulse_potential_start * self.unit_scale_factor) + (self.pulse_potential_delta * self.unit_scale_factor * pulse)
            pulse_y = np.repeat(pulse_V, self.pulse_width)
            pulse_x = np.arange(x[-1] + 1, x[-1] + 1 + self.pulse_width)
            
            y = np.append(y, pulse_y)
            x = np.append(x, pulse_x)

            if pulse != self.pulse_count-1:
                # if there is another pulse after this pulse, add the pulse spacing
                pulse_spacing_y = np.repeat(self.rest_potential * self.unit_scale_factor, self.pulse_spacing)
                pulse_spacing_x = np.arange(x[-1] + 1, x[-1] + 1 + self.pulse_spacing)
                
                y = np.append(y, pulse_spacing_y)
                x = np.append(x, pulse_spacing_x)
            
            elif pulse == self.pulse_count-1:
                # make sure that the voltage comes back to resting for the final pulse
                y = np.append(y, self.rest_potential * self.unit_scale_factor)
                x = np.append(x, x[-1] + 1)

        if plot:
            plt.figure(figsize=(20, 5))
            plt.plot(x, y)
            plt.xlabel('Time (ms)', fontsize=18)
            plt.ylabel('Voltage (V)', fontsize=18)

        return y, x

    def get_waveform(self, plot=False):
        '''Returns the y, x values of the entire waveform, including any repetitions of the pulse train'''
        for train in range(self.repetitions):
            if train == 0:
                y, x = self.get_pulse_train()

            else:
                delay_y = np.repeat(self.rest_potential * self.unit_scale_factor, self.delay_between_reps)
                delay_x = np.arange(x[-1] + 1, x[-1] + 1 + self.delay_between_reps)

                train_y, train_x = self.get_pulse_train()
                train_x = train_x + delay_x[-1] + 1
                y = np.concatenate((y, delay_y, train_y))
                x = np.concatenate((x, delay_x, train_x))

        if plot:
            plt.figure(figsize=(20, 5))
            plt.plot(x, y)
            plt.xlabel('Time (ms)', fontsize=18)
            plt.ylabel('Voltage (V)', fontsize=18)

        return y, x
