from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tifffile

from .bruker_utils import round_microseconds
from .markpoints import MarkPoints
from .voltage_output import VoltageOutput
from caImageAnalysis import Fish
from caImageAnalysis.utils import calculate_fps, load_pickle


class BrukerFish(Fish):
    def __init__(self, folder_path, region=''):
        self.region = region
        super().__init__(folder_path)
        
        self.bruker = True

        self.process_bruker_filestructure()
        self.volumetric = self.check_volumetric()

        if self.volumetric:
            self.fps = calculate_fps(self.data_paths['volumes']['0']['frametimes'])
        else:
            self.fps = calculate_fps(self.data_paths['frametimes'])

        # TODO: self.zoom = self.get_zoom()

    def process_bruker_filestructure(self):
        '''Appends Bruker specific file paths to the data_paths'''
        with os.scandir(self.exp_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path): 
                    if entry.name.startswith('img_stack_'):
                        self.volumetric = True
                    elif entry.name == 'mesmerize-batch':
                        self.data_paths['mesmerize'] = Path(entry.path)
                    elif entry.name == 'stytra':
                        self.data_paths['stytra'] = Path(entry.path)
                    elif len(self.region) > 0 and entry.name.startswith(self.region):
                        self.data_paths['raw'] = Path(entry.path)
                    elif len(self.region) == 0:
                        self.data_paths['raw'] = Path(entry.path)

                elif entry.name.endswith('ch2.tif') and not entry.name.startswith('.'):
                    self.data_paths['raw_image'] = Path(entry.path)
                elif entry.name == 'raw_rotated.tif':
                    self.data_paths['rotated'] = Path(entry.path)
                elif entry.name.endswith('frametimes.h5') and not entry.name.startswith('.'):
                    self.data_paths['frametimes'] = Path(entry.path)
                elif entry.name.endswith('frametimes.txt') and not entry.name.startswith('.') and 'frametimes' not in self.data_paths.keys():
                    self.data_paths['frametimes'] = Path(entry.path)
                    self.raw_text_frametimes_to_df()
                elif entry.name == 'opts.pkl':
                    self.data_paths['opts'] = Path(entry.path)
                elif entry.name == 'clusters.pkl':
                    self.data_paths['clusters'] = Path(entry.path)
                    self.clusters = load_pickle(Path(entry.path))
                elif entry.name == 'temporal.h5':
                    self.data_paths['temporal'] = Path(entry.path)
                    self.temporal_df = pd.read_hdf(self.data_paths['temporal'])
                elif entry.name == 'vol_temporal.pkl':
                    self.data_paths['vol_temporal'] = Path(entry.path)
                    self.vol_temporal = load_pickle(self.data_paths['vol_temporal'])
                elif entry.name == 'anatomy.tif':
                    self.data_paths['anatomy'] = Path(entry.path)
                elif 'C_frames' in entry.name and not entry.name.startswith('.'):
                    self.data_paths['C_frames'] = Path(entry.path)
                    
        if 'raw' in self.data_paths.keys():
            if 'anatomy' not in self.data_paths.keys():
                self.data_paths['anatomy'] = self.get_anatomy()

            with os.scandir(self.data_paths['raw']) as entries:
                for entry in entries:
                    if os.path.isdir(entry.path) and entry.name == 'References':
                        self.data_paths['references'] = Path(entry.path)
                    elif entry.name == self.data_paths['raw'].name + '.xml':
                        self.data_paths['log'] = Path(entry.path)
                    elif (entry.name.endswith('.xml')) and (not entry.name.startswith('.')):
                        if 'MarkPoints' in entry.name:
                            if 'markpoints' in self.data_paths.keys():
                                self.data_paths['markpoints'].append(Path(entry.path))
                            else:
                                self.data_paths['markpoints'] = [Path(entry.path)]
                        elif 'VoltageOutput' in entry.name:
                            self.data_paths['voltage_output'] = Path(entry.path)
                        
        if 'voltage_output' in self.data_paths.keys():
            self.voltage_output = VoltageOutput(self.data_paths['voltage_output'], self.data_paths['log'])
            try:
                self.frametimes_df = self.voltage_output.align_pulses_to_frametimes(self.frametimes_df)
            except AttributeError:
                # if this is the first time initializing, frametimes.txt might not have been created yet
                pass

        if 'markpoints' in self.data_paths.keys():
            self.markpoints = dict()
            self.data_paths['markpoints'].sort()

            for mp_path in self.data_paths['markpoints']:
                mp_path = str(mp_path)
                cycle = int(mp_path[mp_path.find('Cycle')+5:mp_path.rfind('_')])
                mp = MarkPoints(mp_path, self.data_paths['log'], cycle=cycle)
                self.markpoints[cycle] = mp
                try:
                    self.frametimes_df = self.markpoints[cycle].align_pulses_to_frametimes(self.frametimes_df)
                except AttributeError:
                    # if this is the first time initializing, frametimes.txt might not have been created yet
                    pass

        if self.volumetric:
            self.data_paths['volumes'] = dict()
            with os.scandir(self.exp_path) as entries:
                for entry in entries:
                    if entry.name.startswith('img_stack_'):
                        volume_ind = entry.name[entry.name.rfind('_')+1:]
                        self.data_paths['volumes'][volume_ind] = dict()

                        with os.scandir(entry.path) as subentries:
                            for sub in subentries:
                                if sub.name == 'image.tif':
                                    self.data_paths['volumes'][volume_ind]['image'] = Path(sub.path)
                                elif sub.name == 'frametimes.h5':
                                    self.data_paths['volumes'][volume_ind]['frametimes'] = Path(sub.path)

        if 'mesmerize' in self.data_paths.keys():
            self.process_mesmerize_filestructure()

    def get_anatomy(self):
        '''Finds the anatomy stack in the raw data folder'''
        imgs = [path for path in os.listdir(self.data_paths['raw']) if path.endswith('.ome.tif')]

        first_img = tifffile.imread(self.data_paths['raw'].joinpath(imgs[0]))
        last_img = tifffile.imread(self.data_paths['raw'].joinpath(imgs[-1]))

        if first_img.shape[1] != last_img.shape[1]:
            return self.data_paths['raw'].joinpath(imgs[-1])
        else:
            return None
        
    def check_volumetric(self):
        '''Checks if the experiment is volumetric'''
        volumetric = False
        
        with open(self.data_paths['log'], 'r') as file:
            log = file.read()

        Bs_data = BeautifulSoup(log)

        first_sequence = Bs_data.find_all('sequence')[0]
        if 'ZSeries' in first_sequence['type']:
            # if it's a Z-Series, automatically assume volumetric
            volumetric = True
            self.volumetric_type = 'ZSeries'

        else:
            # this is for "fake volumetric" image sequences
            subindexed_value = Bs_data.find_all('subindexedvalue')
            planes = list()
            for val in subindexed_value:
                try:
                    if 'Optotune ETL' in val['description']:
                        planes.append(val['value'])
                except:
                    pass
            
            if len(np.unique(planes)) > 1:
                volumetric = True
                self.volumetric_type = f'fake_volumetric_{len(np.unique(planes))}'

        return volumetric
    
    def create_frametimes_txt(self):
        '''Creates a frametimes.txt file from the log xml file'''
        with open(self.data_paths['log'], 'r') as file:
            log = file.read()

        Bs_data = BeautifulSoup(log)
        frames = Bs_data.find_all('frame')
        first_line = Bs_data.find_all('pvscan')
        str_time = first_line[0]['date'].split(' ')[1]  # start time of the experiment

        dt_time = dt.strptime(str_time, '%H:%M:%S')

        if first_line[0]['date'].split(' ')[2] == 'PM':
            # datetime doesn't handle military time well
            military_adjustment = timedelta(hours=12)
            dt_time = dt_time + military_adjustment

        if len(self.region) > 0:
            frametimes_path = os.path.join(self.exp_path, f'{self.region}_frametimes.txt')
        else:
            frametimes_path = os.path.join(self.exp_path, f'frametimes.txt')
        self.data_paths['frametimes'] = Path(frametimes_path)

        with open(frametimes_path, 'w') as file:
            for i, frame in enumerate(frames):
                if 'anatomy' in frame['parameterset']:  # if the anatomy stack starts
                    break
                else:
                    str_abstime = round_microseconds(frame['absolutetime'])
                    dt_abstime = timedelta(seconds=int(str_abstime[:str_abstime.find('.')]),
                                        microseconds=int(str_abstime[str_abstime.find('.')+1:]))
                    str_final_time = dt.strftime(dt_time + dt_abstime, '%H:%M:%S.%f')
                    file.write(str_final_time + '\n')

        self.raw_text_frametimes_to_df()

        if 'voltage_output' in self.data_paths.keys():
            self.frametimes_df = self.voltage_output.align_pulses_to_frametimes(self.frametimes_df)
        
        elif 'markpoints' in self.data_paths.keys():
            for cycle in self.markpoints:
                self.frametimes_df = self.markpoints[cycle].align_pulses_to_frametimes(self.frametimes_df)
        
        else:
            print('no voltage output or markpoints detected')
    
    def combine_channel_images(self, channel):
        '''Combines a channel's images'''
        channels = ['Ch1', 'Ch2']
        if channel not in channels:
            raise ValueError(f'channel needs to be one of {channels}')

        ch_image_paths = []
        for entry in sorted(os.scandir(self.data_paths['raw']), key=lambda e: e.name):
            if entry.name.endswith('.ome.tif') and channel in entry.name:
                ch_image_paths.append(Path(entry.path))

        ch_images = [np.array(tifffile.imread(img_path)) for img_path in ch_image_paths]
        
        # find out if there is an anatomy stack
        n_planes = ch_images[0].shape[0]
        anatomy_index = [i for i, img in enumerate(ch_images) if img.shape[0] != n_planes]
        if len(anatomy_index) != 0:
            del ch_images[anatomy_index[0]]
            self.data_paths['anatomy'] = ch_image_paths[anatomy_index[0]]

        raw_img = np.concatenate(ch_images)
        
        if len(self.region) > 0:
            ch_image_path = Path(os.path.join(self.exp_path, f'{self.region}_{channel.lower()}.tif'))
        else:
            ch_image_path = Path(os.path.join(self.exp_path, f'{channel.lower()}.tif'))
        tifffile.imwrite(ch_image_path, raw_img)

        self.data_paths['raw_image'] = ch_image_path

        plt.imshow(raw_img[0])

    def split_bruker_volumes(self, channel):
        '''Splits volumes to individual planes'''
        channels = ['Ch1', 'Ch2']
        if channel not in channels:
            raise ValueError(f'channel needs to be one of {channels}')

        ch_image_paths = []

        for entry in sorted(os.scandir(self.data_paths['raw']), key=lambda e: e.name):
            if entry.name.endswith('.ome.tif') and channel in entry.name:
                ch_image_paths.append(Path(entry.path))

        if 'rotated' in self.data_paths.keys():
            img = tifffile.imread(self.data_paths['rotated'])
        else:
            img = tifffile.imread(self.data_paths['raw_image'])

        if self.volumetric_type == 'ZSeries':
            n_planes = tifffile.imread(ch_image_paths[0]).shape[0]
        elif self.volumetric_type.startswith('fake_volumetric'):
            n_planes = int(self.volumetric_type[self.volumetric_type.rfind('_')+1:])
            len_plane = int(img.shape[0]/n_planes)  # number of frames in each plane
        
        for plane in range(n_planes):
            plane_folder_path = os.path.join(self.exp_path, f'img_stack_{plane}')
            if not os.path.exists(plane_folder_path):
                os.mkdir(plane_folder_path)

            if self.volumetric_type == 'ZSeries':
                plane_img = img[plane::n_planes]
                plane_frametimes = self.frametimes_df[plane::n_planes].copy()
            elif self.volumetric_type.startswith('fake_volumetric'):
                plane_img = img[plane*len_plane:(plane+1)*len_plane]
                plane_frametimes = self.frametimes_df[plane*len_plane:(plane+1)*len_plane].copy()
            
            tifffile.imwrite(os.path.join(plane_folder_path, 'image.tif'), plane_img)

            plane_frametimes = plane_frametimes.reset_index(drop=True)
            plane_frametimes.to_hdf(os.path.join(plane_folder_path, 'frametimes.h5'), 'frametimes')
        
        self.process_bruker_filestructure()

    def get_pulse_frames(self):
        '''Gets frame indices for each pulse (for bruker recordings)
        Picks the smallest frame across planes'''
        if not hasattr(self, 'temporal_df'):
            raise AttributeError('Requires a temporal_df: Run temporal.py/save_temporal')
        
        min_pulse_frames = list()
        
        if hasattr(self, 'voltage_output'):
            for i in range(self.voltage_output.n_pulses):
                min_pulse_frames.append(np.min([pulses[i] for pulses in self.temporal_df.pulse_frames]))
        
        # elif hasattr(self, 'markpoints'):
        #     pass
        #     mp = list(self.markpoints.values())[0]
        #     for i in range(mp.n_pulses):
        #         min_pulse_frames.append()

        else:
            raise AttributeError('Requires a VoltageOutput or a MarkPoints')

        return min_pulse_frames
    
    def get_zoom(self):
        '''Extracts the zoom info from log'''
        with open(self.data_paths['log'], 'r') as file:
            log = file.read()

        Bs_data = BeautifulSoup(log)
        values = Bs_data.find_all('pvstatevalue')
        
        for val in values:
            if val['key'] == 'opticalZoom':
                print(val['value'])
        return