from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
from pathlib import Path
import tifffile

from caImageAnalysis import Fish


class BrukerFish(Fish):
    def __init__(self, folder_path):
        super().__init__(folder_path)

        self.bruker = True
        self.volumetric = False

        self.process_bruker_filestructure()

    def process_bruker_filestructure(self):
        '''Appends Bruker specific file paths to the data_paths'''
        with os.scandir(self.exp_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path): 
                    if entry.name.startswith('img_stack_'):
                        self.volumetric = True
                    elif entry.name == 'mesmerize-batch':
                        self.data_paths['mesmerize'] = Path(entry.path)
                    else:
                        self.data_paths['raw'] = Path(entry.path)

                elif entry.name.endswith('ch2.tif'):
                    self.data_paths['raw_image'] = Path(entry.path)
                elif entry.name == 'raw_rotated.tif':
                    self.data_paths['rotated'] = Path(entry.path)
                elif entry.name == 'frametimes.txt':
                    self.data_paths['frametimes'] = Path(entry.path)
                elif entry.name == 'opts.pkl':
                    self.data_paths['opts'] = Path(entry.path)
                    

        if 'raw' in self.data_paths.keys():
            with os.scandir(self.data_paths['raw']) as entries:
                for entry in entries:
                    if os.path.isdir(entry.path) and entry.name == 'References':
                        self.data_paths['references'] = Path(entry.path)
                    elif (entry.name.endswith('.xml')) and (not entry.name.startswith('.')):
                        parsed = entry.name.split('_')
                        if parsed[-2] == 'MarkPoints':
                            self.data_paths['markpoints'] = Path(entry.path)
                        elif parsed[-2] == 'VoltageOutput':
                            self.data_paths['voltage_output'] = Path(entry.path)
                        else:
                            self.data_paths['log'] = Path(entry.path)
                        
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

    def create_frametimes_txt(self):
        '''Creates a frametimes.txt file from the log xml file'''
        with open(self.data_paths['log'], 'r') as file:
            log = file.read()

        Bs_data = BeautifulSoup(log)
        frames = Bs_data.find_all('frame')
        str_time = Bs_data.find_all('sequence')[0]['time']

        str_time = round_microseconds(str_time)
        dt_time = dt.strptime(str_time, '%H:%M:%S.%f')

        frametimes_path = os.path.join(self.exp_path, 'frametimes.txt')
        self.data_paths['frametimes'] = Path(frametimes_path)

        with open(frametimes_path, 'w') as file:
            for frame in frames:
                str_abstime = round_microseconds(frame['absolutetime'])
                dt_abstime = timedelta(seconds=int(str_abstime[:str_abstime.find('.')]),
                                       microseconds=int(str_abstime[str_abstime.find('.')+1:]))
                str_final_time = dt.strftime(dt_time + dt_abstime, '%H:%M:%S.%f')
                file.write(str_final_time + '\n')

        self.raw_text_frametimes_to_df()

    def combine_channel_images(self, channel):
        '''Combines a channel's images'''
        channels = ['Ch1', 'Ch2']
        if channel not in channels:
            raise ValueError(f'channel needs to be one of {channels}')

        ch_image_paths = []
        with os.scandir(self.data_paths['raw']) as entries:
            for entry in entries:
                if entry.name.endswith('.ome.tif') and channel in entry.name:
                    ch_image_paths.append(Path(entry.path))

        ch_images = [np.array(tifffile.imread(img_path)) for img_path in ch_image_paths]
        raw_img = np.concatenate(ch_images)
        
        ch_image_path = Path(os.path.join(self.exp_path, f'{channel.lower()}.tif'))
        tifffile.imwrite(ch_image_path, raw_img)

        self.data_paths['raw_image'] = ch_image_path

    def split_bruker_volumes(self, channel):
        '''Splits volumes to individual planes'''
        channels = ['Ch1', 'Ch2']
        if channel not in channels:
            raise ValueError(f'channel needs to be one of {channels}')

        ch_image_paths = []

        with os.scandir(self.data_paths['raw']) as entries:
            for entry in entries:
                if entry.name.endswith('.ome.tif') and channel in entry.name:
                        ch_image_paths.append(Path(entry.path))

        n_planes = tifffile.imread(ch_image_paths[0]).shape[0]
        
        if 'rotated' in self.data_paths.keys():
            img = tifffile.imread(self.data_paths['rotated'])
        else:
            img = tifffile.imread(self.data_paths['raw_image'])

        for plane in range(n_planes):
            plane_folder_path = os.path.join(self.exp_path, f'img_stack_{plane}')
            if not os.path.exists(plane_folder_path):
                os.mkdir(plane_folder_path)

            plane_img = img[plane::n_planes]
            tifffile.imwrite(os.path.join(plane_folder_path, 'image.tif'), plane_img)

            plane_frametimes = self.frametimes_df[plane::n_planes].copy()
            plane_frametimes.reset_index(drop=True)
            plane_frametimes.to_hdf(os.path.join(plane_folder_path, 'frametimes.h5'), 'frametimes')
        
        self.process_bruker_filestructure()


def round_microseconds(str_time):
    seconds = str_time[str_time.rfind(':')+1:]
    rounded_microsecs = round(float(seconds), 6)

    return str_time[:str_time.rfind(':')+1] + str(rounded_microsecs)