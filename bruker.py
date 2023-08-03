from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
from pathlib import Path
from PIL import Image
import tifffile

from fish import Fish


class BrukerFish(Fish):
    def __init__(self, folder_path):
        super().__init__(folder_path)

        self.bruker = True
        self.data_paths = dict()

        self.process_bruker_filestructure()
        self.create_frametimes_txt()
        self.combine_images()

    def process_bruker_filestructure(self):
        '''Appends Bruker specific file paths to the data_paths'''

        with os.scandir(self.exp_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path) and entry.name.startswith('TSeries-'):
                    self.data_paths['raw'] = Path(entry.path)

        if 'raw' in self.data_paths.keys():
            with os.scandir(self.data_paths['raw']) as entries:
                for entry in entries:
                    if os.path.isdir(entry.path) and entry.name == 'References':
                        self.data_paths['references'] = Path(entry.path)
                    elif entry.name.endswith('.xml') and 'MarkPoints' not in entry.name:
                        self.data_paths['log'] = Path(entry.path)

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

    def combine_images(self):
        '''Combines raw images to separate .tif files of individual planes for each channel'''
        ch1_exists = False
        ch2_exists = False

        with os.scandir(self.data_paths['raw']) as entries:
            for entry in entries:
                if entry.name.endswith('.ome.tif'):
                    if 'Ch1' in entry.name:
                        ch1_exists = True
                    elif 'Ch2' in entry.name:
                        ch2_exists = True

        if ch1_exists:
            self.data_paths['ch1'] = dict()
            self.combine_channel_images('Ch1')

        if ch2_exists:
            self.data_paths['ch2'] = dict()
            self.combine_channel_images('Ch2')

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

        ch_images = [np.array(Image.open(img_path)) for img_path in ch_image_paths]
        ch_folder_path = self.exp_path.joinpath(channel.lower())
        if not os.path.exists(ch_folder_path):
            os.mkdir(ch_folder_path)
        ch_image_path = Path(os.path.join(ch_folder_path, f'{channel}.tif'))
        tifffile.imwrite(ch_image_path, ch_images)

        self.data_paths[channel.lower()]['raw_image'] = ch_new_path

    def rotate_channel_images(self, angle=0):
        '''Rotates each channel's image by angle'''
        if 'ch1' in self.data_paths.keys():
            self.rotate_image(self.data_paths['ch1']['raw_image'], angle=angle, prefix='ch1')

        if 'ch2' in self.data_paths.keys():
            self.rotate_image(self.data_paths['ch2']['raw_image'], angle=angle, prefix='ch2')

    def split_bruker_volumes(self, channel):
        '''Splits volumes to individual planes'''
        channels = ['ch1', 'ch2']
        if channel not in channels:
            raise ValueError(f'channel needs to be one of {channels}')

        ch_image_paths = []

        with os.scandir(self.data_paths['raw']) as entries:
            for entry in entries:
                if entry.name.endswith('.ome.tif') and channel in entry.name:
                        ch_image_paths.append(Path(entry.path))

        planes = np.array([int(img_path.name[img_path.name.find('Cycle')+5:img_path.name.find('Cycle')+10]) for img_path in img_paths])

        if 'rotated' in self.data_paths[channel].keys():
            img_path = self.data_paths[channel]['rotated']
        else:
            img_path = self.data_paths[channel]['raw_image']

        image = np.array(Image.open(img_path))

        for plane in np.unique(planes):
            # For each plane, create a subfolder to save the images
            plane_path = img_path.parent.joinpath(f'img_stack_{plane-1}')
            if not os.path.exists(plane_path):
                os.mkdir(plane_path)

            indices = np.where(planes == plane)
            plane_image = image[indices]
            self.frametimes_df

            new_path = plane_path.jonpath('image.tif')
            tifffile.imwrite(new_path, plane_image)


def round_microseconds(str_time):
    seconds = str_time[str_time.rfind(':')+1:]
    rounded_microsecs = round(float(seconds), 6)

    return str_time[:str_time.rfind(':')+1] + str(rounded_microsecs)