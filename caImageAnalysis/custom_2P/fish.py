from pathlib import Path
import os
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import caiman as cm
from tifffile import imread, imwrite, memmap
from scipy.ndimage import rotate

from caImageAnalysis.utils import crop_image


class Fish:
    '''Adapted from fishy.py of https://github.com/Naumann-Lab/caImageAnalysis'''
    def __init__(self, folder_path):
        self.exp_path = Path(folder_path)
        self.bruker = False
        self.data_paths = dict()

        self.parse_metadata()
        self.process_filestructure()

        if 'injection' in self.data_paths.keys():
            self.get_injection_dt()

    def parse_metadata(self):
        '''Parses metadata from the experiment folder name'''
        parsed = self.exp_path.name.split('_')
        self.exp_date = parsed.pop(-1)

        try:
            self.fish_id = int(parsed[-1])
            parsed.pop(-1)
        except ValueError:
            pass

        for i, item in enumerate(parsed):
            if 'dpf' in item:
                self.age = item[:item.find('dpf')]
                parsed.pop(i)

        for i, item in enumerate(parsed):
            if 'fed' in item:
                self.feed = item
                parsed.pop(i)

        for i, item in enumerate(parsed):
            if 'mM' in item or 'uM' in item:
                self.concentration = item
                self.stimulus = parsed[i-1]
                parsed.pop(i)
                parsed.pop(i-1)

        for i, item in enumerate(parsed):
            if item == 'test':
                parsed.pop(i)

        self.genotype = parsed
              
    def process_filestructure(self):
        '''Creates a data_paths attribute with the paths to different fies'''
        with os.scandir(self.exp_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path) and entry.name.startswith('postgavage_'):
                    self.data_paths['postgavage_path'] = Path(entry.path)
                    self.exp_path = Path(entry.path)
                elif entry.name.startswith('log') and entry.name.endswith('.txt'):
                    self.data_paths['log'] = Path(entry.path)
                elif entry.name == 'injection.txt':
                    self.data_paths['injection'] = Path(entry.path)

        if 'postgavage_path' in self.data_paths.keys():
            with os.scandir(self.data_paths['postgavage_path']) as entries:
                for entry in entries:
                    if entry.name.startswith('postgavage_') and entry.name.endswith('ch1.tif'):
                        self.data_paths['raw_image'] = Path(entry.path)
                    elif entry.name.endswith('_cropped.tif'):
                        self.data_paths['cropped'] = Path(entry.path)
                    elif entry.name.endswith('raw_rotated.tif'):
                        self.data_paths['rotated'] = Path(entry.path)
                    elif entry.name.startswith('postgavage_') and entry.name.endswith('ch1.txt'):
                        self.data_paths['frametimes'] = Path(entry.path)
                    elif entry.name.startswith('raw_flipped'):
                        self.data_paths['flipped'] = Path(entry.path)
                    elif entry.name.startswith('postgavage_') and entry.name.endswith('_anatomy'):
                        self.data_paths['anatomy'] = Path(entry.path)
                    elif entry.name == 'mesmerize-batch':
                        self.data_paths['mesmerize'] = Path(entry.path)
                    elif 'C_frames' in entry.name:
                        self.data_paths['C_frames'] = Path(entry.path)
                    elif 'F_frames' in entry.name:
                        self.data_paths['F_frames'] = Path(entry.path)
                    elif entry.name == 'opts.pkl':
                        self.data_paths['opts'] = Path(entry.path)
                    elif entry.name == 'temporal.h5':
                        self.data_paths['temporal'] = Path(entry.path)

    def raw_text_frametimes_to_df(self):
        '''Returns frametimes txt file as a dataframe'''
        if hasattr(self, "frametimes_df"):
            return

        with open(self.data_paths['frametimes']) as file:
            contents = file.read()
        parsed = contents.split("\n")

        times = []
        for line in range(len(parsed) - 1):
            times.append(dt.strptime(parsed[line], "%H:%M:%S.%f").time())
        frametimes_df = pd.DataFrame(times)
        frametimes_df.rename({0: "time"}, axis=1, inplace=True)
        self.frametimes_df = frametimes_df
        return frametimes_df

    def crop_flyback(self, crop=0.075):
        '''Crops the raw image to get rid of flyback'''
        image = memmap(self.data_paths['raw_image'])
        img_path = self.data_paths['raw_image'].parent.joinpath('img_cropped.tif')
        cropped = crop_image(image, path=img_path, crop=crop)
        self.data_paths['cropped'] = img_path
        return cropped

    def rotate_image(self, img_path, angle=0):
        '''Rotates image by angle'''
        image = imread(img_path)
    
        rot_img_path = img_path.parent.joinpath("raw_rotated.tif")
        rotated_image = [rotate(img, angle=angle) for img in image]
        imwrite(
            rot_img_path, rotated_image, bigtiff=True
        )
        
        self.data_paths['rotated'] = rot_img_path

        plt.imshow(rotated_image[0])

    def flip_image(self):
        '''Flips the image horizontally'''
        image = imread(self.data_paths["raw_image"])
        flip_img_path = self.data_paths['raw_image'].parent.joinpath("raw_flipped.tif")
        flipped_image = [np.fliplr(img) for img in image]
        imwrite(
            flip_img_path, flipped_image, bigtiff=True
        )
        self.data_paths['flipped'] = flip_img_path

    def get_injection_dt(self):
        '''Gets injection time as datetime'''
        inj_file = open(self.data_paths['injection'], 'r')
        lines = inj_file.readlines()
        self.inj_time = dt.strptime(lines[0][:-1], "%H:%M:%S.%f").time()

    def align_injection_to_frames(self, frametimes):
        '''Aligns injection time to a frametimes dataframe'''
        frametimes['injection'] = None
        for i, row in frametimes.iterrows():
            frametimes.loc[i, 'injection'] = (row['time'] >= self.inj_time)
        return frametimes
    