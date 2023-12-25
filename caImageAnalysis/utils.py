from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from scipy.ndimage import rotate
from tifffile import imread, imwrite


def crop_image(image, path=None, crop=0.075):
    '''Crops image by crop. Default to 0.075 for 2P scanning flyback'''
    cropped = image[:, :, int(image.shape[2] * crop) :]
    if path is not None:
        imwrite(path, cropped, bigtiff=True)
    return cropped


def rotate_image(image, path=None, angle=0):
    '''Rotates image by angle'''
    rotated_image = [rotate(img, angle=angle) for img in image]
    if path is not None:
        imwrite(path, rotated_image, bigtiff=True)
    return rotated_image


def save_seq_with_text(fish, plane):
    '''TODO: Save individual tif files with a stimulus text'''
    '''Saves individual tif files with the stimulus text'''
    inj_file = open(fish.data_paths['injection'], 'r')
    inj = inj_file.readline().strip()
    inj_dt = dt.strptime(inj, '%H:%M:%S.%f').time()

    try:
        frametimes = pd.read_hdf(fish.data_paths['volumes'][plane]['frametimes'])
        img = imread(fish.data_paths['volumes'][plane]['image'])
        path = fish.data_paths['volumes'][plane]['image'].parent
    except KeyError:
        frametimes = pd.read_hdf(fish.data_paths['volumes'][str(plane)]['frametimes'])
        img = imread(fish.data_paths['volumes'][str(plane)]['image'])
        path = fish.data_paths['volumes'][str(plane)]['image'].parent

    stim_path = os.path.join(path, 'stim_labeled')
    if not os.path.exists(stim_path):
        os.mkdir(stim_path)

    for i, time in enumerate(frametimes['time']):
        plt.imshow(img[i, :, :], cmap=colormap.Greys_r)
        plt.axis('off')
        if time >= inj_dt:
            plt.text(5, 5, f"{fish.stimulus} ({fish.concentration})",
                     dict(color='white', position=(20,380)))
        plt.savefig(os.path.join(stim_path, 'plane%003d.tif' % i))


def calculate_fps(frametimes_path):
    '''Calculates the imaging rate in frames per second'''
    frametimes = pd.read_hdf(frametimes_path)

    increment = 10
    test0 = 0
    test1 = increment
    while True:
        testerBool = (
                frametimes.loc[:, "time"].values[test0].minute
                == frametimes.loc[:, "time"].values[test1].minute
        )
        if testerBool:
            break
        else:
            test0 += increment
            test1 += increment

        if test0 >= len(frametimes):
            increment = increment // 2
            test0 = 0
            test1 = increment

    times = [
        float(str(f.second) + "." + str(f.microsecond))
        for f in frametimes.loc[:, "time"].values[test0:test1]
    ]
    return 1 / np.mean(np.diff(times))


def get_injection_frame(frametimes):
    '''Gets injection frame index (for custom_2P recordings)'''
    if isinstance(frametimes, str) or isinstance(frametimes, Path):
        frametimes = pd.read_hdf(frametimes)
    frametimes = frametimes.reset_index()
    inj_frame = (frametimes['injection'] == True).idxmax()
    return inj_frame


def save_pickle(val, path):
    '''Saves val as a pickle in the given path'''
    with open(path, 'wb') as p_file:
        pickle.dump(val, p_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    '''Loads a pickle file and returns the value'''
    with open(path, 'rb') as p_file:
        p = pickle.load(p_file)
    return p


def get_image_around_frame(img, frame, n_frames_pre=3, n_frames_post=3):
    '''Visualizes around a given frame number of an image
    This can be used to confirm pulse injections or twitch events
    n_frames_pre: number of frames to visualize before the given frame
    n_frames_post: number of frames to visualize after the given frame'''
    
    return img[frame-n_frames_pre:frame+n_frames_post+1]