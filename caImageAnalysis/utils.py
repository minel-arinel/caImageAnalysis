from datetime import datetime as dt
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from pathlib import Path
import pickle


def crop_image(img, path=None, crop=0.075):
    '''Crops image by crop. Default to 0.075 for 2P scanning flyback'''
    cropped = img[:, :, int(img.shape[2] * crop) :]
    if path is not None:
        imwrite(path, cropped, bigtiff=True)
    return cropped


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

    increment = 5
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
    '''Gets injection frame index'''
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
    '''Loads the params pickle file under the mesmerize-batch folder as a dict'''
    with open(path, 'rb') as p_file:
        p = pickle.load(p_file)
    return p


def load_temporal(fish):
    '''Load the temporal.h5 file'''
    df = pd.read_hdf(fish.data_paths['temporal'])
    return df


def compute_dff(fish):
    '''Computes the dF/F signal for each component'''
    temporal_df = load_temporal(fish)
    temporal_df['dff'] = None

    for i, row in temporal_df.iterrows():
        inj = row.inj_frame
        dffs = []

        for comp in row.temporal:
            baseline = comp[:inj]
            f0 = np.median(baseline)
            dff = (comp - f0)/abs(f0)
            dffs.append(dff)

        temporal_df['dff'][i] = dffs

    temporal_df.to_hdf(fish.data_paths['postgavage_path'].joinpath('temporal.h5'), key='temporal')

    return temporal_df


def normalize_dff(fish):
    '''Normalizes each dF/F signal between 0 and 1'''
    temporal_df = load_temporal(fish)
    temporal_df['norm_dff'] = None

    for i, row in temporal_df.iterrows():
        norm_dffs = []

        for comp in row.dff:
            norm_dff = (comp - min(comp)) / (max(comp) - min(comp))
            norm_dffs.append(norm_dff)

        temporal_df['norm_dff'][i] = norm_dffs

    temporal_df.to_hdf(fish.data_paths['postgavage_path'].joinpath('temporal.h5'), key='temporal')

    return temporal_df
