from datetime import datetime as dt
from fastplotlib import ImageWidget
from tifffile import memmap, imwrite
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from pathlib import Path, WindowsPath
import pickle


def visualize_images(imgs=list(), path=None, names=None):
    '''Visualize a list of images within the same plot using ImageWidget'''
    names = names
    add_names = False

    if names is None:
        add_names = True
        names = []

    if path is not None:
        img = memmap(path)
        imgs.append(img)

    if isinstance(imgs, list):
        for i, img in enumerate(imgs):
            img_rot = np.flip(img, axis=1)
            imgs[i] = img_rot
            if add_names:
                names.append(f'index: {i}')
    else:
        if add_names:
            names.append('image')

    iw = ImageWidget(imgs, names=names, vmin_vmax_sliders=True, cmap="gnuplot2")
    return iw


def visualize_volumes(fish, names=None):
    '''Visualize individual planes of a given fish'''
    planes = []

    for i in fish.data_paths['volumes'].keys():
        img = memmap(fish.data_paths['volumes'][i]['image'])
        planes.append(img)

    iw = visualize_images(imgs=planes, names=names)
    return iw


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

    increment = 15
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


def plot_temporal(fish, plane, indices=None, heatmap=False, key=None):
    '''Plots individual temporal components of a plane'''
    temporal_df = load_temporal(fish)
    row = temporal_df[temporal_df.plane == plane].iloc[0]
    inj_frame = row.inj_frame

    if key == 'dff':
        temporal = np.array(row.dff)
    elif key == 'norm_dff':
        temporal = np.array(row.norm_dff)
    else:
        temporal = row.temporal

    if indices is not None:
        temporal = temporal[indices]

    if not heatmap:
        fig = plt.figure(4, figsize=(10, temporal.shape[0]))
        gs = fig.add_gridspec(temporal.shape[0], hspace=0)
        axs = gs.subplots(sharex=True)

        for i, t in enumerate(temporal):
            axs[i].plot(t)
            axs[i].vlines(inj_frame, t.min(), t.max(), colors='r')

        plt.show()

    else:
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(temporal, cmap='plasma', interpolation='nearest')
        plt.vlines(inj_frame, 0, len(temporal)-1, color='r')
        plt.title(f'Plane {plane}: Temporal heatmap')
        plt.show()


def plot_temporal_volume(fish, indices=None, heatmap=False, key=None):
    '''TODO: Add line plot option for volume'''
    '''Plots individual temporal components of a volume'''
    temporal_df = load_temporal(fish)

    temporals = []
    inj_frames = []

    for i, row in temporal_df.iterrows():
        inj_frames.append(row.inj_frame)

        if key == 'dff':
            temporal = np.array(row.dff)
        elif key == 'norm_dff':
            temporal = np.array(row.norm_dff)
        else:
            temporal = row.temporal

        if indices is not None:
            temporal = temporal[indices]

        temporals.append(temporal)

    min_len = np.array([plane.shape[1] for plane in temporals]).min()

    new_temp = []
    for plane in temporals:
        if plane.shape[1] > min_len:
            inds = [plane.shape[1]-i-1 for i in range(plane.shape[1] - min_len)]
            new_plane = np.delete(plane, inds, axis=1)
            new_temp.append(new_plane)
        else:
            new_temp.append(plane)

    new_temp = np.concatenate(new_temp)

    if heatmap:
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(new_temp, cmap='plasma', interpolation='nearest')
        plt.vlines(min(inj_frames), 0, len(new_temp)-1, color='r')
        plt.title(f'Temporal heatmap')
        plt.show()

    return new_temp


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
