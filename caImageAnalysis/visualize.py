from fastplotlib import ImageWidget
import matplotlib.pyplot as plt
import numpy as np
from tifffile import memmap

from caImageAnalysis.utils import load_temporal


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
