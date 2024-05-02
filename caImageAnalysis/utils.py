import cv2
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


def cross_corr(y1, y2):
    """Calculates the cross correlation and lags without normalization.

    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html

    Args:
    y1, y2: Should have the same length.

    Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
    """
    if len(y1) != len(y2):
        raise ValueError('The lengths of the inputs should be the same.')

    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(
        np.ones(len(y1)), np.ones(len(y1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2

    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)
    return max_corr, argmax_corr - shift


def fix_phase_offset(imgs, img_row, n_cols=3, show_phase_plots=True, show_updated_plots=True, show_difference_plots=True):
    '''Fixes the incorrect phase offset shifts'''
    fixed_imgs = imgs.copy()
    
    for i, img in enumerate(imgs):
        _, lag = cross_corr(img[img_row, :], img[img_row+1, :])
        
        first_rows = np.arange((img_row+1) % 2, img.shape[0], 2)
        for row in first_rows:
            fixed_imgs[i][row, :] = np.roll(img[row, :], lag, axis=0)

        # second_rows = np.arange((img_row) % 2, img.shape[0], 2)
        # for row in second_rows:
        #     fixed_imgs[i][row, :] = np.roll(img[row, :], -lag, axis=0)

        if show_phase_plots:
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].plot(img[img_row, :], label=f'row={img_row}')
            ax[0].plot(img[img_row+1, :], label=f'row={img_row+1}')
            ax[0].legend()
            ax[0].set_title('Before phase correction')

            ax[1].plot(fixed_imgs[i][img_row, :], label=f'row={img_row}')
            ax[1].plot(fixed_imgs[i][img_row+1, :], label=f'row={img_row+1}')
            ax[1].legend()
            ax[1].set_title('After phase correction')

    if show_updated_plots:
        # visualize updated images
        fig, ax = plt.subplots(int(len(imgs) / n_cols)+1, n_cols, figsize=(5*n_cols, np.ceil(len(imgs) / n_cols) * 5))

        for i in range(len(imgs)):
            row = int(i / n_cols)
            col = i % n_cols

            ax[row, col].imshow(fixed_imgs[i], vmin=0, vmax=27.9)

    if show_difference_plots:
        # visualize difference
        fig, ax = plt.subplots(int(len(imgs) / n_cols)+1, n_cols, figsize=(5*n_cols, np.ceil(len(imgs) / n_cols) * 5))

        for i in range(len(imgs)):
            row = int(i / n_cols)
            col = i % n_cols

            old_img = imgs[i] - np.mean(imgs[i])
            new_img = fixed_imgs[i] - np.mean(fixed_imgs[i])

            ax[row, col].imshow(old_img, vmin=0, vmax=27.9, alpha=0.7, cmap='Reds')
            ax[row, col].imshow(new_img, vmin=0, vmax=27.9, alpha=0.7, cmap='Greens')

    return fixed_imgs
        
