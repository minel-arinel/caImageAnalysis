import cv2
from datetime import datetime as dt
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from scipy.ndimage import rotate
from scipy.signal import butter, filtfilt, find_peaks
from tifffile import imread, imwrite


def crop_image(image, path=None, crop=0.075):
    '''Crops image by crop. Default to 0.075 for 2P scanning flyback'''
    cropped = image[:, :, int(image.shape[2] * crop) :]
    if path is not None:
        imwrite(path, cropped, bigtiff=True)
    return cropped


def rotate_image(image, path=None, angle=0):
    '''Rotates image by angle'''
    if len(image.shape) == 3:
        rotated_image = [rotate(img, angle=angle) for img in image]
    elif len(image.shape) == 2:
        rotated_image = rotate(image, angle=angle)
    
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
        

def sort_by_peak(data, window=10):
    """
    Sorts an array of arrays by the peak values using a sliding window sum.
    Parameters:
        data (list of lists): A list where each element is an array to be sorted.
        window (int): The size of the sliding window to compute the sum. Default is 10.
    Returns:
        list of lists: The input list sorted by the peak values of each array.
    """
    sorted_data = sorted(data, key=lambda arr: np.argmax(np.convolve(arr, np.ones(window), 'valid')))

    return sorted_data


def sort_by_peak_with_indices(data, separate_array=None, window=10):
    """
    Sorts the data based on peak response times and applies the same sorting to a separate array (if provided).
    Parameters:
        data: Numpy array to sort (e.g., neuron responses across time).
        separate_array: Optional array to apply the same sorting indices to.
        window: Sliding window size for smoothing the data.
    Returns:
        sorted_data: Data sorted by peak response times.
        sorted_separate_array: The separate array sorted using the same indices (if provided).
        sorting_indices: Indices used for sorting.
    """
    # Calculate peak indices for each row using sliding window smoothing
    smoothing_window = np.ones(window)
    peak_indices = [np.argmax(np.convolve(arr, smoothing_window, 'valid')) for arr in data]

    # Get the sorting indices based on peak indices
    sorting_indices = np.argsort(peak_indices)

    # Apply sorting to data
    sorted_data = data[sorting_indices]

    # Apply sorting to separate_array if provided
    if separate_array is not None:
        sorted_separate_array = separate_array[sorting_indices]
        return sorted_data, sorted_separate_array, sorting_indices

    return sorted_data, sorting_indices


def get_last_indices(ordered_array):
	"""
	Get the indices where the value changes in an ordered array.
	If the array has only one unique value, return the last index.
	"""
	last_indices = np.where(ordered_array[:-1] != ordered_array[1:])[0]
	if len(last_indices) == 0:
		last_indices = np.array([ordered_array.shape[0]])
	return last_indices


def read_avi_as_numpy(file_path, start_frame=0, end_frame=None, **kwargs):
    """
    Reads an AVI video file and converts it to a NumPy array of grayscale frames.
    Parameters:
        file_path (str): Path to the AVI video file.
        start_frame (int, optional): Frame index to start reading from. Defaults to 0.
        end_frame (int, optional): Frame index to stop reading at. If None, reads until the end. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - meta_data (dict): Metadata of the video file.
            - frames (np.ndarray): Array of grayscale frames.
    """
    reader = imageio.get_reader(file_path, format="ffmpeg", **kwargs)
    meta_data = reader.get_meta_data()
    total_frames = meta_data['nframes']

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    frames = []
    for i, frame in enumerate(reader):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        
        # Convert RGB frame to grayscale
        gray_frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        frames.append(gray_frame)

    reader.close()

    frames = np.array(frames)
    frames = frames.astype('float32')  # Convert to float32 to save memory

    return meta_data, frames


def calculate_pixel_change(video_data):
    """
    Calculate the sum of pixel changes between consecutive frames in a video.
    Parameters:
        video_data (numpy.ndarray): A 3D array representing the video frames 
                                    with shape (num_frames, height, width).
    Returns:
        numpy.ndarray: A 1D array containing the sum of pixel changes for each frame.
    """
    pixel_change = np.abs(np.diff(video_data, axis=0))
    pixel_change_sum = np.sum(pixel_change, axis=(1, 2))
    return pixel_change_sum


def remove_artifacts(data, height, **kwargs):
    """
    Remove artifacts from the data by replacing peaks above a certain height with the average of surrounding values.
    Parameters:
        data (array-like): The input data containing potential artifacts.
        height (float): The threshold height above which data points are considered artifacts.
        **kwargs: Additional keyword arguments to pass to the find_peaks function.
    Returns:
        array-like: The cleaned data with artifacts removed.
    """
    cleaned_data = data.copy()
    peaks, _ = find_peaks(data, **kwargs)
    for peak in peaks:
        start = peak
        while start > 0 and data[start] > height:
            start -= 1
        end = peak
        while end < len(data) and data[end] > height:
            end += 1

        # Get the last non-artifact data point before the artifacts
        if start == 0:
            before_value = data[start]
        else:
            before_value = data[start - 1]

        # Get the first non-artifact data point after the artifacts
        if end == len(data):
            after_value = data[end - 1]
        else:
            after_value = data[end]

        # Calculate the average value
        average_value = (before_value + after_value) / 2

        # Replace the artifact data points with the average value
        cleaned_data[start:end] = average_value
    
    return cleaned_data


def lowpass_filter(data, cutoff, fps, order=5):
    """
    Apply a low-pass Butterworth filter to the input data.
    Parameters:
        data (array-like): The input signal to be filtered.
        cutoff (float): The cutoff frequency of the filter.
        fps (float): The sampling frequency of the input signal.
        order (int, optional): The order of the filter. Default is 5.
    Returns:
        array-like: The filtered signal.
    """
    b, a = butter(order, cutoff, btype='low', analog=False, fs=fps)
    y = filtfilt(b, a, data)
    return y