from kneed import KneeLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import pynndescent
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import rel_entr
from scipy.stats import gaussian_kde, sem, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import auc

from caImageAnalysis.statistics import spearman_correlation_repeated_measures


def determine_baseline_frame(temporal_df, pre_frame_num=100, save_path=None):
    """
    Determine the optimal baseline frame duration using the standard error of the mean (SEM).
    Calculates the SEM for different baseline frame durations and identifies the optimal duration 
    using the KneeLocator method. Plots the SEM values against the number of frames before injection 
    and marks the ideal number of frames.
    Parameters:
        temporal_df (pd.DataFrame): DataFrame with temporal data containing 'pulse_frames' and 'raw_norm_temporal'.
        pre_frame_num (int): Maximum number of frames before injection to consider for baseline duration. Default is 100.
        save_path (str or Path, optional): Path to save the figure as determine_baseline_frame.pdf. Default is None.
    Returns:
        None
    """
    x = np.arange(1, pre_frame_num+1)

    sems = list()
    for t in x:
        traces = list()
        for _, row in temporal_df.iterrows():
            pulses = row['pulse_frames']

            for pulse in pulses:
                start_frame = pulse - t  # when the neuron traces will start
                stop_frame = pulse  # when the neuron traces will end

                for neuron in row["raw_norm_temporal"]:
                    trace = neuron[start_frame:stop_frame]
                    traces.extend(trace)

        sems.append(sem(traces))

    kn = KneeLocator(x, sems, curve='convex', direction='decreasing')

    plt.plot(x, sems)
    plt.vlines(kn.knee, 0, plt.ylim()[1], linestyles='dashed')
    plt.ylim(0)
    plt.xticks(np.arange(0, 101, 5))
    _, labels = plt.xticks()
    for label in labels[1::2]:
        label.set_visible(False)
    plt.xlabel('# of frames before injection')
    plt.ylabel('sem of trace values')
    
    if save_path:
        plt.savefig(save_path.joinpath('determine_baseline_frame.pdf'), transparent=True)
    
    plt.show()

    print(f"Ideal number of frames: {kn.knee}")


def determine_peak_frame(temporal_df, sigma=4, save_path=None):
    """
    Determine the optimal number of frames after injection where most peaks occur.
    Finds the peak frame for each trace and plots a histogram of these peak frames.
    Parameters:
        temporal_df (pd.DataFrame): DataFrame with temporal data containing 'pulse_frames' and 'raw_norm_temporal'.
        sigma (int): Standard deviation for Gaussian kernel used in smoothing the histogram. Default is 4.
        save_path (str or Path, optional): Path to save the figure as determine_peak_frame.pdf. Default is None.
    Returns:
        None
    """
    min_distance = float('inf')
    for _, row in temporal_df.iterrows():
        pulses = row['pulse_frames']
        if len(pulses) > 1:
            distances = np.diff(pulses)
            min_distance = min(min_distance, *distances)

    peak_frames = list()
    for _, row in temporal_df.iterrows():
        pulses = row['pulse_frames']
        for pulse in pulses:
            start_frame = pulse
            stop_frame = pulse + min_distance

            for neuron in row["raw_norm_temporal"]:
                trace = neuron[start_frame:stop_frame]
                peak_frame = np.argmax(trace)
                peak_frames.append(peak_frame)

    hist, bin_edges = np.histogram(peak_frames, bins=np.arange(min(peak_frames), max(peak_frames) + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.hist(peak_frames, bins=np.arange(min(peak_frames), max(peak_frames) + 1), alpha=0.5, label='Histogram')
    
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma)
    plt.plot(bin_centers, smoothed_hist, label='Smoothed Curve', color='red')

    kn = KneeLocator(bin_centers, smoothed_hist, curve='convex', direction='decreasing')
    plt.vlines(kn.knee, 0, max(smoothed_hist), linestyles='dashed', label='Plateau Point')

    plt.xlabel('Time to peak (frames)')
    plt.ylabel('Count')
    plt.title('Histogram of Time to Peak Frames with Smoothed Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path.joinpath('determine_peak_frame.pdf'), transparent=True)

    print(f"Plateau point: {kn.knee}")


def unroll_temporal_df(fish, min_pulses=3, **kwargs):
    """
    Expands the temporal_df of a fish object so that each row represents a single neuron.
    Parameters:
        fish (object): The fish object containing temporal_df and other related data.
        min_pulses (int, optional): Minimum number of pulses required to consider a neuron responsive. Default is 3.
        **kwargs: Additional keyword arguments passed to the find_stimulus_responsive function.
    Returns:
        pd.DataFrame: A DataFrame where each row represents a single neuron with associated temporal data and stimulus response information.
    Note:
        - The function assumes that fish.temporal_df contains columns: 'roi_indices', 'plane', 'raw_temporal', 'temporal', 
        'raw_norm_temporal', 'norm_temporal', 'coms', and 'pulse_frames'.
    """
    unrolled_df = pd.DataFrame(columns=['fish_id', 'plane', 'neuron', 'raw_temporal', 
                                        'temporal', 'raw_norm_temporal', 'norm_temporal',
                                        'roi_index', 'com', 'pulse_frames'])
    
    # Iterate over each row in fish.temporal_df and each neuron within that row to populate 
    # the unrolled DataFrame.
    neuron_count = -1
    for i, row in fish.temporal_df.iterrows():
        for j in range(len(row['roi_indices'])):
            unrolled_row = dict()

            neuron_count += 1

            unrolled_row['fish_id'] = int(fish.fish_id)
            unrolled_row['plane'] = row['plane']
            unrolled_row['neuron'] = neuron_count
            unrolled_row['raw_temporal'] = row['raw_temporal'][j]
            unrolled_row['temporal'] = row['temporal'][j]
            unrolled_row['raw_norm_temporal'] = row['raw_norm_temporal'][j]
            unrolled_row['norm_temporal'] = row['norm_temporal'][j]
            unrolled_row['roi_index'] = row['roi_indices'][j]
            unrolled_row['com'] = row['coms'][j]
            unrolled_row['pulse_frames'] = row['pulse_frames']

            unrolled_df = pd.concat([unrolled_df, pd.DataFrame([unrolled_row])], ignore_index=True)

    # Identify stimulus-responsive neurons and update the DataFrame with response information.
    stim_responsive, activated, suppressed, pulse_responses = find_stimulus_responsive(fish, **kwargs)
    
    unrolled_df['responsive'] = False
    unrolled_df['activated'] = None
    unrolled_df['suppressed'] = None
    unrolled_df['pulse_response'] = None

    for i, neuron in enumerate(stim_responsive):
        unrolled_df.at[neuron, 'pulse_response'] = pulse_responses[i]

        if len(pulse_responses[i]) >= min_pulses:
            unrolled_df.loc[neuron, 'responsive'] = True

            if neuron in activated:
                unrolled_df.loc[neuron, 'activated'] = True
                unrolled_df.loc[neuron, 'suppressed'] = False
            
            elif neuron in suppressed:
                unrolled_df.loc[neuron, 'suppressed'] = True
                unrolled_df.loc[neuron, 'activated'] = False

    # Save the unrolled DataFrame to an HDF5 file.
    unrolled_df.to_hdf(fish.exp_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal')

    # Re-process the file structure.
    fish.process_bruker_filestructure()

    return unrolled_df


def find_stimulus_responsive(fish, pre_frame_num=15, post_frame_num=5, peak_threshold=None, min_threshold=None, key=None, normalize=False, 
                             normalize_by_first=False, pulse_key="pulse_frames"):
    """
    Identifies stimulus-responsive neurons in a given fish dataset.
    Parameters:
        fish (object): Fish object containing temporal data in a pandas DataFrame.
        pre_frame_num (int): Number of frames before the pulse to consider as baseline (default is 15).
        post_frame_num (int): Number of frames after the pulse to analyze (default is 5).
        peak_threshold (float, optional): Custom threshold for peak fluorescence intensity 
            to classify a neuron as activated. If None, a default threshold is used.
        min_threshold (float, optional): Custom threshold for minimum fluorescence intensity 
            to classify a neuron as suppressed. If None, a default threshold is used.
        key (str, optional): Key to access neuron data in the DataFrame (default is 'raw_norm_temporal').
        normalize (bool): If True, normalizes fluorescence intensity using the pre-stimulus 
            period as baseline for each pulse.
        normalize_by_first (bool): If True, normalizes fluorescence intensity using the 
            pre-stimulus period of the first pulse as baseline for all pulses.
        pulse_key (str): Key to access pulse frame data in the DataFrame (default is 'pulse_frames').
    Returns:
        tuple: A tuple containing:
            - stim_responsive_neurons (list): Indices of neurons classified as stimulus-responsive.
            - activated_neurons (list): Indices of neurons classified as activated.
            - suppressed_neurons (list): Indices of neurons classified as suppressed.
            - pulse_responses (list): List of neuron responses to individual pulses. Each entry 
                is a list of tuples, where each tuple contains the pulse number and response type 
                (0 for suppression, 1 for activation).
    Raises:    
        ValueError: If both `normalize` and `normalize_by_first` are set to True.
    Notes:
        - A neuron is classified as activated if its peak response in the post-stimulus period 
            exceeds 2 times the standard deviation of the pre-stimulus period.
        - A neuron is classified as suppressed if its minimum response in the post-stimulus 
            period is below 2 times the standard deviation of the pre-stimulus period.
    """
    if key is None:
        key = 'raw_norm_temporal'

    if normalize and normalize_by_first:
        raise ValueError('normalize and normalize_by_first cannot both be True. Pick one method to normalize.')

    neurons = list()
    pulse_frames=list()

    for i, row in fish.temporal_df.iterrows():
        neurons.extend(row[key])

        for j in range(len(row[key])):  # add pulse frames for each neuron in each plane
            pulse_frames.append(row[pulse_key])

    stim_responsive_neurons = list()  # list of neuron indices that are selected to be stimulus responsive
    activated_neurons = list()  # list of neuron indices that are activated to the injection on average
    suppressed_neurons = list()  # list of neuron indices that are suppressed by the injection on average
    pulse_responses = list()

    for i, neuron in enumerate(neurons):
        pulses = pulse_frames[i]
        traces = list()

        for p, pulse in enumerate(pulses):
            start_frame = pulse - pre_frame_num  # when the neuron traces will start
            stop_frame = pulse + post_frame_num  # when the neuron traces will end

            trace = neuron[start_frame:stop_frame+1]

            if normalize or (normalize_by_first and p == 0):
                baseline = np.median(neuron[start_frame:pulse])
                trace = (trace - baseline)/baseline
            elif normalize_by_first and p != 0:
                trace = (trace - baseline)/baseline

            traces.append(trace)
        
        if len(np.array(traces).shape) == 1:
            avg_trace = np.array(traces)
        else:
            avg_trace = np.array(traces).mean(axis=0)

        # To determine if a neuron is stimulus responsive, we will first calculate
        # the standard deviation of "pre".
        pre_stdev = np.array(avg_trace)[:pre_frame_num].std()

        response_count = 0  # to calculate how many injections the neuron responds to
        responsive = False

        # if the neuron responds to specific pulses, store which pulses those are in a list
        # each item is a tuple, in the format (pulse number, 0 or 1)
        # 0 means it's suppressed by the pulse, 1 means it's activated by the pulse
        neuron_pulse_response = list()  

        # Activated neurons: If the peak response in "post" is bigger
        # than 2 times the "pre" standard deviation, the neuron is stimulus 
        # responsive
        activated_thresh = np.median(np.array(avg_trace)[:pre_frame_num]) + (2 * pre_stdev)

        # suppressed neurons: If the minimum response in "post" is smaller
        # than 2 times the "pre" standard deviation, the neuron is stimulus 
        # responsive
        suppressed_thresh = np.median(np.array(avg_trace)[:pre_frame_num]) - (2 * pre_stdev)

        if check_if_activated(avg_trace, activated_thresh, pre_frame_num=pre_frame_num, peak_threshold=peak_threshold):
            responsive = True
            stim_responsive_neurons.append(i)
            activated_neurons.append(i)
            print(f'neuron {i} is activated')

            for t, trace in enumerate(traces):
                # now let's determine how many of the stimuli individual neurons are responding to
                pre_stdev = np.array(trace)[:pre_frame_num].std()
                activated_thresh = np.median(np.array(trace)[:pre_frame_num]) + (2 * pre_stdev)

                if check_if_activated(trace, activated_thresh, pre_frame_num=pre_frame_num, peak_threshold=peak_threshold):
                    response_count += 1
                    neuron_pulse_response.append((t+1, 1))
                    print(f'neuron {i} responds to stimulus {t+1} (activated)')

        elif check_if_suppressed(avg_trace, suppressed_thresh, pre_frame_num=pre_frame_num, min_threshold=min_threshold):
            responsive = True
            stim_responsive_neurons.append(i)
            suppressed_neurons.append(i)
            print(f'neuron {i} is suppressed')

            for t, trace in enumerate(traces):
                # now let's determine how many of the stimuli individual neurons are responding to
                pre_stdev = np.array(trace)[:pre_frame_num].std()
                suppressed_thresh = np.median(np.array(trace)[:pre_frame_num]) - (2 * pre_stdev)

                if check_if_suppressed(trace, suppressed_thresh, pre_frame_num=pre_frame_num, min_threshold=min_threshold):
                    response_count += 1
                    neuron_pulse_response.append((t+1, 0))
                    print(f'neuron {i} responds to stimulus {t+1} (suppressed)')

        if responsive:
            print(f'neuron {i} responds to {(response_count/len(traces))*100}% of injections\n')
            pulse_responses.append(neuron_pulse_response)

    print(f'{len(stim_responsive_neurons)} out of {len(neurons)} neurons is stimulus responsive: {len(stim_responsive_neurons)/len(neurons)*100}%')
    print(f'number of suppressed neurons: {len(suppressed_neurons)}')
    
    if len(stim_responsive_neurons) != 0:
        print(f'% of suppressed neurons: {len(suppressed_neurons)/len(stim_responsive_neurons)*100}')
    else:
        print(f'% of suppressed neurons: 0%')

    print(f'number of activated neurons: {len(activated_neurons)}')

    if len(stim_responsive_neurons) != 0:
        print(f'% of activated neurons: {len(activated_neurons)/len(stim_responsive_neurons)*100}')
    else:
        print(f'% of activated neurons: 0%')

    return stim_responsive_neurons, activated_neurons, suppressed_neurons, pulse_responses


def check_if_activated(trace, threshold, pre_frame_num=15, peak_threshold=None):
    """
    Checks if a neural trace is activated.
    Parameters:
        trace (array-like): Neural trace data.
        threshold (float): Activation threshold.
        pre_frame_num (int): Number of frames before the pulse. Default is 15.
        peak_threshold (float, optional): Minimum peak response threshold. Default is 20% above baseline.
    Returns:
        bool: True if the neuron is activated, False otherwise.
    """
    peak_response = trace[pre_frame_num:].max()
    mdn_baseline = np.median(trace[:pre_frame_num])

    if peak_threshold is None and mdn_baseline != 0:
        peak_threshold = abs(mdn_baseline * 0.2) + mdn_baseline
    elif peak_threshold is None and mdn_baseline == 0:
        peak_threshold = 0.2

    return peak_response > threshold and peak_response > peak_threshold


def check_if_suppressed(trace, threshold, pre_frame_num=15, min_threshold=None):
    """
    Checks if a neural trace is suppressed.
    Parameters:
        trace (array-like): Neural trace data.
        threshold (float): Suppression threshold.
        pre_frame_num (int): Number of frames before the pulse. Default is 15.
        min_threshold (float, optional): Minimum suppression threshold. Default is 20% below baseline.
    Returns:
        bool: True if the neuron is suppressed, False otherwise.
    """
    min_response = trace[pre_frame_num:].min()
    mdn_baseline = np.median(trace[:pre_frame_num])

    if min_threshold is None and mdn_baseline != 0:
        min_threshold = abs(mdn_baseline * 0.2) - mdn_baseline
    elif min_threshold is None and mdn_baseline == 0:
        min_threshold = -0.2

    return min_response < threshold and min_response < min_threshold


def calculate_percentage_metric_neurons_per_fish(df, metrics, values, filterby=None, inverse=False):
    """
    Calculate the percentage of neurons per fish that meet specific metric conditions, optionally filtering by specified columns.
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        metrics (list of str): List of column names in df to evaluate.
        values (list of any): List of values that the rows need to match in the corresponding metric columns.
        filterby (list, optional): List of columns to filter the data by. Defaults to None.
        inverse (bool, optional): If True, calculate percentage where metrics are not equal to values. Defaults to False.
    Returns:
        pd.DataFrame: DataFrame with percentage of metric neurons per fish for each filter group.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    if not isinstance(values, list):
        values = [values]
    
    if len(metrics) != len(values):
        raise ValueError("The length of metrics and values must be the same.")

    if filterby is not None:
        results = dict()
    else:
        results = list()

    for fish_id in df.fish_id.unique():
        fish_df = df[df.fish_id == fish_id]

        if filterby:
            filter_groups = fish_df.groupby(filterby).size().reset_index()

            for _, row in filter_groups.iterrows():
                conditions = [row[col] for col in filterby]

                filters = list()
                for col, cond in zip(filterby, conditions):
                    if isinstance(cond, str):
                        filters.append(f"(fish_df['{col}'] == '{cond}')")
                    else:
                        filters.append(f"(fish_df['{col}'] == {cond})")

                sub_fish_df = fish_df[eval(" & ".join(filters))]
                label = " - ".join([str(cond) for cond in conditions])
                
                if label not in results:
                    results[label] = list()

                if not inverse:
                    condition = np.all([sub_fish_df[metric] == value if value is not None else sub_fish_df[metric].isnull() for metric, value in zip(metrics, values)], axis=0)
                else:
                    condition = np.all([sub_fish_df[metric] != value if value is not None else sub_fish_df[metric].notnull() for metric, value in zip(metrics, values)], axis=0)

                percentage = len(sub_fish_df[condition]) / len(fish_df) * 100
                results[label].append(percentage)
        else:
            if not inverse:
                condition = np.all([fish_df[metric] == value if value is not None else fish_df[metric].isnull() for metric, value in zip(metrics, values)], axis=0)
            else:
                condition = np.all([fish_df[metric] != value if value is not None else fish_df[metric].notnull() for metric, value in zip(metrics, values)], axis=0)

            percentage = len(fish_df[condition]) / len(fish_df) * 100
            results.append(percentage)

    if filterby is not None:
        results = dict([(k, pd.Series(v)) for k, v in results.items()])
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(results, columns=["Percentage"])
    
    print(results_df)
    results_df.to_clipboard()
    return results_df
    

def get_traces(df, pre_frame_num=15, post_frame_num=13, normalize=False, 
               normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, 
               return_col=None, specific_pulse=None, pulse_key="pulse_frames"):
    """
    Extracts neural traces around pulses from a DataFrame.
    Parameters:
        df (pd.DataFrame or pd.Series): DataFrame or Series containing neuron data.
        pre_frame_num (int): Number of frames before the pulse.
        post_frame_num (int): Number of frames after the pulse.
        normalize (bool): If True, normalizes traces using the pre-injection period.
        normalize_by_first (bool): If True, normalizes using the first pulse's pre-injection period.
        key (str): Column name to extract traces from.
        only_responsive (bool): If True, extracts traces only for responsive pulses.
        return_col (str or list, optional): If provided, returns additional column values for the traces.
        specific_pulse (int, optional): If provided, extracts traces only for the specified pulse.
        pulse_key (str): Column name containing pulse frame information. Default is "pulse_frames".
    Returns:
        tuple: 
            - x (np.ndarray): x-axis values (in frames).
            - traces (list): List of extracted traces.
            - return_col_list or return_col_lists (optional): Additional column values if return_col is provided.
    """
    x = np.arange(0-pre_frame_num, 0+post_frame_num+1)
    traces = list()

    if return_col is not None:
        if isinstance(return_col, list):
            return_col_lists = {col: list() for col in return_col}
        else:
            return_col_list = list()

    if isinstance(df, pd.Series):
        df = df.to_frame().T

    for _, neuron in df.iterrows():
        pulses = neuron[pulse_key]

        if len(pulses) > 0:
            if only_responsive:
                responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
                pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, suppressed

                if normalize_by_first:
                    baseline = 0

                for i, pulse in enumerate(responsive_pulses):
                    if (pulse_activity[i] == 1 and neuron['activated'] == True) or (pulse_activity[i] == 0 and neuron['suppressed'] == True):
                        start_frame = pulses[pulse-1] - pre_frame_num  # when the neuron traces will start
                        stop_frame = pulses[pulse-1] + post_frame_num  # when the neuron traces will end

                        trace = neuron[key][start_frame:stop_frame+1]

                        if normalize:
                            baseline = np.median(neuron[key][start_frame:pulses[pulse-1]])
                            trace = (trace - baseline) / baseline
                        elif normalize_by_first and i == 0:
                            baseline = np.median(neuron[key][start_frame:pulses[pulse-1]])
                            trace = (trace - baseline) / baseline
                        elif normalize_by_first:
                            trace = (trace - baseline) / baseline
                        
                        if specific_pulse is not None and pulse != specific_pulse:
                            continue
                        
                        traces.append(trace)

                        if return_col is not None:
                            if isinstance(return_col, list):
                                for col in return_col:
                                    return_col_lists[col].append(neuron[col])
                            else:
                                return_col_list.append(neuron[return_col])

            else:
                if normalize_by_first:
                    baseline = 0

                for i, pulse in enumerate(pulses):
                    start_frame = pulse - pre_frame_num  # when the neuron traces will start
                    stop_frame = pulse + post_frame_num  # when the neuron traces will end

                    trace = neuron[key][start_frame:stop_frame+1]

                    if normalize:
                        baseline = np.median(neuron[key][start_frame:pulse])
                        trace = (trace - baseline) / baseline
                    elif normalize_by_first and i == 0:
                        baseline = np.median(neuron[key][start_frame:pulse])
                        trace = (trace - baseline) / baseline
                    elif normalize_by_first:
                        trace = (trace - baseline) / baseline

                    if specific_pulse is not None and i + 1 != specific_pulse:
                        continue
                    
                    traces.append(trace)

                    if return_col is not None:
                        if isinstance(return_col, list):
                            for col in return_col:
                                return_col_lists[col].append(neuron[col])
                        else:
                            return_col_list.append(neuron[return_col])

    if return_col is not None:
        if isinstance(return_col, list):
            return x, traces, return_col_lists
        else:
            return x, traces, return_col_list
    else:
        return x, traces
    

def calculate_time_at_half_maximum(x, trace, suppressed=False, fps=1, frame_interval=list()):
    """
    Returns the time at half maximum of the given trace.
    Parameters:
        x (array-like): The x values (e.g., time points).
        trace (array-like): The y values (e.g., intensity or response values).
        suppressed (bool): If True, finds the minimum value for suppressed traces.
        fps (int): Frames per second for time conversion.
        frame_interval (list): List containing the start and end frame numbers to consider for peak extraction.
    Returns:
        float: The interpolated time at half maximum.
    """  
    # Find the index of the peak
    if len(frame_interval) != 0:
        start_idx = frame_interval[0]
        stop_idx = frame_interval[1] if len(frame_interval) > 1 else len(trace)
        if not suppressed:
            peak_idx = np.argmax(trace[start_idx:stop_idx + 1]) + start_idx
        else:
            peak_idx = np.argmin(trace[start_idx:stop_idx + 1]) + start_idx
    else:
        if not suppressed:
            peak_idx = np.argmax(trace)
        else:
            peak_idx = np.argmin(trace)
    
    # Get the maximum value and half maximum value
    max_value = trace[peak_idx]
    half_max = max_value / 2
    
    # Consider only the segment before the peak
    if len(frame_interval) != 0:
        pre_peak_trace = trace[start_idx:peak_idx + 1]
        pre_peak_x = x[start_idx:peak_idx + 1]
    else:
        pre_peak_trace = trace[:peak_idx + 1]
        pre_peak_x = x[:peak_idx + 1]

    # Find where the trace crosses the half maximum before the peak
    if not suppressed:
        above_half_max = pre_peak_trace >= half_max
    else:
        above_half_max = pre_peak_trace <= half_max

    crossing_indices = np.where(np.diff(above_half_max.astype(int)) != 0)[0]

    if len(crossing_indices) == 0:
        return 0

    # Interpolate to find a more accurate time at half maximum
    x1, x2 = pre_peak_x[crossing_indices[0]], pre_peak_x[crossing_indices[0] + 1]
    y1, y2 = pre_peak_trace[crossing_indices[0]], pre_peak_trace[crossing_indices[0] + 1]

    interpolator = interp1d([y1, y2], [x1, x2], kind='linear')
    time_at_half_max = interpolator(half_max)

    return time_at_half_max / fps


def calculate_time_to_decay(x, trace, suppressed=False, fps=1, frame_interval=list(), non_sig_cutoff=2):
    """
    Calculate the time it takes for a signal to decay to baseline levels.
    Parameters:
        trace (array-like): The signal trace data.
        suppressed (bool): If True, find the minimum peak instead of the maximum. Default is False.
        fps (int): Frames per second of the trace data. Default is 1.
        frame_interval (list): List containing start and optional stop indices for peak search. Default is an empty list.
        non_sig_cutoff (int): Number of consecutive non-significant frames to confirm decay. Default is 2.
    Returns:
        float: Time to decay in seconds.
    """
    # Find the index of the peak
    if len(frame_interval) != 0:
        start_idx = frame_interval[0]
        stop_idx = frame_interval[1] if len(frame_interval) > 1 else len(trace)
        if not suppressed:
            peak_idx = np.argmax(trace[start_idx:stop_idx + 1]) + start_idx
        else:
            peak_idx = np.argmin(trace[start_idx:stop_idx + 1]) + start_idx
    else:
        if not suppressed:
            peak_idx = np.argmax(trace)
        else:
            peak_idx = np.argmin(trace)

    if len(frame_interval) != 0:
        baseline_distribution = trace[:start_idx]
    else:
        baseline_distribution = trace[:peak_idx]

    final_sig_frame = -1  # keep track of the last significant frame

    for t in range(peak_idx + 1, len(trace)):
        arr = trace[t]
        frame = t - peak_idx

        _, p_value = wilcoxon(baseline_distribution - arr) 

        if p_value < 0.05:
            final_sig_frame = frame

        elif final_sig_frame != -1 and frame - final_sig_frame == non_sig_cutoff:
            return final_sig_frame / fps

    if final_sig_frame != -1:
        return final_sig_frame / fps
    elif final_sig_frame == -1:
        return 1 / fps
    

def peak_func(x, trace, suppressed=False, fps=1, frame_interval=list()):
    """
    Calculate the peak value of the trace.
    Parameters:
        x (array-like): The x values (e.g., time points).
        trace (array-like): The y values (e.g., intensity or response values).
        suppressed (bool): If True, find the minimum peak instead of the maximum.
        fps (int): Frames per second for time conversion.
        frame_interval (list): List containing the start and end frame numbers to consider for peak extraction.
    Returns:
        float: The peak value of the trace.
    """
    if len(frame_interval) != 0:
        start_idx = frame_interval[0]
        stop_idx = frame_interval[1] if len(frame_interval) > 1 else len(trace)
        trace = trace[start_idx:stop_idx + 1]
    return np.min(trace) if suppressed else np.max(trace)


def AUC_func(x, trace, suppressed=False, fps=1, frame_interval=list()):
    """
    Calculate the area under the curve (AUC) of the trace.
    Parameters:
        x (array-like): The x values (e.g., time points).
        trace (array-like): The y values (e.g., intensity or response values).
        suppressed (bool): If True, find the minimum peak instead of the maximum.
        fps (int): Frames per second for time conversion.
        frame_interval (list): List containing the start and end frame numbers to consider for AUC calculation.
    Returns:
        float: The area under the curve of the trace.
    """
    if len(frame_interval) != 0:
        start_idx = frame_interval[0]
        stop_idx = frame_interval[1] if len(frame_interval) > 1 else len(trace)
        x = x[start_idx:stop_idx + 1]
        trace = trace[start_idx:stop_idx + 1]
    return auc(x, trace)


def thm_func(x, trace, suppressed=False, fps=1, frame_interval=list()):
    """
    Calculate the time at half maximum (thm) of the trace.
    Parameters:
        x (array-like): The x values (e.g., time points).
        trace (array-like): The y values (e.g., intensity or response values).
        suppressed (bool): If True, find the minimum value for suppressed traces.
        fps (int): Frames per second for time conversion.
        frame_interval (list): List containing the start and end frame numbers to consider for peak extraction.
    Returns:
        float: The time at half maximum of the trace.
    """
    return calculate_time_at_half_maximum(x, trace, suppressed=suppressed, fps=fps, frame_interval=frame_interval)


def decay_func(x, trace, suppressed=False, fps=1, frame_interval=list()):
    """
    Calculate the time to decay of the trace.
    Parameters:
        x (array-like): The x values (e.g., time points).
        trace (array-like): The y values (e.g., intensity or response values).
        suppressed (bool): If True, find the minimum value for suppressed traces.
        fps (int): Frames per second for time conversion.
        frame_interval (list): List containing the start and end frame numbers to consider for peak extraction.
    Returns:
        float: The time to decay of the trace.
    """
    return calculate_time_to_decay(x, trace, suppressed=suppressed, fps=fps, frame_interval=frame_interval)


def extract_traces_and_apply_function(df, func, flip_suppressed=True, filterby=None, frame_interval=list(), fps=1, return_col=None, **kwargs):
    """
    Extracts traces from a DataFrame, applies a specified function to each trace, and optionally filters by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        func (function): The function to apply to each trace.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for peak extraction. Defaults to an empty list.
        fps (int, optional): Frames per second for time conversion. Defaults to 1.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        dict: A dictionary containing the results of the applied function for each group or the entire DataFrame.
        dict (optional): A dictionary containing additional column values if return_col is specified.
    """
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        results = dict()

        if return_col is not None:
            other_col_values = dict()

        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            
            if flip_suppressed:
                x, traces, suppressed = get_traces(subdf, return_col='suppressed', **kwargs)
            elif return_col is not None:
                x, traces, other_col = get_traces(subdf, return_col=return_col, **kwargs)
                other_col_values[" - ".join([str(cond) for cond in conditions])] = other_col
            else:
                x, traces = get_traces(subdf, **kwargs)

            if flip_suppressed:
                results[" - ".join([str(cond) for cond in conditions])] = [
                    func(x, trace, suppressed=suppressed[i], fps=fps, frame_interval=frame_interval) 
                    for i, trace in enumerate(traces)
                ]
            else:
                results[" - ".join([str(cond) for cond in conditions])] = [
                    func(x, trace, fps=fps, frame_interval=frame_interval) 
                    for trace in traces
                ]

        results = dict([(k, pd.Series(v)) for k, v in results.items()])

    else:
        if flip_suppressed:
            x, traces, suppressed = get_traces(df, return_col='suppressed', **kwargs)
        elif return_col is not None:
            x, traces, other_col_values = get_traces(df, return_col=return_col, **kwargs)
        else:
            x, traces = get_traces(df, **kwargs)

        if flip_suppressed:
            results = [
                func(x, trace, suppressed=suppressed[i], fps=fps, frame_interval=frame_interval) 
                for i, trace in enumerate(traces)
            ]
        else:
            results = [
                func(x, trace, fps=fps, frame_interval=frame_interval) 
                for trace in traces
            ]

    pd.DataFrame(results).to_clipboard()
    if return_col is not None:
        return results, other_col_values
    else:
        return results


def get_peaks(df, flip_suppressed=True, filterby=None, frame_interval=list(), return_col=None, **kwargs):
    """
    Extracts peak values from traces in a DataFrame, optionally filtering by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for peak extraction. Defaults to an empty list.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        dict: A dictionary containing the peak values for each group or the entire DataFrame.
        dict (optional): A dictionary containing additional column values if return_col is specified.
    """

    return extract_traces_and_apply_function(df, peak_func, flip_suppressed, filterby, frame_interval, return_col=return_col, **kwargs)


def get_AUCs(df, flip_suppressed=True, filterby=None, frame_interval=list(), return_col=None, **kwargs):
    """
    Extracts area under the curve (AUC) values from traces in a DataFrame, optionally filtering by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for AUC calculation. Defaults to an empty list.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        dict: A dictionary containing the AUC values for each group or the entire DataFrame.
        dict (optional): A dictionary containing additional column values if return_col is specified.
    """
    return extract_traces_and_apply_function(df, AUC_func, flip_suppressed, filterby, frame_interval, return_col=return_col, **kwargs)


def get_times_at_half_maximum(df, flip_suppressed=True, filterby=None, frame_interval=list(), fps=1, **kwargs):
    """
    Calculate times at half maximum (THM) for each neuron in the dataframe.
    Parameters:
        df (pd.DataFrame): DataFrame containing neuron data.
        flip_suppressed (bool): If True, finds the minimum value for suppressed neurons.
        filterby (list): List of columns to filter the DataFrame by.
        frame_interval (list): Indices to calculate peaks from (start and stop index, inclusive).
        fps (int): Frames per second for time conversion.
        **kwargs: Additional arguments for get_traces function.
    Returns:
        dict: Dictionary with THM times for each neuron group or individual neuron.
    """

    return extract_traces_and_apply_function(df, thm_func, flip_suppressed, filterby, frame_interval, fps=fps, **kwargs)


def get_times_to_decay(df, flip_suppressed=True, filterby=None, frame_interval=list(), fps=1, **kwargs):
    """
    TODO: non_sig_cutoff cannot be changed outside of the function. Fix this.
    Calculate times to decay for each neuron in the dataframe.
    Parameters:
        df (pd.DataFrame): DataFrame containing neuron data.
        flip_suppressed (bool): If True, finds the minimum value for suppressed neurons.
        filterby (list): List of columns to filter the DataFrame by.
        frame_interval (list): Indices to calculate peaks from (start and stop index, inclusive).
        fps (int): Frames per second for time conversion.
        **kwargs: Additional arguments for get_traces function.
    Returns:
        dict: Dictionary with decay times for each neuron group or individual neuron.
    """

    return extract_traces_and_apply_function(df, decay_func, flip_suppressed, filterby, frame_interval, fps=fps, **kwargs)
    

def extract_traces_and_apply_function_across_pulses(df, func, flip_suppressed=True, filterby=None, frame_interval=list(), fps=1, return_col=None, **kwargs):
    """
    Extracts traces from a DataFrame, categorizes them based on pulses, applies a specified function to each categorized trace, 
    and optionally filters by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        func (function): The function to apply to each trace.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for peak extraction. Defaults to an empty list.
        fps (int, optional): Frames per second for time conversion. Defaults to 1.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        dict: A dictionary containing the results of the applied function for each pulse and optionally additional column values if return_col is specified.
    """
    return_cols = ["pulse_frames"]
            
    if flip_suppressed:
        return_cols.append('suppressed')
    if return_col is not None:
        return_cols.append(return_col)

    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        results = dict()

        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]

            x, traces, return_col_list = get_traces(subdf, return_col=return_cols, **kwargs)
            
            pulse_frames = return_col_list['pulse_frames']
            categorized_traces = {i: [] for i in range(1, 6)}

            i = 0
            while i < len(pulse_frames):
                neuron_pulse_frames = pulse_frames[i]
                num_pulses = len(neuron_pulse_frames)
                for pulse in range(num_pulses):
                    categorized_traces[pulse + 1].append(traces[i])
                    i += 1

            for pulse in categorized_traces:
                categorized_traces[pulse] = np.array(categorized_traces[pulse])

            if flip_suppressed:
                results[" - ".join([str(cond) for cond in conditions])] = {
                    pulse: [func(x, trace, suppressed=suppressed, fps=fps, frame_interval=frame_interval) 
                            for trace, suppressed in zip(categorized_traces[pulse], return_col_list['suppressed'])]
                    for pulse in categorized_traces
                }
            else:
                results[" - ".join([str(cond) for cond in conditions])] = {
                    pulse: [func(x, trace, fps=fps, frame_interval=frame_interval) 
                            for trace in categorized_traces[pulse]]
                    for pulse in categorized_traces
                }

    else:
        x, traces, return_col_list = get_traces(df, return_col=return_cols, **kwargs)

        pulse_frames = return_col_list['pulse_frames']
        categorized_traces = {i: [] for i in range(1, 6)}

        i = 0
        while i < len(pulse_frames):
            neuron_pulse_frames = pulse_frames[i]
            num_pulses = len(neuron_pulse_frames)
            for pulse in range(num_pulses):
                categorized_traces[pulse + 1].append(traces[i])
                i += 1

        for pulse in categorized_traces:
            categorized_traces[pulse] = np.array(categorized_traces[pulse])

        if flip_suppressed:
            results = {
                pulse: [func(x, trace, suppressed=suppressed, fps=fps, frame_interval=frame_interval) 
                        for trace, suppressed in zip(categorized_traces[pulse], return_col_list['suppressed'])]
                for pulse in categorized_traces
            }
        else:
            results = {
                pulse: [func(x, trace, fps=fps, frame_interval=frame_interval) 
                        for trace in categorized_traces[pulse]]
                for pulse in categorized_traces
            }

    # Organize results into a DataFrame with multi-level columns
    max_values_per_pulse = {pulse: 0 for pulse in range(1, 6)}
    
    if filterby is not None:
        for _, pulses in results.items():
            for pulse, values in pulses.items():
                max_values_per_pulse[pulse] = max(max_values_per_pulse[pulse], len(values))
    else:
        for pulse, values in results.items():
            max_values_per_pulse[pulse] = max(max_values_per_pulse[pulse], len(values))
    
    max_columns = max(max_values_per_pulse.values())
    print(f"Number of columns: {max_columns}")

    organized_results = {pulse: [] for pulse in range(1, 6)}
    
    if filterby is not None:
        for condition, pulses in results.items():
            for pulse, values in pulses.items():
                if len(values) < max_columns:
                    values.extend([None] * (max_columns - len(values)))
                organized_results[pulse].append((condition, values))
    else:
        for pulse, values in results.items():
            if len(values) < max_columns:
                values.extend([None] * (max_columns - len(values)))
            organized_results[pulse].append(('Values', values))
    
    if filterby is not None:
        columns = pd.MultiIndex.from_product(
            [list(results.keys()), range(max_columns)], 
            names=['Condition', 'Index']
        )
    else:
        columns = pd.MultiIndex.from_product(
            [['Values'], range(max_columns)], 
            names=['Condition', 'Index']
        )
    organized_df = pd.DataFrame(index=range(1, 6), columns=columns)
    
    for pulse, data in organized_results.items():
        for condition, values in data:
            organized_df.loc[pulse, condition] = values
    
    organized_df.to_clipboard()

    if return_col is not None:
        return organized_df, return_col_list[return_col]
    else:
        return organized_df


def get_peaks_across_pulses(df, flip_suppressed=True, filterby=None, frame_interval=list(), return_col=None, **kwargs):
    """
    Extracts peak values from traces in a DataFrame, categorizes them based on pulses, and optionally filters by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for peak extraction. Defaults to an empty list.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        pd.DataFrame: A DataFrame containing the peak values for each pulse and optionally additional column values if return_col is specified.
    """
    
    return extract_traces_and_apply_function_across_pulses(df, peak_func, flip_suppressed, filterby, frame_interval, return_col=return_col, **kwargs)


def get_AUCs_across_pulses(df, flip_suppressed=True, filterby=None, frame_interval=list(), return_col=None, **kwargs):
    """
    Extracts area under the curve (AUC) values from traces in a DataFrame, categorizes them based on pulses, and optionally filters by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for AUC calculation. Defaults to an empty list.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        pd.DataFrame: A DataFrame containing the AUC values for each pulse and optionally additional column values if return_col is specified.
    """
    return extract_traces_and_apply_function_across_pulses(df, AUC_func, flip_suppressed, filterby, frame_interval, return_col=return_col, **kwargs)


def get_thms_across_pulses(df, flip_suppressed=True, filterby=None, frame_interval=list(), return_col=None, **kwargs):
    """
    Extracts time at half-maximum (thm) values from traces in a DataFrame, categorizes them based on pulses, and optionally filters by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for peak extraction. Defaults to an empty list.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        pd.DataFrame: A DataFrame containing the thm values for each pulse and optionally additional column values if return_col is specified.
    """
    return extract_traces_and_apply_function_across_pulses(df, thm_func, flip_suppressed, filterby, frame_interval, return_col=return_col, **kwargs)


def get_decays_across_pulses(df, flip_suppressed=True, filterby=None, frame_interval=list(), return_col=None, **kwargs):
    """
    Extracts time to decay values from traces in a DataFrame, categorizes them based on pulses, and optionally filters by specified columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        flip_suppressed (bool, optional): If True, flips the suppressed traces. Defaults to True.
        filterby (list, optional): List of column names to filter the DataFrame by. Defaults to None.
        frame_interval (list, optional): List containing the start and end frame numbers to consider for peak extraction. Defaults to an empty list.
        return_col (str, optional): Column name to return additional values from. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the get_traces function.
    Returns:
        pd.DataFrame: A DataFrame containing the time to decay values for each pulse and optionally additional column values if return_col is specified.
    """
    return extract_traces_and_apply_function_across_pulses(df, decay_func, flip_suppressed, filterby, frame_interval, return_col=return_col, **kwargs)


def label_monotonic_neurons(df, alpha=0.05, save_path=None, **kwargs):
    """
    Labels neurons in the DataFrame as 'integrating', 'habituating', or 'monotonic' based on their monotonicity.
    Parameters:
        df (pd.DataFrame): DataFrame containing neuron data with a 'responsive' column.
        alpha (float, optional): Significance level for monotonicity test. Default is 0.05.
        save_path (str or Path, optional): Path to save the labeled DataFrame as an HDF5 file. Default is None.
        **kwargs: Additional arguments passed to get_peaks_across_pulses.
    Returns:
        pd.DataFrame: DataFrame with an added 'monotonic' column indicating the type of monotonicity.
    """
    df["monotonic"] = None
    
    for i, row in df.iterrows():
        if row["responsive"]:
            peaks = get_peaks_across_pulses(row, **kwargs)
            [corr], [p_value] = spearman_correlation_repeated_measures(peaks["Values"], verbose=False)

            if row["activated"]:
                if p_value < alpha and corr > 0:
                    df.at[i, "monotonic"] = "integrating"
                elif p_value < alpha and corr < 0:
                    df.at[i, "monotonic"] = "habituating"
                elif p_value >= alpha:
                    df.at[i, "monotonic"] = "monotonic"
            
            elif row["suppressed"]:
                if p_value < alpha and corr > 0:
                    df.at[i, "monotonic"] = "habituating"
                elif p_value < alpha and corr < 0:
                    df.at[i, "monotonic"] = "integrating"
                elif p_value >= alpha:
                    df.at[i, "monotonic"] = "monotonic"

    if save_path:		
        df.to_hdf(save_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal', mode='w')

    return df


def label_transient_neurons(df, max_transient_threshold, min_sustained_threshold, save_path=None, **kwargs):
    """
    Labels neurons in the DataFrame as 'transient', 'semi-sustained', or 'sustained' based on their decay duration.
    Parameters:
        df (pd.DataFrame): DataFrame containing neuron data with a 'responsive' column.
        max_transient_threshold (float): Maximum threshold for transient responses.
        min_sustained_threshold (float): Minimum threshold for sustained responses.
        save_path (str or Path, optional): Path to save the labeled DataFrame as an HDF5 file. Default is None.
        **kwargs: Additional arguments passed to get_peaks_across_pulses.
    Returns:
        pd.DataFrame: DataFrame with an added 'decay_type' column indicating the type of response.
    """
    decays = get_decays_across_pulses(df, **kwargs)
    median_values = decays.median(axis=0)

    transient_neurons = decays["Values"].columns[median_values < max_transient_threshold]
    sustained_neurons = decays["Values"].columns[median_values > min_sustained_threshold]

    df["decay_type"] = None
    for i, row in df.iterrows():
        if row["responsive"] == True and i in transient_neurons:
            df.loc[i, "decay_type"] = "transient"
        elif row["responsive"] == True and i in sustained_neurons:
            df.loc[i, "decay_type"] = "sustained"
        elif row["responsive"] == True:
            df.loc[i, "decay_type"] = "semi-sustained"

    if save_path:
        df.to_hdf(save_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal')

    return df


def compute_kde(df, axis, x_vals, categories, values, scale_factor=1):
    """
    Compute KDEs for given categories along a specified axis.
    Parameters:
        df (DataFrame): Input dataframe containing neuron data.
        axis (int): Axis along which to compute the KDE (0 = LR, 1 = AP, 2 = DV).
        x_vals (array): Range of values to evaluate the KDE.
        categories (list of str): Categories to compute KDEs for.
        values (list of str): Corresponding column names in df for each category.
        scale_factor (float): Conversion factor for microns per pixel. Default is 1.
    Returns:
        np.array: Array of KDE values for each category.
    """
    assert len(categories) == len(values), "Categories and values lists must be the same length."

    kdes = []

    for category, value in zip(categories, values):
        subset = df[df[category] == value]  # Select rows where the value column is True
        
        if subset.empty:
            kdes.append(np.zeros_like(x_vals))  # If no data, return zero density
        else:
            if axis == 2:
                data_values = np.array([p * scale_factor for p in subset["plane"].dropna()])  # Use 'plane' for z-axis
            else:
                data_values = np.array([com[axis] * scale_factor for com in subset["com_aligned"].dropna()])
            
            try:
                kdes.append(gaussian_kde(data_values)(x_vals))
            except ValueError:
                kdes.append([None] * len(x_vals))

    return np.array(kdes)


def extract_xyz_coordinates(df, categories):
    """
    Extracts XYZ coordinates for each neuron category.
    Parameters:
        df (pd.DataFrame): DataFrame containing neuron spatial information.
        categories (dict): Dictionary mapping category names to their filter conditions.
    Returns:
        dict: Dictionary where keys are category names and values are numpy arrays of XYZ coordinates.
    """
    xyz_coordinates = {category: [] for category in categories}

    # Iterate over the DataFrame only once
    for _, row in df.iterrows():
        for category, condition in categories.items():
            if condition(row):
                xyz_coordinates[category].append([row["com_aligned"][0], 
                                                  row["com_aligned"][1], 
                                                  row["plane"]])
    
    # Convert lists to numpy arrays
    return {key: np.array(value) for key, value in xyz_coordinates.items()}


def compute_kld(X1, X2, num_samples=100):
    """
    Computes the average pairwise Kullback-Leibler divergence (KLD) between two distributions.
    Parameters:
        X1 (np.ndarray): XYZ coordinates for the first set of ROIs (shape: Nx3).
        X2 (np.ndarray): XYZ coordinates for the second set of ROIs (shape: Mx3).
        num_samples (int): Number of points to sample for KDE evaluation.
    Returns:
        float: Symmetrized KLD between the two distributions.
    """
    try:
        # Create KDE estimators for both sets of coordinates
        kde1 = gaussian_kde(X1.T)  # Transpose because KDE expects (dim, num_samples)
        kde2 = gaussian_kde(X2.T)
    except ValueError:
        # If we only have a couple data points, the KDE may fail
        return None

    # Define a common evaluation grid spanning both distributions
    xmin, ymin, zmin = np.minimum(X1.min(axis=0), X2.min(axis=0))
    xmax, ymax, zmax = np.maximum(X1.max(axis=0), X2.max(axis=0))
    
    # Generate a grid of sample points
    x_vals = np.linspace(xmin, xmax, num_samples)
    y_vals = np.linspace(ymin, ymax, num_samples)
    z_vals = np.linspace(zmin, zmax, num_samples)
    
    # Create a mesh grid for evaluation
    X_grid, Y_grid, Z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])  # Shape (3, num_samples^3)

    # Evaluate KDEs on the grid
    P = kde1(grid_points)
    Q = kde2(grid_points)

    # Normalize to ensure they are valid probability distributions
    P /= P.sum()
    Q /= Q.sum()

    # Compute the Kullback-Leibler divergence in both directions
    kld_PQ = np.sum(rel_entr(P, Q))  # D_KL(P || Q)
    kld_QP = np.sum(rel_entr(Q, P))  # D_KL(Q || P)

    # Return the symmetric KLD
    return (kld_PQ + kld_QP) / 2


def compute_same_class_clustering_index(X, labels, max_radius, step=10, n_neighbors=None, randomize=False):
    """
    Computes the same-class clustering index at increasing distances, normalized by 
    global class probability, using PyNNDescent for fast nearest-neighbor search.
    Parameters:
        X (np.ndarray): (N, 3) Array of spatial coordinates for each neuron.
        labels (np.ndarray): Array of class labels for each neuron.
        max_radius (float): Maximum radius to compute probabilities.
        step (float): Distance increment to evaluate same-class probability.
        n_neighbors (int or None): Number of neighbors sampled per iteration. 
                                    If None or greater than len(X) - 1, it is set to len(X) - 1.
        randomize (bool): If True, shuffles the labels to compute a random baseline.
    Returns:
        pd.DataFrame: A DataFrame where rows represent distance values and columns 
                        contain same-class clustering indices per neuron for each label.
    """
    # Ensure n_neighbors is set appropriately
    n_neighbors = min(n_neighbors if n_neighbors is not None else len(X) - 1, len(X) - 1)
    print("Number of neighbors:", n_neighbors)

    # Build nearest-neighbor graph
    nn_index = pynndescent.NNDescent(X, n_neighbors=n_neighbors, metric="euclidean")
    indices, distances = nn_index.neighbor_graph

    # If randomize, shuffle the labels
    if randomize:
        labels = np.random.permutation(labels)

    unique_labels = np.unique(labels)
    distance_values = np.arange(step, max_radius + step, step)  # Distance bins

    # Initialize storage for same-class clustering indices
    same_class_clust_inds = {label: [] for label in unique_labels}

    # Global probability of encountering each class
    global_probabilities = {label: np.mean(labels == label) for label in unique_labels}

    # Iterate over distances
    for radius in distance_values:
        same_class_radius_probs = {label: [] for label in unique_labels}

        for idx, neuron_label in enumerate(labels):
            # Get neighbors within the current radius
            neighbor_indices = indices[idx, distances[idx, :] <= radius]

            # Count same-class neighbors
            same_class_neighbors = np.sum(labels[neighbor_indices] == neuron_label)
            total_neighbors = len(neighbor_indices)

            same_class_radius_probs[neuron_label].append(same_class_neighbors/total_neighbors)

        # Compute clustering indices for each label
        for label in unique_labels:
            clustering_indices = same_class_radius_probs[label]# / global_probabilities[label]
            same_class_clust_inds[label].append(np.array(clustering_indices))

    # Determine the maximum number of neurons across all categories
    max_neurons = max(max(arr.shape[0] for arr in cat_list) for cat_list in same_class_clust_inds.values())
    same_class_clust_inds_df = {}

    for category, distances_list in same_class_clust_inds.items():
        # Create a 2D array filled with NaNs, shape (num_distances, max_neurons)
        category_array = np.full((len(distance_values), max_neurons), np.nan)

        # Fill with actual values
        for i, neuron_values in enumerate(distances_list):
            category_array[i, :len(neuron_values)] = neuron_values

        # Store in dictionary with multi-index column names
        for j in range(max_neurons):
            same_class_clust_inds_df[(category, f"Neuron_{j+1}")] = category_array[:, j]
    
    same_class_clust_inds_df = pd.DataFrame(same_class_clust_inds_df, index=distance_values)
    same_class_clust_inds_df.index.name = "Distance (µm)"

    return same_class_clust_inds_df


def sort_by_peak_with_indices(data, separate_array=None, window=10):
    """
    Sort the data based on peak response times and apply the same sorting to a separate array (if provided).
    Parameters:
        data (np.ndarray): Array to sort (e.g., neuron responses across time).
        separate_array (np.ndarray, optional): Array to apply the same sorting indices to. Default is None.
        window (int): Sliding window size for smoothing the data. Default is 10.
    Returns:
        tuple: Contains the following:
            - sorted_data (np.ndarray): Data sorted by peak response times.
            - sorted_separate_array (np.ndarray, optional): The separate array sorted using the same indices (if provided).
            - sorting_indices (np.ndarray): Indices used for sorting.
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


def find_twitch_responsive(tank, min_pulses=None, **kwargs):
    """
    Identify twitch-responsive neurons in a tank and update the corresponding DataFrame.
    This function processes each fish in the provided tank to determine which neurons are 
    responsive to twitch stimuli. It updates the unrolled DataFrame for each fish with 
    information about twitch responsiveness, activation, and suppression. The updated 
    DataFrame is then saved to an HDF5 file.
    Parameters:
        tank (BrukerTank): The BrukerTank object containing fish and their associated data.
        min_pulses (int, optional): Minimum number of pulses required for a neuron to be 
            considered twitch-responsive. If not provided, defaults to half the number 
            of twitch frames.
        **kwargs: Additional keyword arguments passed to the `find_stimulus_responsive` 
            function.
    Returns:
        None
    Notes:
        - The function modifies the unrolled DataFrame for each fish in the tank.
    """
    fish_dfs = []
    for fish in tank.fish:
        fish_df = tank.unrolled_df[tank.unrolled_df["fish_id"] == fish.fish_id].reset_index(drop=True)
        
        fish_df["twitch_frames"] = [fish.temporal_df.loc[0, "twitch_frames"]] * len(fish_df)
        fish_df["twitch_responsive"] = None
        fish_df["twitch_activated"] = None
        fish_df["twitch_suppressed"] = None

        if len(fish.temporal_df.loc[0, "twitch_frames"]) > 0:
            if min_pulses is None:
                min_pulses = np.ceil(len(fish.temporal_df.loc[0, "twitch_frames"]) / 2)

            # Identify twitch-responsive neurons and update the DataFrame with response information.
            stim_responsive, activated, suppressed, pulse_responses = find_stimulus_responsive(fish, pulse_key="twitch_frames", **kwargs)
        
            fish_df["twitch_responsive"] = False
            fish_df["twitch_activated"] = False
            fish_df["twitch_suppressed"] = False
            for i, neuron in enumerate(stim_responsive):
                if len(pulse_responses[i]) >= min_pulses:
                    fish_df.at[neuron, 'twitch_responsive'] = True

                    if neuron in activated:
                        fish_df.at[neuron, 'twitch_activated'] = True
                    
                    elif neuron in suppressed:
                        fish_df.at[neuron, 'twitch_suppressed'] = True

        fish_dfs.append(fish_df)
        fish_df.to_hdf(fish.exp_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal', mode='w')

    # Concatenate the DataFrames for all fish
    df = pd.concat(fish_dfs, ignore_index=True)
    df.to_hdf(tank.folder_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal', mode='w')
    
    return df


def pca_clustering(df, filterby=None, colorby=None, colors=None, key='raw_norm_temporal', n_components=3):
    '''PCA clustering on individual neuron responses. Plots 3D components using matplotlib.
    filterby: runs separate clustering based on the filters
    colorby: colors each point based on the filter'''
    if colorby is not None and colorby not in df.columns:
        raise ValueError("Given colorby filter is not a column in the df")

    def plot_components(components, color_vals, total_var, title, n_components):
        if n_components == 3:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            if color_vals is not None:
                scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=color_vals, s=30, alpha=0.5)
                if hasattr(color_vals, 'unique'):
                    labels = color_vals.unique()
                else:
                    labels = np.unique(color_vals)
                if len(labels) < 20:  # Only show legend if not too many classes
                    legend1 = ax.legend(*scatter.legend_elements(), title=colorby)
                    ax.add_artist(legend1)
            else:
                ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=color_vals, s=30)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')
            ax.set_title(f"{title} - Total Explained Variance: {total_var:.2f}%")
            plt.tight_layout()
            plt.show()
        
        elif n_components == 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            if color_vals is not None:
                scatter = ax.scatter(components[:, 0], components[:, 1], c=color_vals, s=30, alpha=0.5)
                if hasattr(color_vals, 'unique'):
                    labels = color_vals.unique()
                else:
                    labels = np.unique(color_vals)
                if len(labels) < 20:
                    legend1 = ax.legend(*scatter.legend_elements(), title=colorby)
                    ax.add_artist(legend1)
            else:
                ax.scatter(components[:, 0], components[:, 1], s=30)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(f"{title} - Total Explained Variance: {total_var:.2f}%")
            plt.tight_layout()
            plt.show()

    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            traces = np.array(subdf.loc[:, key])
            traces = np.array([np.array(trace) for trace in traces])
                    
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(traces)

            total_var = pca.explained_variance_ratio_.sum() * 100

            title = " - ".join([str(cond) for cond in conditions])
            color_vals = colors if colors is not None else None
            plot_components(components, color_vals, total_var, title, n_components)
    else:
        traces = np.array(df.loc[:, key])
        traces = np.array([np.array(trace) for trace in traces])
                
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(traces)

        total_var = pca.explained_variance_ratio_.sum() * 100

        title = ""
        color_vals = colors if colors is not None else None
        plot_components(components, color_vals, total_var, title, n_components)
