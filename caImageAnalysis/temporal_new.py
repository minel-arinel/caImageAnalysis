from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem, wilcoxon
from sklearn.metrics import auc

from caImageAnalysis.statistics import check_monotonicity_repeated_measures


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
                             normalize_by_first=False):
    """
    Identifies stimulus responsive neurons in a given fish.
    Parameters:
        fish (object): Fish object containing temporal data.
        pre_frame_num (int): Number of frames before the pulse (default is 15).
        post_frame_num (int): Number of frames after the pulse (default is 5).
        peak_threshold (float, optional): Minimum normalized fluorescence intensity for an activated neuron.
        min_threshold (float, optional): Maximum normalized fluorescence intensity for an suppressed neuron.
        key (str, optional): Key to access neuron data in the dataframe (default is 'raw_norm_temporal').
        normalize (bool): If True, normalizes using the pre-injection period as baseline.
        normalize_by_first (bool): If True, normalizes using the pre-injection period of the first pulse as baseline.
    Returns:
        tuple: Contains lists of stimulus responsive neurons, activated neurons, suppressed neurons, and pulse responses.
    Raises:
        ValueError: If both normalize and normalize_by_first are True.
    Notes:
        - Activated neurons have peak responses greater than 2 times the pre-injection standard deviation.
        - Suppressed neurons have minimum responses smaller than 2 times the pre-injection standard deviation.
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
            pulse_frames.append(row['pulse_frames'])

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


def calculate_percentage_metric_neurons(df, metric, value, filterby=None, inverse=False):
    """
    Calculate the percentage of neurons per fish that meet a specific metric condition.
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        metric (str): Column name in df to evaluate.
        value (any): Value that the rows need to match in the metric column.
        filterby (list, optional): List of columns to filter the data by. Defaults to None.
        inverse (bool, optional): If True, calculate percentage where metric is not equal to value. Defaults to False.
    Returns:
        pd.DataFrame: DataFrame with percentage of metric neurons per fish for each filter group.
    """
    if filterby is not None:
        # Check if all filter columns exist in the DataFrame
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        
        # Group the DataFrame by the filter columns
        filter_groups = df.groupby(filterby).size().reset_index()

        perc_metric_neurons = dict()

        # Iterate over each group in the filter groups
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            label = " - ".join([str(cond) for cond in conditions])
            perc_metric_neurons[label] = list()

            # Calculate the percentage of metric neurons for each fish
            for fish in subdf.fish_id.unique():
                fish_df = df[df.fish_id == fish]
                sub_fish_df = subdf[subdf.fish_id == fish]

                if not inverse:
                    if value is None:
                        perc_metric_neurons[label].append(len(sub_fish_df[sub_fish_df[metric].isnull()]) / len(fish_df) * 100)
                    else:
                        perc_metric_neurons[label].append(len(sub_fish_df[sub_fish_df[metric] == value]) / len(fish_df) * 100)
                else:
                    if value is None:
                        perc_metric_neurons[label].append(len(sub_fish_df[sub_fish_df[metric].notnull()]) / len(fish_df) * 100)
                    else:
                        perc_metric_neurons[label].append(len(sub_fish_df[sub_fish_df[metric] != value]) / len(fish_df) * 100)

        # Convert the dictionary to a DataFrame and print it
        perc_metric_neurons = dict([(k, pd.Series(v)) for k, v in perc_metric_neurons.items()])
        print(pd.DataFrame(perc_metric_neurons))
        pd.DataFrame(perc_metric_neurons).to_clipboard()
        return pd.DataFrame(perc_metric_neurons)
    
    else:
        perc_metric_neurons = list()

        # Calculate the percentage of metric neurons for each fish
        for fish in df.fish_id.unique():
            fish_df = df[df.fish_id == fish]

            if not inverse:
                if value is None:
                    perc_metric_neurons.append(len(fish_df[fish_df[metric].isnull()]) / len(fish_df) * 100)
                else:
                    perc_metric_neurons.append(len(fish_df[fish_df[metric] == value]) / len(fish_df) * 100)
            else:
                if value is None:
                    perc_metric_neurons.append(len(fish_df[fish_df[metric].notnull()]) / len(fish_df) * 100)
                else:
                    perc_metric_neurons.append(len(fish_df[fish_df[metric] != value]) / len(fish_df) * 100)

        # Convert the list to a DataFrame and print it
        perc_metric_neurons_df = pd.DataFrame(perc_metric_neurons, columns=[value])
        print(perc_metric_neurons_df)
        perc_metric_neurons_df.to_clipboard()
        return perc_metric_neurons_df
    

def get_traces(df, pre_frame_num=15, post_frame_num=13, normalize=False, 
               normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, 
               return_col=None, specific_pulse=None):
    """
    Extracts neural traces around pulses from a DataFrame.
    Returns the x-axis (in frames) and a list of traces.
    Parameters:
        pre_frame_num (int): Number of frames before the pulse.
        post_frame_num (int): Number of frames after the pulse.
        normalize (bool): If True, normalizes traces using the pre-injection period.
        normalize_by_first (bool): If True, normalizes using the first pulse's pre-injection period.
        key (str): Column name to extract traces from.
        only_responsive (bool): If True, extracts traces only for responsive pulses.
        return_col (str or list): If provided, returns additional column values for the traces.
        specific_pulse (int, optional): If provided, extracts traces only for the specified pulse.
    Returns:
        tuple: x-axis values, list of traces, and optionally additional column values.
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
        pulses = neuron['pulse_frames']

        if only_responsive:
            responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
            pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, suppressed

            if normalize_by_first:
                baseline = 0

            for i, pulse in enumerate(responsive_pulses):
                if specific_pulse is not None and pulse != specific_pulse:
                    continue

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
                if specific_pulse is not None and i + 1 != specific_pulse:
                    continue

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
        dict: Dictionary with THM times for each neuron group or individual neuron.
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
    peaks = get_peaks_across_pulses(df, **kwargs)

    correlations, p_values = check_monotonicity_repeated_measures(peaks, return_results=True)

    correlations = np.array(correlations)
    p_values = np.array(p_values)

    integrating_neurons = peaks["Values"].columns[(p_values < alpha) & (correlations > 0)]
    habituating_neurons = peaks["Values"].columns[(p_values < alpha) & (correlations < 0)]

    df["monotonic"] = None
    for i, row in df.iterrows():
        if row["responsive"] == True and i in integrating_neurons:
            df.loc[i, "monotonic"] = "integrating"
        elif row["responsive"] == True and i in habituating_neurons:
            df.loc[i, "monotonic"] = "habituating"
        elif row["responsive"] == True:
            df.loc[i, "monotonic"] = "monotonic"

    if save_path:		
        df.to_hdf(save_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal')

    return df
