import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem

from caImageAnalysis.temporal_new import get_traces
from caImageAnalysis.utils import sort_by_peak, sort_by_peak_with_indices


def plot_heatmap(data, sort=True, fps=1.3039181000348583, pulses=[391, 548, 704, 861, 1017], 
                 tick_interval=60):
    '''Plots a heatmap of temporal data.
    tick_interval: interval between x-axis ticks in seconds'''
    fig = plt.figure(figsize=(20, 15))
    
    if sort:
        data = sort_by_peak(np.vstack(data))
    else:
        data = np.vstack(data)

    plt.imshow(data, cmap='inferno', interpolation='nearest', aspect='auto')
    
    duration_in_mins = round(len(data[0])/fps/60)
    ticks = np.arange(0, (duration_in_mins+1)*60*fps, tick_interval*fps)
    plt.xticks(ticks=ticks, labels=np.round(ticks/fps).astype(int))
    
    for pulse in pulses:
        plt.vlines(pulse, -0.5, len(data)-0.5, color='w', lw=3)
    
    plt.xlabel('Time (s)')
    plt.grid(visible=False)


def plot_grouped_heatmaps(df, filterby, sort=True, key='norm_temporal',
                          save_path=None, colors=None, **kwargs):
    """
    Plots grouped heatmaps based on specified filters.
    Parameters:
        df (pd.DataFrame): DataFrame containing the data to be plotted.
        filterby (list): List of column names to filter the DataFrame by.
        sort (bool, optional): If True, sorts the heatmap by peak values. Default is True.
        key (str, optional): Column name containing the data to be plotted in the heatmap. Default is 'norm_temporal'.
        save_path (Path, optional): Path to save the figure if savefig is True. Default is None.
        colors (list, optional): List of colors for the heatmap. Default is None.
        **kwargs: Additional keyword arguments for the heatmap plotting functions.
    Raises:
        ValueError: If any filter is not a column in the DataFrame.
    Returns:
        None
    """
    for filter in filterby:
        if filter not in df.columns:
            raise ValueError("Given filter is not a column in the temporal_df")
        
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
        traces = list(subdf.loc[:, key])

        if colors is None:
            plot_heatmap(np.vstack(traces), sort=sort, **kwargs)
            plt.title(" - ".join([str(cond) for cond in conditions]))
        else:
            plot_heatmap_with_colorbar(np.vstack(traces), colors, sort=sort, **kwargs)

        if save_path:
            plt.savefig(save_path.joinpath("heatmap_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)


def plot_heatmap_with_colorbar(data, colors, sort=True, fps=1.3039181000348583, pulses=[391, 548, 704, 861, 1017], x_tick_interval=60, y_tick_interval=100):
    """
    Plots temporal components as a heatmap with a custom colorbar for row annotations.

    Parameters:
    - data: 2D Numpy array, rows are neurons and columns are time points.
    - colors: List of RGB/Hex color values for each row (length = n_neurons).
    - sort: Whether to sort rows by their peak response time (default: True).
    - fps: Frames per second, used to calculate time from frame indices.
    - pulses: List of pulse frame indices to mark on the heatmap.
    - tick_interval: Interval for time ticks on the x-axis (in seconds).
    """
    # Sort data if needed
    if sort:
        data, colors, _ = sort_by_peak_with_indices(data, colors)

    fig, (ax_heatmap, ax2) = plt.subplots(1, 2, sharey=True, width_ratios=[20, 1], height_ratios=[1], figsize=(20, 15))

    # Plot the heatmap
    ax_heatmap.imshow(data, cmap='inferno', interpolation='nearest', aspect='auto')

    # Add pulse lines
    for pulse in pulses:
        ax_heatmap.vlines(pulse, -0.5, len(data) - 0.5, color='w', lw=3)

    # Set x-axis ticks
    duration_in_mins = round(len(data[0]) / fps / 60)
    ticks = np.arange(0, (duration_in_mins + 1) * 60 * fps, x_tick_interval * fps)
    ax_heatmap.set_xticks(ticks)
    ax_heatmap.set_xticklabels(np.round(ticks / fps).astype(int))
    ax_heatmap.set_xlabel('Time (s)')

    # Add custom colorbar for row annotations
    add_row_colors(colors, ax2)
    
    # Restore Y-Ticks for ax_heatmap
    ax_heatmap.set_yticks(np.arange(0, data.shape[0], y_tick_interval))  # Set y-tick positions
    ax_heatmap.set_yticklabels([str(i) for i in range(0, data.shape[0], y_tick_interval)])  # Set y-tick labels
    ax_heatmap.tick_params(axis='y', which='both', length=5)  # Customize tick size

    # Remove grid
    ax_heatmap.grid(visible=False)

    plt.subplots_adjust(wspace=0)


def add_row_colors(colors, ax_colorbar, bar_width=0.1):
    """
    Plots a custom colorbar for row annotations.

    Parameters:
        - colors: List of RGB/Hex color values for each row (length = n_neurons).
        - ax_colorbar: Axes object to plot the colorbar on.
        - bar_width: Width of the bar.
    """
    bottom = -0.5  # y-coordinates of the bottom side of the bar
    ax_colorbar.set_prop_cycle(None)  # Reset color cycle in case of previous configurations

    # Iterate through colors and plot bars
    for i, color in enumerate(colors):
        # Plot a bar for each row
        p = ax_colorbar.bar(0, 1, bar_width, label=i, bottom=bottom, color=color, align="edge")
        bottom += 1
    
   # Clean up axis for the colorbar
    ax_colorbar.spines["top"].set_visible(False)
    ax_colorbar.spines["right"].set_visible(False)
    ax_colorbar.spines["left"].set_visible(False)
    ax_colorbar.spines["bottom"].set_visible(False)
    ax_colorbar.set_xticks([])
    ax_colorbar.set_yticks([])

    return ax_colorbar


def plot_random_neurons(df, n_neurons, key="raw_norm_temporal", fps=1.3039181000348583):
    """
    Plots the activity of randomly selected neurons from a dataframe.
    Parameters:
        df (pd.DataFrame): DataFrame containing neuron data.
        n_neurons (int): Number of neurons to randomly select and plot.
        key (str): Column name for neuron activity data. Default is "raw_norm_temporal".
        fps (float): Frames per second for time axis scaling. Default is 1.3039181000348583.
    Returns:
        None
    """
    selected_neurons = df.sample(n=n_neurons)

    traces = list()
    pulse_frames = list()

    for i, row in selected_neurons.iterrows():
        traces.append(row[key])
        pulse_frames.append(row["pulse_frames"])

    for i, t in enumerate(traces):
        fig, axes = plt.subplots(1, 1, figsize=(10, 1))
        for pulse in pulse_frames[i]:
            axes.vlines(pulse, 0, 1, color='r')

        axes.plot(t, color='black')
        axes.set_title(f"neuron {selected_neurons.index[i]}")
        
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)

        ticks = np.arange(0, 16*60*fps, 60*fps)
        axes.set_xticks(ticks=ticks, labels=np.round(ticks/fps).astype(int))

        axes.set_xlabel('Time (s)')
        plt.show()


def plot_neuron_traces(df, neuron_ids=None, key="raw_norm_temporal", fps=1.3039181000348583, save_path=None, sigma=0, file_name=None):
    """
    Plots neuron activity traces with stimulus pulse markers.
    Parameters:
        df (pd.DataFrame): DataFrame containing neuron data.
        neuron_ids (list, optional): List of neuron IDs to plot. If None, plots all neurons. Default is None.
        key (str, optional): Column name for neuron traces. Default is "raw_norm_temporal".
        fps (float, optional): Frames per second for time axis. Default is 1.3039181000348583.
        save_path (Path, optional): Path to save the plot as a PDF. Default is None.
        sigma (float, optional): Standard deviation for Gaussian kernel. If non-zero, smooths the traces. Default is 0.
        file_name (str, optional): Custom file name for saving the plot. Default is None.
    Returns:
        None
    """

    traces = list()
    pulse_frames = list()

    for i, row in df.iterrows():
        traces.append(row[key])
        pulse_frames.append(row["pulse_frames"])

    if neuron_ids is None:
        neuron_ids = list(range(len(df)))

    if len(neuron_ids) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 1))
        axes = [ax]
    else:
        fig, axes = plt.subplots(len(neuron_ids), 1, figsize=(10, len(neuron_ids)))

    for i, val in enumerate(neuron_ids):
        trace = np.array(traces)[val]
        if sigma > 0:
            trace = gaussian_filter1d(trace, sigma=sigma)

        for pulse in np.array(pulse_frames)[val]:
            axes[i].vlines(pulse, 0, 1, color='#e11f25')

        axes[i].plot(trace, color='#000000')

        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)

        ticks = np.arange(0, 16*60*fps, 60*fps)
        axes[i].set_xticks(ticks=ticks, labels=np.round(ticks/fps).astype(int))

        if i != len(neuron_ids)-1:
            axes[i].tick_params(
                axis='both',          # changes apply to both the x and y axes
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                labelbottom=False, # labels along the bottom edge are off
                left=False,
                labelleft=False
            )
        else:
            axes[i].set_xlabel("Time (s)")

    if save_path:
        if file_name is None:
            file_name = f"neuron_traces_{'_'.join(map(str, neuron_ids))}.pdf"
        plt.savefig(save_path.joinpath(file_name))
    plt.show()


def plot_average_trace_overlayed(df, overlay_filter, filterby=None, color_order=None, 
                                  overlay_order=None, fps=1, save_path=None, ylim=None, **kwargs):
    """
    Plots pulse averages overlayed on top.
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        overlay_filter (str): Column name to overlay traces by.
        filterby (list, optional): List of column names to filter the data by. Defaults to None.
        color_order (list, optional): List of colors for each overlay filter. Defaults to None.
        overlay_order (list, optional): Specific order for overlay filters. Defaults to None.
        fps (int, optional): Frames per second for x-axis scaling. Defaults to 1.
        save_path (Path, optional): Path to save the plot. Defaults to None.
        ylim (tuple, optional): Tuple specifying the y-axis limits. Defaults to None.
        **kwargs: Additional arguments for the get_traces function.
    Raises:
        ValueError: If overlay_filter or any filter in filterby is not a column in df.
    Returns:
        None

    TODO: The number of fish in the legend may be inaccurate if a specific pulse is given 
    to get_traces and only_responsive is True. Or this can happen if the specific pulse is
    5, and there are fish with only 4 pulses.
    """
    if overlay_filter not in df.columns:
        raise ValueError("Given overlay_filter is not a column in the df")
    
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
            x, traces, overlay_filters = get_traces(subdf, return_col=overlay_filter, **kwargs)

            plt.figure(figsize=(10, 10))

            if overlay_order is not None:  # if you want the overlay_filters to go in a specific order
                for i, of in enumerate(overlay_order):
                    tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                    avg_trace = np.mean(np.array(tr), axis=0)
                    sems = sem(np.array(tr), axis=0)

                    if color_order is not None:
                        plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}', color=color_order[i])
                        plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                    else:
                        plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}')
                        plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

            else:  # if you don't care about the order, it will just find the unique overlay filters
                for i, of in enumerate(np.unique(overlay_filters)):
                    tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                    avg_trace = np.mean(np.array(tr), axis=0)
                    sems = sem(np.array(tr), axis=0)

                    if color_order is not None:
                        plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}', color=color_order[i])
                        plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                    else:
                        plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}')
                        plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

            plt.axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
            plt.legend()
            plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)
            if fps != 1:
                plt.xlabel('Time (s)')
            else:
                plt.xlabel('Time (frames)')
            if ylim:
                plt.ylim(ylim)
            plt.show()

            if save_path:
                plt.savefig(save_path.joinpath("average_trace_by_" + overlay_filter + "_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)
    
    else:
        x, traces, overlay_filters = get_traces(df, return_col=overlay_filter, **kwargs)

        plt.figure(figsize=(10, 10))

        if overlay_order is not None:  # if you want the overlay_filters to go in a specific order
            for i, of in enumerate(overlay_order):
                tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                avg_trace = np.mean(np.array(tr), axis=0)
                sems = sem(np.array(tr), axis=0)

                if color_order is not None:
                    plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(df[df[overlay_filter] == of])}', color=color_order[i])
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                else:
                    plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(df[df[overlay_filter] == of])}')
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

        else:  # if you don't care about the order, it will just find the unique overlay filters
            for i, of in enumerate(np.unique(overlay_filters)):
                tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                avg_trace = np.mean(np.array(tr), axis=0)
                sems = sem(np.array(tr), axis=0)

                if color_order is not None:
                    plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(df[df[overlay_filter] == of])}', color=color_order[i])
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                else:
                    plt.plot(x/fps, avg_trace, zorder=102, label=f'{of}, n={len(df[df[overlay_filter] == of])}')
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

        plt.axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
        plt.legend()
        plt.title(f"Average trace by {overlay_filter}", fontsize=18)
        if fps != 1:
            plt.xlabel('Time (s)')
        else:
            plt.xlabel('Time (frames)')
        if ylim:
            plt.ylim(ylim)
        plt.show()

        if save_path:
            plt.savefig(save_path.joinpath(f"average_trace_by_{overlay_filter}.pdf"), transparent=True)

def plot_pulse_aligned_traces(row, fps=1, draw_lines=[], save_path=None, specific_pulse=None, **kwargs):
    """
    Plots traces aligned to pulses from a single row of a DataFrame.
    Parameters:
        row (pd.Series): A row from a DataFrame containing neuron data.
        fps (float): Frames per second for time axis scaling. Default is 1.
        draw_lines (list): List of y-values to draw horizontal dashed lines. Default is an empty list.
        save_path (Path, optional): Path to save the plot as a PDF. Default is None.
        specific_pulse (int, optional): Specific pulse to plot. Default is None.
        **kwargs: Additional keyword arguments for the get_traces function.
    Returns:
        None
    """
    x, traces = get_traces(row, specific_pulse=specific_pulse, **kwargs)
    avg_trace = np.mean(traces, axis=0)
    sems = sem(traces, axis=0)
    x = x / fps

    plt.figure(figsize=(5, 5))
    for trace in traces:
        plt.plot(x, trace, color='lightgray', alpha=0.5)
    
    plt.plot(x, avg_trace, color='black')
    plt.fill_between(x, avg_trace - sems, avg_trace + sems, color='black', alpha=0.2)
    plt.axvspan(-1/fps, 0, color='red', alpha=0.5)
    
    for line in draw_lines:
        plt.axhline(y=line, linestyle='--')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized dF/F (pre-inj.)')

    if save_path:
        if specific_pulse is not None:
            plt.savefig(save_path.joinpath(f'neuron_{row.name}_pulse_{specific_pulse}_aligned_traces.pdf'), transparent=True)
        else:
            plt.savefig(save_path.joinpath(f'neuron_{row.name}_pulse_aligned_traces.pdf'), transparent=True)
