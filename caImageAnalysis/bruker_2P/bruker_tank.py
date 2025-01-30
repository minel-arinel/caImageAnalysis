import numpy as np
import os
import pandas as pd
from pathlib import Path

from caImageAnalysis import BrukerFish
from caImageAnalysis.temporal_new import *

class BrukerTank():
    def __init__(self, folder_path, fish_ids, prefix='elavl3H2BGCaMP', load_fish=False, region=''):
        self.folder_path = Path(folder_path)
        self.fish_ids = fish_ids
        self.fish = list()
        self.data_paths = dict()
        self.prefix = prefix
        self.region = region

        if load_fish:
            self.load_fish()

        self.process_tank_filestructure()

    def process_tank_filestructure(self):
        '''Appends tank specific file paths to the data_paths attribute'''
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path) and entry.name == 'figures':
                    self.data_paths['figures'] = Path(entry.path)
                elif entry.name == 'temporal.h5':
                    self.data_paths['temporal'] = Path(entry.path)
                    self.temporal_df = pd.read_hdf(entry.path)
                elif entry.name == 'unrolled_temporal.h5':
                    self.data_paths['unrolled_temporal'] = Path(entry.path)
                    self.unrolled_df = pd.read_hdf(entry.path)
                elif entry.name == 'monotonic_temporal.h5':
                    self.data_paths['monotonic_temporal'] = Path(entry.path)
                    self.monotonic_df = pd.read_hdf(entry.path)

            if 'figures' not in self.data_paths:
                os.mkdir(self.folder_path.joinpath("figures"))
                self.data_paths['figures'] = self.folder_path.joinpath("figures")
    
    def load_fish(self):
        '''Finds the folders of the  given fish and loads them'''
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path) and entry.name.startswith(self.prefix):
                    parsed = entry.name.split('_')
                    
                    try:
                        if int(parsed[-2]) in self.fish_ids:
                            self.fish.append(BrukerFish(entry.path, region=self.region))
                    except:
                        # sometimes you cannot parse properly with some folders
                        pass

        self.fps = np.mean([fish.fps for fish in self.fish])
                        
    def save_tank_temporal(self, overwrite=False):
        '''Saves the temporal components of final ROIs from all fish as a temporal.h5 file
        Also calculates the dF/F0 and adds it to the dataframe
        
        overwrite: if True, will recalculate and save each fish's temporal.h5'''
        if len(self.fish) == 0:
            self.load_fish()
        
        tank_temporal_df = list()

        for fish in self.fish:
            if overwrite or 'temporal' not in fish.data_paths:
                fish.save_temporal()
                fish.normalize_temporaldf()
                df = fish.add_coms_to_temporaldf()

                print(f'saved {fish.exp_path.name}')
                
            else:
                df = fish.temporal_df

            df['fish'] = fish.fish_id
            df['age'] = int(fish.age)
            try:
                df['stimulus'] = str(fish.stimulus)
            except AttributeError:
                # if we don't give any stimuli
                df['stimulus'] = None
            
            try:
                # for mM or uM
                df['concentration'] = float(fish.concentration[:-2])
            except ValueError:
                # for ugml
                df['concentration'] = float(fish.concentration[:-4])
            except AttributeError:
                # if we don't give any stimuli
                df["concentration"] = None
            
            df['region'] = str(fish.data_paths['raw'].name[:fish.data_paths['raw'].name.rfind('-')])

            tank_temporal_df.append(df)

        self.temporal_df = pd.concat(tank_temporal_df).reset_index().drop(columns='index')
        self.temporal_df.to_hdf(self.folder_path.joinpath('temporal.h5'), key='temporal')
        print('saved tank temporal_df')

        return self.temporal_df
    
    def update_tank_temporal(self, overwrite=False, window=None, percentile=75):
        '''Does not overwrite the entire temporal_df but adds any additional fish to it
        Requires an already saved temporal_df
        
        overwrite: if True, will recalculate and save the additional fish's temporal.h5
        window: if an integer, calculates baseline as the first n frames
        percentile: the nth percentile for normalization'''
        if len(self.fish) == 0:
            self.load_fish()
        
        for fish in self.fish:
            if fish.fish_id in self.temporal_df.fish.unique():
                pass

            else:
                if overwrite:
                    save_temporal(fish)
                    compute_median_dff(fish, window=window)
                    normalize_dff(fish)
                    normalize_temporaldf(fish)
                    df = add_coms_to_temporaldf(fish)

                    print(f'saved {fish.exp_path.name}')

                else:
                    df = fish.temporal_df

                df['fish'] = fish.fish_id
                df['age'] = int(fish.age)
                df['stimulus'] = fish.stimulus
                df['concentration'] = float(fish.concentration[:-2])
                df['region'] = fish.data_paths['raw'].name[:fish.data_paths['raw'].name.rfind('-')]

                self.temporal_df = pd.concat([self.temporal_df, df]).reset_index().drop(columns='index')

        self.temporal_df.to_hdf(self.folder_path.joinpath('temporal.h5'), key='temporal')

        return self.temporal_df
    
    def plot_individual_fish_heatmaps(self, sort=True, savefig=True, tick_interval=60):
        '''Plots heatmaps of individual fish'''
        if savefig:
            save_path = self.data_paths['figures'].joinpath("individual_fish_heatmaps")
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        for fish in self.fish_ids:
            traces = list()
            fish_df = self.temporal_df[self.temporal_df.fish == fish]

            for _, row in fish_df.iterrows():
                traces.extend(row['norm_temporal'])

            if sort:
                data = sort_by_peak(np.vstack(traces))
            else:
                data = np.vstack(traces)

            if len(fish_df.pulse_frames.iloc[0]) == 1:
                pulses = fish_df.pulse_frames.iloc[0]
                plot_heatmap(data, fps=self.fps, pulses=pulses, tick_interval=tick_interval)
            else:
                plot_heatmap(data, fps=self.fps, tick_interval=tick_interval)

            plt.title(f"Fish {fish} - {fish_df.iloc[0, 13]} - {fish_df.iloc[0, 15]}")

            if savefig:
                plt.savefig(save_path.joinpath(f"heatmap_fish{fish}.pdf"), transparent=True)
                
    def plot_grouped_heatmaps(self, filterby, sort=True, key='norm_temporal', savefig=False):
        '''Plots heatmaps based on filtered columns.
        filterby: list of column names that the temporal_df will be filtered by'''
        if savefig:
            save_path = self.data_paths['figures']

        for filter in filterby:
            if filter not in self.temporal_df.columns:
                raise ValueError("Given filter is not a column in the temporal_df")
            
        filter_groups = self.temporal_df.groupby(filterby).size().reset_index()

        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(self.temporal_df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(self.temporal_df['{col}'] == {cond})")

            subdf = self.temporal_df[eval(" & ".join(filters))]
            traces = list(subdf.loc[:, key])

            if sort:
                data = sort_by_peak(np.vstack(traces))
            else:
                data = np.vstack(traces)

            plot_heatmap(data)
            plt.title(" - ".join([cond for cond in conditions]))

            if savefig:
                plt.savefig(save_path.joinpath("heatmap_" + "_".join([cond for cond in conditions]) + ".pdf"), transparent=True)

    def unroll_temporal_df(self, overwrite=False, **kwargs):
        '''Unrolls the temporal_df of the tank'''
        if len(self.fish) == 0:
            self.load_fish()

        dfs = list()
        for fish in self.fish:
            if overwrite or 'unrolled_temporal' not in fish.data_paths:
                df = unroll_temporal_df(fish, **kwargs)
            else:
                df = fish.unrolled_df

            df['age'] = int(fish.age)
            try:
                df['stimulus'] = fish.stimulus
                df['concentration'] = float(fish.concentration[:-2])
            except AttributeError:
                # if we're not giving any stimuli
                df["stimulus"] = None
                df["concentration"] = None
            df['region'] = fish.data_paths['raw'].name[:fish.data_paths['raw'].name.rfind('-')]

            dfs.append(df)

        self.unrolled_df = pd.concat(dfs, ignore_index=True)
        self.unrolled_df.to_hdf(self.folder_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal')

        self.process_tank_filestructure()

        return self.unrolled_df
    
    def hierarchical_clustering(self, df, max_inter_cluster_dist, filterby=None, key='raw_norm_temporal', sorted=False, savefig=False, save_path=None):
        '''Hierarchically clusters the temporal responses of all neurons
        filterby: runs separate clustering based on the filters
        colorby: colors each point based on the filter'''
        if savefig and save_path is None:
            raise ValueError("Enter a save_path to save the figure")
        
        if filterby is not None and not isinstance(max_inter_cluster_dist, list):
            raise ValueError("If you need to filter, max_inter_cluster_dist must be a list of distances")
        
        if filterby is not None:
            for filter in filterby:
                if filter not in df.columns:
                    raise ValueError("Given filter is not a column in the unrolled_df")
            filter_groups = df.groupby(filterby).size().reset_index()

            if len(max_inter_cluster_dist) != len(filter_groups):
                raise ValueError(f"The length of max_inter_cluster_dist should match the length of the total filter groups, which is {len(filter_groups)}.")
                
            for i, row in filter_groups.iterrows():
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
                        
                clusters, _ = hierarchical_clustering(traces, max_inter_cluster_dist[i])
                print("got the clusters")

                peak_clusters = dict()
                peak_indices = list()

                for cl in clusters:
                    # if sorted:
                    sorted_ts = sort_by_peak(clusters[cl])
                    peak_clusters[cl] = sorted_ts
                    # else:
                    #     peak_clusters[cl] = clusters[cl]

                    # calculate an index that takes a sliding window sum of each trace to find the "peak" time point
                    # this "peak" time point is then added for each trace of the cluster to create a peak_index
                    # the peak_index will be used to sort the clusters
                    peak_index = np.array([np.argmax(np.convolve(arr, np.ones(10), 'valid')) for arr in peak_clusters[cl]]).mean()
                    peak_indices.append(peak_index)
                    print(f"sorted cluster {cl}")
                print("sorted individual traces per cluster")

                sorted_peak_clusters = sorted(peak_clusters, key=lambda k: peak_indices[k-1])
                print("sorted across clusters")   

                sorted_traces = list()
                for cl in sorted_peak_clusters:
                    sorted_traces.extend(peak_clusters[cl])

                pulses=[391, 548, 704, 861, 1017]

                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, width_ratios=[20, 1], height_ratios=[1], figsize=(20, 10))
                ax1.imshow(sorted_traces, cmap='inferno', interpolation='nearest', aspect='auto')
                ax1.set_title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

                ticks = np.arange(0, 16*60*self.fps, 60*self.fps)
                ax1.set_xticks(ticks=ticks, labels=np.round(ticks/self.fps).astype(int))

                for pulse in pulses:
                    ax1.vlines(pulse, -0.5, len(sorted_traces)-0.5, color='w', lw=3)

                ax1.set_xlabel('Time (s)')

                bottom = -0.5  # y-coordinates of the bottom side of the bar
                x = 0
                width = 0.5

                cmap = get_cmap('Set1')  # type: matplotlib.colors.ListedColormap
                colors = cmap.colors  # type: list
                ax2.set_prop_cycle(color=colors)

                # Make the colormap for the clusters
                for cl in sorted_peak_clusters:
                    temps = peak_clusters[cl]

                    p = ax2.bar(x, len(temps), width, label=str(cl), bottom=bottom)
                    bottom += len(temps)
                    
                    ax2.bar_label(p, labels=[str(cl)], label_type='center')
                ax1.grid(visible=False)
                plt.subplots_adjust(wspace=0)
                
                plot_average_traces(clusters, pulses=pulses, fps=self.fps)

                if savefig:
                    plt.savefig(save_path.joinpath(f"hierarchical_clustering_{max_inter_cluster_dist[i]}_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)

        else:
            traces = np.array(df.loc[:, key])
            traces = np.array([np.array(trace) for trace in traces])
                    
            clusters, _ = hierarchical_clustering(traces, max_inter_cluster_dist)
            print("got the clusters")

            peak_clusters = dict()
            peak_indices = list()

            for cl in clusters:
                sorted_ts = sort_by_peak(clusters[cl])
                peak_clusters[cl] = sorted_ts

                # calculate an index that takes a sliding window sum of each trace to find the "peak" time point
                # this "peak" time point is then added for each trace of the cluster to create a peak_index
                # the peak_index will be used to sort the clusters
                peak_index = np.array([np.argmax(np.convolve(arr, np.ones(10), 'valid')) for arr in sorted_ts]).mean()
                peak_indices.append(peak_index)
                print(f"sorted cluster {cl}")
            print("sorted individual traces per cluster")
            
            sorted_peak_clusters = sorted(peak_clusters, key=lambda k: peak_indices[k-1])
            print("sorted across clusters")  

            sorted_traces = list()
            for cl in sorted_peak_clusters:
                sorted_traces.extend(peak_clusters[cl])

            pulses=[391, 548, 704, 861, 1017]

            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, width_ratios=[20, 1], height_ratios=[1], figsize=(20, 10))
            ax1.imshow(sorted_traces, cmap='plasma', interpolation='nearest', aspect='auto')

            ticks = np.arange(0, 16*60*self.fps, 60*self.fps)
            ax1.set_xticks(ticks=ticks, labels=np.round(ticks/self.fps).astype(int))

            for pulse in pulses:
                ax1.vlines(pulse, -0.5, len(sorted_traces)-0.5, color='w', lw=3)

            ax1.set_xlabel('Time (s)')

            bottom = -0.5  # y-coordinates of the bottom side of the bar
            x = 0
            width = 0.5

            cmap = get_cmap('Set1')  # type: matplotlib.colors.ListedColormap
            colors = cmap.colors  # type: list
            ax2.set_prop_cycle(color=colors)

            # Make the colormap for the clusters
            for cl in sorted_peak_clusters:
                temps = peak_clusters[cl]

                p = ax2.bar(x, len(temps), width, label=str(cl), bottom=bottom)
                bottom += len(temps)
                
                ax2.bar_label(p, labels=[str(cl)], label_type='center')
            ax1.grid(visible=False)
            plt.subplots_adjust(wspace=0)

            plot_average_traces(clusters, pulses=pulses, fps=self.fps)

            if savefig:
                plt.savefig(save_path.joinpath(f"hierarchical_clustering_{max_inter_cluster_dist}.pdf"), transparent=True)
