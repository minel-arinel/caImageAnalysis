import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from typing import Any, Dict, Tuple, Union

import h5py

from caImageAnalysis import BrukerFish
from caImageAnalysis.temporal_new import *


# --- Temporal pack HDF5 (one file: metadata group + 4 (n_rois, T) trace datasets) ---


def build_roi_long_df_from_fish(fish: Any, recompute_temporal: bool = False) -> pd.DataFrame:
    """
    One row per ROI for a single BrukerFish (plane loop unrolled).

    Columns: fish, age, stimulus, concentration, plane, region, roi_index,
             temporal, raw_temporal, norm_temporal, raw_norm_temporal, pulse_frames
    """
    if recompute_temporal or "temporal" not in fish.data_paths:
        fish.save_temporal()
        fish.normalize_temporaldf()
        df_plane = fish.add_coms_to_temporaldf()
    else:
        df_plane = fish.temporal_df.copy()

    fish_id = fish.fish_id
    age = int(fish.age)
    try:
        stimulus = str(fish.stimulus)
    except AttributeError:
        stimulus = None
    try:
        concentration = float(fish.concentration[:-2])
    except ValueError:
        concentration = float(fish.concentration[:-4])
    except AttributeError:
        concentration = None
    region = str(fish.data_paths["raw"].name[: fish.data_paths["raw"].name.rfind("-")])

    rows = []
    for _, row in df_plane.iterrows():
        plane = row["plane"]
        temps = row["temporal"]
        raw_temps = row["raw_temporal"]
        norm_temps = row.get("norm_temporal")
        raw_norm_temps = row.get("raw_norm_temporal")
        roi_idx_list = row["roi_indices"]
        pulse_frames = row["pulse_frames"]
        n = len(temps)
        for j in range(n):
            t = np.asarray(temps[j], dtype=np.float32)
            rt = np.asarray(raw_temps[j], dtype=np.float32)
            if norm_temps is not None and j < len(norm_temps) and norm_temps[j] is not None:
                nt = np.asarray(norm_temps[j], dtype=np.float32)
            else:
                nt = np.zeros_like(t)
            if raw_norm_temps is not None and j < len(raw_norm_temps) and raw_norm_temps[j] is not None:
                rnt = np.asarray(raw_norm_temps[j], dtype=np.float32)
            else:
                rnt = np.zeros_like(t)
            rows.append(
                {
                    "fish": fish_id,
                    "age": age,
                    "stimulus": stimulus,
                    "concentration": concentration,
                    "plane": int(plane),
                    "region": region,
                    "roi_index": int(roi_idx_list[j]),
                    "temporal": t,
                    "raw_temporal": rt,
                    "norm_temporal": nt,
                    "raw_norm_temporal": rnt,
                    "pulse_frames": pulse_frames,
                }
            )
    return pd.DataFrame(rows)


def save_temporal_pack_hdf5(df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    """
    Write one HDF5 file with metadata group and four (n_rois, T) float32 datasets.

    Layout:
      /metadata/{fish, age, stimulus, concentration, plane, region, roi_index, pulse_frames}
      /temporal, /raw_temporal, /norm_temporal, /raw_norm_temporal
    """
    out_path = Path(out_path)
    df = df.reset_index(drop=True)
    n = len(df)
    trace_cols = ["temporal", "raw_temporal", "norm_temporal", "raw_norm_temporal"]

    if n == 0:
        raise ValueError("DataFrame is empty; nothing to save.")

    T = int(np.asarray(df[trace_cols[0]].iloc[0]).shape[0])
    for c in trace_cols:
        if int(np.asarray(df[c].iloc[0]).shape[0]) != T:
            raise ValueError(f"All traces must have length {T}; mismatch in column {c}")

    pf_col = "pulse_frames" if "pulse_frames" in df.columns else "pulse frames"

    with h5py.File(out_path, "w") as f:
        g = f.create_group("metadata")

        g.create_dataset("fish", data=df["fish"].to_numpy(dtype=np.int32))
        g.create_dataset("age", data=df["age"].to_numpy(dtype=np.int32))

        str_dt = h5py.string_dtype(encoding="utf-8")
        stim = df["stimulus"].astype("string").fillna("").to_numpy(dtype=object)
        g.create_dataset("stimulus", data=stim, dtype=str_dt)

        conc = pd.to_numeric(df["concentration"], errors="coerce").to_numpy(dtype=np.float32)
        g.create_dataset("concentration", data=conc)

        g.create_dataset("plane", data=df["plane"].to_numpy(dtype=np.int32))

        reg = df["region"].astype("string").fillna("").to_numpy(dtype=object)
        g.create_dataset("region", data=reg, dtype=str_dt)

        g.create_dataset("roi_index", data=df["roi_index"].to_numpy(dtype=np.int32))

        vlen_i32 = h5py.vlen_dtype(np.dtype("int32"))
        d_pf = g.create_dataset("pulse_frames", shape=(n,), dtype=vlen_i32)
        for i in range(n):
            pf = df[pf_col].iloc[i]
            if pf is None or (isinstance(pf, float) and np.isnan(pf)):
                d_pf[i] = np.array([], dtype=np.int32)
            else:
                d_pf[i] = np.asarray(pf, dtype=np.int32).ravel()

        for name in trace_cols:
            X = np.stack(
                [np.asarray(df[name].iloc[i], dtype=np.float32) for i in range(n)],
                axis=0,
            )
            assert X.shape == (n, T)
            f.create_dataset(name, data=X, compression="gzip", compression_opts=4)

        f.attrs["T"] = T
        f.attrs["n_rois"] = n

    return out_path


def load_temporal_pack_hdf5(path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load temporal_pack.h5. Returns (metadata DataFrame, dict of trace arrays)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        g = f["metadata"]
        meta = pd.DataFrame(
            {
                "fish": g["fish"][:],
                "age": g["age"][:],
                "stimulus": g["stimulus"].asstr()[:],
                "concentration": g["concentration"][:],
                "plane": g["plane"][:],
                "region": g["region"].asstr()[:],
                "roi_index": g["roi_index"][:],
                "pulse_frames": [np.array(x, copy=True) for x in g["pulse_frames"]],
            }
        )
        traces = {
            k: np.array(f[k][:], copy=True)
            for k in ["temporal", "raw_temporal", "norm_temporal", "raw_norm_temporal"]
        }
    return meta, traces


def long_df_from_pack(meta: pd.DataFrame, traces: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Rebuild a long-format DataFrame (one ROI per row) from pack components."""
    n = len(meta)
    rows = []
    for i in range(n):
        rows.append(
            {
                **{c: meta.iloc[i][c] for c in meta.columns},
                "temporal": traces["temporal"][i],
                "raw_temporal": traces["raw_temporal"][i],
                "norm_temporal": traces["norm_temporal"][i],
                "raw_norm_temporal": traces["raw_norm_temporal"][i],
            }
        )
    return pd.DataFrame(rows)


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
                elif entry.name == 'temporal_pack.h5':
                    self.data_paths['temporal_pack'] = Path(entry.path)
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
        self.fish = []
        
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

    def save_tank_temporal_pack(
        self,
        out_filename="temporal_pack.h5",
        overwrite=False,
        sort_fish_by_id=True,
    ):
        """
        Build one ROI-per-row dataframe for all fish, then save a single HDF5 file:

          /metadata/{fish, age, stimulus, concentration, plane, region, roi_index, pulse_frames}
          /temporal, /raw_temporal, /norm_temporal, /raw_norm_temporal  (n_rois, T) float32

        All traces must share the same length T (asserted before write).

        Sets ``self.temporal_df`` to the long-format DataFrame and registers
        ``self.data_paths['temporal_pack']``.

        Parameters
        ----------
        out_filename : str
            Written under ``self.folder_path``.
        overwrite : bool
            If True, recompute each fish's per-plane temporal.h5 before unrolling.
        sort_fish_by_id : bool
            If True, process fish in ascending ``fish_id`` order (recommended).
        """
        if len(self.fish) == 0:
            self.load_fish()

        fish_list = sorted(self.fish, key=lambda f: f.fish_id) if sort_fish_by_id else list(self.fish)

        parts = []
        for fish in fish_list:
            if overwrite or "temporal" not in fish.data_paths:
                fish.save_temporal()
                fish.normalize_temporaldf()
                fish.add_coms_to_temporaldf()
                print(f"saved {fish.exp_path.name}")

            df_long = build_roi_long_df_from_fish(fish, recompute_temporal=False)
            parts.append(df_long)

        long_df = pd.concat(parts, ignore_index=True)
        out_path = self.folder_path.joinpath(out_filename)
        save_temporal_pack_hdf5(long_df, out_path)

        self.temporal_df = long_df
        self.data_paths["temporal_pack"] = out_path
        print(f"saved temporal pack: {out_path} (n_rois={len(long_df)})")
        return out_path

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

            # Debug: per-fish dataframe summary and potentially problematic columns
            print(f"[BrukerTank.save_tank_temporal] fish {fish.fish_id}: df shape = {df.shape}")
            for col in ['raw_temporal', 'temporal', 'norm_temporal', 'raw_norm_temporal', 'roi_indices', 'pulse_frames']:
                if col in df.columns:
                    try:
                        lengths = [len(x) if hasattr(x, '__len__') and not isinstance(x, (str, bytes)) else 0 for x in df[col]]
                        max_len = max(lengths) if lengths else 0
                        total_len = sum(lengths)
                        print(f"  col '{col}': n_rows={len(df)}, max_len={max_len}, total_len={total_len}")
                    except Exception as e:
                        print(f"  col '{col}': error computing lengths: {e}")

            tank_temporal_df.append(df)

        # Combine all fish dataframes
        self.temporal_df = pd.concat(tank_temporal_df).reset_index().drop(columns='index')

        # Debug: combined dataframe summary
        print(f"[BrukerTank.save_tank_temporal] combined temporal_df shape = {self.temporal_df.shape}")
        for col in ['raw_temporal', 'temporal', 'norm_temporal', 'raw_norm_temporal', 'roi_indices', 'pulse_frames']:
            if col in self.temporal_df.columns:
                try:
                    lengths = [len(x) if hasattr(x, '__len__') and not isinstance(x, (str, bytes)) else 0 for x in self.temporal_df[col]]
                    max_len = max(lengths) if lengths else 0
                    total_len = sum(lengths)
                    print(f"  combined col '{col}': n_rows={len(self.temporal_df)}, max_len={max_len}, total_len={total_len}")
                except Exception as e:
                    print(f"  combined col '{col}': error computing lengths: {e}")

        # Try writing subsets of columns incrementally to identify problematic column(s)
        base_cols = ['plane', 'fish', 'age', 'stimulus', 'concentration', 'region']
        path = self.folder_path.joinpath('temporal_debug.h5')

        print("[BrukerTank.save_tank_temporal] Testing HDF5 writes with incremental column subsets")
        try:
            self.temporal_df[base_cols].to_hdf(path, key='temporal', mode='w')
            print("  OK: base_cols only")
        except Exception as e:
            print(f"  FAIL: base_cols only -> {e}")

        for col in ['roi_indices', 'raw_temporal', 'temporal', 'norm_temporal', 'raw_norm_temporal', 'pulse_frames']:
            if col in self.temporal_df.columns:
                cols = base_cols + [col]
                try:
                    self.temporal_df[cols].to_hdf(path, key='temporal', mode='w')
                    print(f"  OK: base_cols + '{col}'")
                except Exception as e:
                    print(f"  FAIL: base_cols + '{col}' -> {e}")

        # Finally, attempt to save full dataframe to the intended file
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
                conc_str = str(fish.concentration)
                match = re.search(r"[-+]?\d*\.?\d+", conc_str)
                df['concentration'] = float(match.group(0)) if match else None
            except AttributeError:
                # if we're not giving any stimuli
                df["stimulus"] = None
                df["concentration"] = None
            df['region'] = fish.data_paths['raw'].name[:fish.data_paths['raw'].name.rfind('-')]

            dfs.append(df)

        self.unrolled_df = pd.concat(dfs, ignore_index=True)
        self.unrolled_df.to_hdf(self.folder_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal', mode='w')

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

    def find_twitches(self, pulse_frames, **kwargs):
        """
        Identify and filter twitches for each fish in the dataset.
        This method processes twitch events detected for each fish, removes twitches
        that are too close to pulse frames or other twitches, and returns a dictionary
        of filtered twitch events.
        Parameters:
            pulse_frames (dict): A dictionary where keys are plane indices (as integers)
                and values are lists or arrays of pulse frame indices for the corresponding plane.
            **kwargs: Additional keyword arguments passed to the `find_twitches` method
                of each fish object.
        Returns:
            dict: A dictionary where keys are fish IDs and values are arrays of unique,
                filtered twitch frame indices for each fish.
        Notes:
            - Twitches that occur within 1 frame of a pulse frame are removed.
            - Twitches that occur within 1 frame of each other across different planes
              are also removed, keeping the smaller frame index of the conflicting twitches.
        """
        twitches_dict = {fish.fish_id: [] for fish in self.fish}
        
        for fish in self.fish:
            twitches = fish.find_twitches(plot=False, **kwargs)
            
            for plane in twitches:
                pfs = pulse_frames[int(plane)]
                tws = twitches[plane]
                
                # Remove twitches that are within 1 frame of a pulse frame
                tws = np.array([tw for tw in tws if not any(abs(tw - pf) <= 1 for pf in pfs)])
                
                if len(tws) > 0:
                    twitches_dict[fish.fish_id].extend(tws)
            
            twitches_dict[fish.fish_id] = np.unique(twitches_dict[fish.fish_id])
            
            # Remove twitches that are within 1 frame of each other due to twitches from different planes
            # Keep the smaller frame of the twitch
            twitches_dict[fish.fish_id] = twitches_dict[fish.fish_id][np.diff(np.insert(twitches_dict[fish.fish_id], 0, -2)) > 1]
        
        return twitches_dict