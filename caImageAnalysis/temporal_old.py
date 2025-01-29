from kneed import KneeLocator
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
from pathlib import Path
import plotly.express as px
import random
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import kmeans2
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, median_abs_deviation, mstats, sem, spearmanr, t, wilcoxon, zscore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
import umap

from .mesm import get_plane_number, load_mesmerize, load_rois, uuid_to_plane
from .utils import save_pickle


def save_temporal(fish):
    '''Saves the temporal components of final ROIs as a temporal.h5 file
    Also calculates the dF/F0 and adds it to the dataframe'''
    mes_df = uuid_to_plane(load_mesmerize(fish))
    final_rois = load_rois(fish)

    planes = list()
    raw_temporal = list()
    temporal = list()
    roi_indices = list()
    pulse_frames = list()

    for i, row in mes_df.iterrows():
        if row.algo == 'cnmf':

            try:
                plane = get_plane_number(row)

                name = row['item_name']
                if name not in final_rois.keys():
                    continue

                indices = final_rois[name]

                raw_temp = row.cnmf.get_temporal("good", add_residuals=True)  # raw temporal responses: C+YrA
                raw_temporal.append(raw_temp[indices])

                planes.append(int(plane))
                roi_indices.append(indices)

                temp = row.cnmf.get_temporal('good')  # denoised temporal responses: C
                temporal.append(temp[indices])

                fts = pd.read_hdf(fish.data_paths['volumes'][plane]['frametimes'])
                try:
                    pulses = [fts[fts.pulse == pulse].index.values[0] for pulse in fts.pulse.unique() if pulse != fts.loc[0, 'pulse']]
                except:
                    pulses = [0]

                if 'DOI' in str(fish.data_paths['raw']) and pulses == [0]:
                    with os.scandir(fish.exp_path) as entries:
                        for entry in entries:
                            if '_pre' in entry.name:
                                pre_path = Path(entry.path)

                    pulses = [len([file for file in os.listdir(pre_path) if file.endswith('.ome.tif')])]

                pulse_frames.append(pulses)

                print(f'finished plane {plane}')
            
            except ValueError:
                # if none of the cells made it
                pass

    temporal_df = pd.DataFrame({'plane': planes,
                                'raw_temporal': raw_temporal,
                                'temporal': temporal,
                                'roi_indices': roi_indices,
                                'pulse_frames': pulse_frames})
    temporal_df.sort_values(by=['plane'], ignore_index=True, inplace=True)
    temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    fish.process_bruker_filestructure()


def save_temporal_volume(fish, indices=None, key=None):
    '''Combines the temporal components of each plane to a volumetric numpy array of shape (# of cells, frames)'''
    temporals = []

    for _, row in fish.temporal_df.iterrows():

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
    save_pickle(new_temp, fish.exp_path.joinpath('vol_temporal.pkl'))

    fish.process_bruker_filestructure()

    return new_temp


def compute_median_dff(fish, window=None):
    '''Computes the dF/F signal for each component manually by using the signal median.
    F0 is calculated as the baseline median.
    If a window is an integer n, calculates baseline as the first n frames.
    If window is None, calculates baseline as the time before the first pulse.'''
    fish.temporal_df['dff'] = None
    fish.temporal_df['raw_dff'] = None

    for i, row in fish.temporal_df.iterrows():
        pulse_0 = row.pulse_frames[0]
        dffs = list()
        raw_dffs = list()

        for comp in row.temporal:  # for each component
            if window is None:
                baseline = comp[:pulse_0]
            else:
                baseline = comp[:window]
            
            f0 = np.median(baseline)
            dff = (comp - f0)/abs(f0)
            dffs.append(dff)

        for comp in row.raw_temporal:  # for each component
            if window is None:
                baseline = comp[:pulse_0]
            else:
                baseline = comp[:window]
            
            f0 = np.median(baseline)
            dff = (comp - f0)/abs(f0)
            raw_dffs.append(dff)

        fish.temporal_df['dff'][i] = dffs
        fish.temporal_df['raw_dff'][i] = raw_dffs

    fish.temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    return fish.temporal_df


def normalize_dff(fish):
    '''Normalizes both the raw and denoised dF/F signals between 0 and 1'''
    fish.temporal_df['norm_dff'] = None
    fish.temporal_df['raw_norm_dff'] = None

    for i, row in fish.temporal_df.iterrows():
        norm_dffs = list()
        raw_norm_dffs = list()

        for comp in row.dff:
            norm_dff = (comp - min(comp)) / (max(comp) - min(comp))
            norm_dffs.append(norm_dff)

        for comp in row.raw_dff:
            raw_norm_dff = (comp - min(comp)) / (max(comp) - min(comp))
            raw_norm_dffs.append(raw_norm_dff)

        fish.temporal_df['norm_dff'][i] = norm_dffs
        fish.temporal_df['raw_norm_dff'][i] = raw_norm_dffs

    fish.temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    return fish.temporal_df


def add_coms_to_temporaldf(fish):
    '''Adds a column for centers of mass for each "good" neuron'''
    fish.temporal_df["coms"] = None

    mes_df = uuid_to_plane(load_mesmerize(fish))
    for i, row in fish.temporal_df.iterrows():
        plane = f'img_stack_{row.plane}'
        mes_row = mes_df[(mes_df.algo == 'cnmf') & (mes_df.item_name == plane)].iloc[0]

        _, coms = mes_row.cnmf.get_contours('good', swap_dim=False)  
        coms = np.array(coms)
        coms = coms[row.roi_indices]  # get the accepted components
        
        fish.temporal_df["coms"][i] = coms

    fish.temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    return fish.temporal_df


def zscore_temporaldf(fish):
    '''Z-scores both the raw and denoised signals'''
    fish.temporal_df['zscore'] = None
    fish.temporal_df['raw_zscore'] = None

    for i, row in fish.temporal_df.iterrows():
        zscores = list()
        raw_zscores = list()

        for comp in row.temporal:
            zs = zscore(comp)
            zscores.append(zs)

        for comp in row.raw_temporal:
            raw_zscore = zscore(comp)
            raw_zscores.append(raw_zscore)

        fish.temporal_df['zscore'][i] = zscores
        fish.temporal_df['raw_zscore'][i] = raw_zscores

    fish.temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    return fish.temporal_df


def percentile_norm_temporaldf(fish, percentile=75):
    '''Normalizes both the raw and denoised signals based on their 75th percentile'''
    fish.temporal_df['percentile'] = None
    fish.temporal_df['raw_percentile'] = None

    for i, row in fish.temporal_df.iterrows():
        percentiles = list()
        raw_percentiles = list()

        for comp in row.temporal:
            prc = comp/np.percentile(comp, percentile)
            percentiles.append(prc)

        for comp in row.raw_temporal:
            raw_prc = comp/np.percentile(comp, percentile)
            raw_percentiles.append(raw_prc)

        fish.temporal_df['percentile'][i] = percentiles
        fish.temporal_df['raw_percentile'][i] = raw_percentiles

    fish.temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    return fish.temporal_df


def umap_clustering(df, filterby=None, colorby=None, key='raw_norm_temporal', savefig=False, save_path=None):
    '''UMAP clustering on individual neuron responses
    filterby: runs separate clustering based on the filters
    colorby: colors each point based on the filter'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")

    if colorby not in df.columns:
        raise ValueError("Given colorby filter is not a column in the df")

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
                    
            reducer = umap.UMAP(random_state=1113)
            scaled_data = StandardScaler().fit_transform(traces)
            embedding = reducer.fit_transform(scaled_data)
                    
            plt.figure(figsize=(10, 10))
                    
            # plot t-SNE embedding in two-dimentional space
            if colorby is None:
                plt.scatter(embedding[:, 0], embedding[:, 1])

            else:
                for filt in subdf[colorby].unique():
                    plt.scatter(embedding[np.where(subdf[colorby] == filt), 0], embedding[np.where(subdf[colorby] == filt), 1], label=filt)

            plt.legend()
            plt.xlabel('UMAP_1')
            plt.ylabel('UMAP_2')
            plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

            if savefig:
                plt.savefig(save_path.joinpath("umap_clustering_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)

    else:
        traces = np.array(df.loc[:, key])
        traces = np.array([np.array(trace) for trace in traces])
                
        reducer = umap.UMAP(random_state=1113)
        scaled_data = StandardScaler().fit_transform(traces)
        embedding = reducer.fit_transform(scaled_data)
                
        plt.figure(figsize=(10, 10))
                
        # plot t-SNE embedding in two-dimentional space
        if colorby is None:
            plt.scatter(embedding[:, 0], embedding[:, 1])

        else:
            for filt in df[colorby].unique():
                plt.scatter(embedding[np.where(df[colorby] == filt), 0], embedding[np.where(df[colorby] == filt), 1], label=filt)

        plt.legend()
        plt.xlabel('UMAP_1')
        plt.ylabel('UMAP_2')
        plt.title("UMAP clustering", fontsize=18)

        if savefig:
                plt.savefig(save_path.joinpath("umap_clustering.pdf"), transparent=True)


def tsne_clustering(df, filterby=None, colorby=None, key='raw_norm_temporal', savefig=False, save_path=None):
    '''t-SNE clustering on individual neuron responses
    filterby: runs separate clustering based on the filters
    colorby: colors each point based on the filter'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")

    if colorby not in df.columns:
        raise ValueError("Given colorby filter is not a column in the df")

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
                    
            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', random_state=888).fit_transform(traces)
                    
            plt.figure(figsize=(10, 10))
                    
            # plot t-SNE embedding in two-dimentional space
            if colorby is None:
                plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

            else:
                for filt in subdf[colorby].unique():
                    plt.scatter(X_embedded[np.where(subdf[colorby] == filt), 0], X_embedded[np.where(subdf[colorby] == filt), 1], label=filt)

            plt.legend()
            plt.xlabel('t-SNE_1')
            plt.ylabel('t-SNE_2')
            plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

            if savefig:
                plt.savefig(save_path.joinpath("tsne_clustering_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)

    else:
        traces = np.array(df.loc[:, key])
        traces = np.array([np.array(trace) for trace in traces])
                
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', random_state=888).fit_transform(traces)
                
        plt.figure(figsize=(10, 10))
                
        # plot t-SNE embedding in two-dimentional space
        if colorby is None:
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

        else:
            for filt in df[colorby].unique():
                plt.scatter(X_embedded[np.where(df[colorby] == filt), 0], X_embedded[np.where(df[colorby] == filt), 1], label=filt)

        plt.legend()
        plt.xlabel('t-SNE_1')
        plt.ylabel('t-SNE_2')
        plt.title("t-SNE clustering", fontsize=18)

        if savefig:
            plt.savefig(save_path.joinpath("tsne_clustering.pdf"), transparent=True)


def pca_clustering(df, filterby=None, colorby=None, key='raw_norm_temporal', n_components=3):
    '''PCA clustering on individual neuron responses. Plots 3D components.
    filterby: runs separate clustering based on the filters
    colorby: colors each point based on the filter'''
    if colorby not in df.columns:
        raise ValueError("Given colorby filter is not a column in the df")

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
            
            if n_components == 3:
                if colorby is not None:
                    fig = px.scatter_3d(
                        components, x=0, y=1, z=2, color=subdf[colorby],
                        title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
                        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}, width=512, height=512
                    )
                else:
                    fig = px.scatter_3d(
                        components, x=0, y=1, z=2,
                        title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
                        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}, width=512, height=512
                    )
            elif n_components == 2:
                if colorby is not None:
                    fig = px.scatter(
                        components, x=0, y=1, color=subdf[colorby],
                        title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
                        labels={'0': 'PC 1', '1': 'PC 2'}, width=512, height=512
                    )
                else:
                    fig = px.scatter(
                        components, x=0, y=1,
                        title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
                        labels={'0': 'PC 1', '1': 'PC 2'}, width=512, height=512
                    )

            fig.show()

    else:
        traces = np.array(df.loc[:, key])
        traces = np.array([np.array(trace) for trace in traces])
                
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(traces)

        total_var = pca.explained_variance_ratio_.sum() * 100
        
        if n_components == 3:
            if colorby is not None:
                fig = px.scatter_3d(
                    components, x=0, y=1, z=2, color=df[colorby],
                    title=f'Total Explained Variance: {total_var:.2f}%',
                    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}, width=512, height=512
                )
            else:
                fig = px.scatter_3d(
                    components, x=0, y=1, z=2,
                    title=f'Total Explained Variance: {total_var:.2f}%',
                    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}, width=512, height=512
                )

        elif n_components == 2:
            if colorby is not None:
                fig = px.scatter(
                    components, x=0, y=1, color=df[colorby],
                    title=f'Total Explained Variance: {total_var:.2f}%',
                    labels={'0': 'PC 1', '1': 'PC 2'}, width=512, height=512
                )
            else:
                fig = px.scatter(
                    components, x=0, y=1, 
                    title=f'Total Explained Variance: {total_var:.2f}%',
                    labels={'0': 'PC 1', '1': 'PC 2'}, width=512, height=512
                )

        fig.show()


def kmeans_clustering(data, k):
    '''Takes in an array of neural responses and performs k-means clustering on them'''
    # K-means relies on random number generation, we can fix the seed to have same result each time 
    centroid, labels = kmeans2(data, k, seed=1111111, minit='points')

    kmeans_clusters = dict()

    for i, cl in enumerate(labels):
        if cl+1 not in kmeans_clusters.keys():
            kmeans_clusters[cl+1] = list()

        kmeans_clusters[cl+1].append(data[i])

    return kmeans_clusters, centroid


def hierarchical_clustering(data, max_inter_cluster_dist, save_path=str()):
    '''Takes in an array of neural responses and performs hierarchical clustering on them'''
    # Hierarchical clustering of temporal responses
    d = sch.distance.pdist(data, metric='cosine')  # pairwise distances between observations
    Z = sch.linkage(d, method='complete')

    # Flat clustering of the dendrogram
    T = sch.fcluster(Z, max_inter_cluster_dist, criterion='distance', depth=3)
    print(f'Number of clusters: {T.max()}')

    sort_inds = np.argsort(T)  # indices for ordering components by cluster number

    fig = plt.figure(figsize=(25, 10))
    dn = sch.dendrogram(Z)
    plt.hlines(max_inter_cluster_dist, 0, len(d), color='r')
    plt.title(f'Clustering: max_inter_cluster_dist={max_inter_cluster_dist}')
    plt.grid(visible=False)
    plt.show()

    if len(str(save_path)) != 0:
       plt.savefig(save_path.joinpath(f"hierarchical_clustering_{max_inter_cluster_dist}.pdf"), transparent=True)
    print("saved dendrogram") 

    # Create a clusters dictionary to store all temporal responses per cluster
    clusters = dict()
    for i, cluster in enumerate(T[sort_inds]):
        if cluster in clusters:
            clusters[cluster].append(data[sort_inds][i])
        else:
            clusters[cluster] = [data[sort_inds][i]]
    print("made clusters dict")

    return clusters, T


def sort_by_peak(data, window=10):
    '''Sorts an array by the peak values using a sum of sliding window'''
    sorted_data = sorted(data, key=lambda arr: np.argmax(np.convolve(arr, np.ones(window), 'valid')))

    return sorted_data


def sort_clusters_by_peak(clusters):
    '''Sorts a dictionary of clustered traces by the peak values using a sum of sliding window'''
    peak_clusters = dict()

    for cl in clusters:
        sorted_ts = sort_by_peak(clusters[cl])
        inds = list()
        for t in clusters[cl]:
            i = np.argmax(t)
            inds.append(i)
        peak_clusters[cl] = np.median(inds)


def cluster_temporal(fish, max_inter_cluster_dist, sort=True, savefig=False):
    '''
    Clusters temporal responses using hierarchical and flat clustering
    
    max_inter_cluster_dist: threshold for maximum inter-cluster distance allowed
    sort: sorts the cluster keys in ascending order if True
    '''
    # Hierarchical clustering of temporal responses
    d = sch.distance.pdist(fish.vol_temporal, metric='cosine')  # pairwise distances between observations
    Z = sch.linkage(d, method='complete')

    # Flat clustering of the dendrogram
    T = sch.fcluster(Z, max_inter_cluster_dist, criterion='distance', depth=3)
    print(f'Number of clusters: {T.max()}')

    sort_inds = np.argsort(T)  # indices for ordering components by cluster number

    # Create a clusters dictionary to store all temporal responses per cluster
    clusters = dict()
    for i, cluster in enumerate(T[sort_inds]):
        if cluster in clusters:
            clusters[cluster].append(fish.vol_temporal[sort_inds][i])
        else:
            clusters[cluster] = [fish.vol_temporal[sort_inds][i]]

    # Create a dictionary that holds the median of maximum peak indices in a cluster
    peak_clusters = dict()
    for cl in clusters:
        inds = []
        for t in clusters[cl]:
            i = np.argmax(t)
            inds.append(i)
        peak_clusters[cl] = np.median(inds)

    # Group the centers of mass of spatial components based on clusters
    mes_df = uuid_to_plane(load_mesmerize(fish))
    all_coms = list()

    for i, row in fish.temporal_df.iterrows():
        plane = f'img_stack_{row.plane}'
        mes_row = mes_df[(mes_df.algo == 'cnmf') & (mes_df.item_name == plane)].iloc[0]
        _, coms = mes_row.cnmf.get_contours('good', swap_dim=False)  
        coms = np.array(coms)
        coms = coms[row.roi_indices]  # get the accepted components
        all_coms.append(coms)

    all_coms = np.concatenate(all_coms)

    com_clusters = dict()
    for i, cluster in enumerate(T[sort_inds]):
        if cluster in com_clusters:
            com_clusters[cluster].append(all_coms[sort_inds][i])
        else:
            com_clusters[cluster] = [all_coms[sort_inds][i]]
    
    fig = plt.figure(figsize=(25, 10))
    dn = sch.dendrogram(Z)
    plt.axhline(y=max_inter_cluster_dist, color='r')
    plt.title(f'Clustering: max_inter_cluster_dist={max_inter_cluster_dist}')
    plt.show()

    if savefig:
        plt.savefig(fish.exp_path.joinpath(f"hierarchical_clustering_{max_inter_cluster_dist}.pdf"), transparent=True)

    if sort:  
        # Sorted cluster keys      
        sorted_keys = sorted(clusters, key=lambda k: len(clusters[k]), reverse=True)  
        sorted_peak_keys = sorted(peak_clusters, key=lambda k: peak_clusters[k])
        
        # Actually sorted clusters
        clusters = {key: clusters[key] for key in sorted_keys}
        peak_clusters = {key: clusters[key] for key in sorted_peak_keys}
        com_clusters = {key: com_clusters[key] for key in sorted_peak_keys}
    
    fish.clusters = {
        'max_inter_cluster_dist': max_inter_cluster_dist,
        'sort_inds': sort_inds,
        'clusters': clusters,
        'peak_clusters': peak_clusters,
        'com_clusters': com_clusters
    }

    save_pickle(fish.clusters, fish.exp_path.joinpath('clusters.pkl'))
    
    return fish.clusters


def plot_heatmap(data, sort=True, fps=1.3039181000348583, pulses=[391, 548, 704, 861, 1017], tick_interval=60):
    '''Plots temporal components
    tick_interval: how long the duration between each tick should be (in seconds)'''
    # Just simple heatmaps
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


def sort_by_peak_with_indices(data, separate_array=None, window=10):
    """
    Sorts the data based on peak response times and applies the same sorting to a separate array (if provided).

    Parameters:
    - data: Numpy array to sort (e.g., neuron responses across time).
    - separate_array: Optional array to apply the same sorting indices to.
    - window: Sliding window size for smoothing the data.

    Returns:
    - sorted_data: Data sorted by peak response times.
    - sorted_separate_array: The separate array sorted using the same indices (if provided).
    - sorting_indices: Indices used for sorting.
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


def plot_temporal_plane(fish, plane, indices=None, heatmap=False, key=None):
    '''Plots individual temporal components of a plane'''
    row = fish.temporal_df[fish.temporal_df.plane == plane].iloc[0]

    if key is not None:
        temporal = np.array(row[key])
    else:
        temporal = row.temporal

    if indices is not None:
        temporal = temporal[indices]

    if not heatmap:
        fig = plt.figure(4, figsize=(10, temporal.shape[0]))
        gs = fig.add_gridspec(temporal.shape[0], hspace=0)
        axs = gs.subplots(sharex=True)

        for i, t in enumerate(temporal):
            axs[i].plot(t) # np.arange(len(t))/fish.fps, as x
            for pulse in row.pulse_frames:
                axs[i].vlines(pulse, t.min(), t.max(), colors='r')

        plt.xlabel('Time (s)')
        
        ticks = np.arange(0, 16*60*fish.fps, 60*fish.fps)
        plt.xticks(ticks=ticks, labels=np.round(ticks/fish.fps).astype(int))
        plt.show()

    else:
        plot_heatmap(temporal, fps=fish.fps, pulses=fish.get_pulse_frames())
        plt.title(f'Plane {plane}: Temporal heatmap')
        plt.show()


def plot_temporal_volume(fish, data=None, title=None, clusters=None, savefig=False, **kwargs):
    '''Plots heatmap of individual temporal components of a volume'''
    if len(kwargs) > 0:
        save_temporal_volume(fish, **kwargs)

    if data is None:
        data = fish.vol_temporal

    if title is None:
        title = 'Temporal heatmap'

    if clusters is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, width_ratios=[20, 1], height_ratios=[1], figsize=(20, 10))
        ax1.imshow(data, cmap='plasma', interpolation='nearest', aspect='auto')
        
        ticks = np.arange(0, 16*60*fish.fps, 60*fish.fps)
        ax1.set_xticks(ticks=ticks, labels=np.round(ticks/fish.fps).astype(int))
        
        try:
            for pulse in fish.get_pulse_frames():
                ax1.vlines(pulse, -0.5, len(data)-0.5, color='r')
        except:
            pass
        
        ax1.title.set_text(title)
        ax1.set_xlabel('Time (s)')

        bottom = -0.5  # y-coordinates of the bottom side of the bar
        x = 0
        width = 0.5

        cmap = get_cmap('Set1')  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors  # type: list
        ax2.set_prop_cycle(color=colors)

        # Make the colormap for the clusters
        for cluster, temps in clusters.items():
            p = ax2.bar(x, len(temps), width, label=str(cluster), bottom=bottom)
            bottom += len(temps)
            
            ax2.bar_label(p, labels=[str(cluster)], label_type='center')

        plt.subplots_adjust(wspace=0)

    else:
        # Just simple heatmaps
        fig = plt.figure(figsize=(20, 10))
        plt.imshow(data, cmap='plasma', interpolation='nearest', aspect='auto')
        
        ticks = np.arange(0, 16*60*fish.fps, 60*fish.fps)
        plt.xticks(ticks=ticks, labels=np.round(ticks/fish.fps).astype(int))
        
        try:
            for pulse in fish.get_pulse_frames():
                plt.vlines(pulse, -0.5, len(data)-0.5, color='r')
        except:
            pass
        
        plt.title(title)
        plt.xlabel('Time (s)')

    if savefig:
        plt.savefig(fish.exp_path.joinpath("heatmap_clusters.pdf"), transparent=True)

    plt.show()


def plot_representative_trace(fish, clusters, savefig=False):
    '''Given a clusters dictionary, plots a random temporal trace per cluster'''
    fig, axes = plt.subplots(len(clusters), 1, sharex=True, sharey=True, figsize=(20, 20))

    for i, (cluster, temp) in enumerate(clusters.items()):
        t = random.choice(temp)
        axes[i].plot(t)
        axes[i].title.set_text(cluster)

        try:
            for pulse in fish.get_pulse_frames():
                axes[i].vlines(pulse, 0, 1, color='r')
        except:
            pass

    ticks = np.arange(0, 16*60*fish.fps, 60*fish.fps)
    plt.xticks(ticks=ticks, labels=np.round(ticks/fish.fps).astype(int))
    plt.xlabel('Time (s)')

    if savefig:    
        plt.savefig(fish.exp_path.joinpath("cluster_representative_traces.pdf"), transparent=True)
    

def plot_average_traces(clusters, fish=None, pulses=list(), save_path=str(), savefig=False, fps=None):
    '''Given a clusters dictionary, plots the mean temporal trace per cluster'''
    fig, axes = plt.subplots(len(clusters), 1, sharex=True, sharey=True, figsize=(20, 20))

    if fish is not None and len(pulses) == 0:
        pulses = fish.get_pulse_frames()

    for i, (cluster, temp) in enumerate(clusters.items()):
        t = np.mean(temp, axis=0)
        err = sem(temp)
        x = np.linspace(0, len(t), len(t))

        axes[i].plot(t)
        axes[i].fill_between(x, t-err, t+err, alpha=0.2)
        axes[i].title.set_text(cluster)

        try:
            for pulse in pulses:
                axes[i].vlines(pulse, 0, 1, color='r')
        except:
            pass

    if fish is not None:
        ticks = np.arange(0, 16*60*fish.fps, 60*fish.fps)
        plt.xlabel('Time (s)')
        plt.xticks(ticks=ticks, labels=np.round(ticks/fish.fps).astype(int))
    elif fish is None and fps is not None:
        ticks = np.arange(0, 16*60*fps, 60*fps)
        plt.xlabel('Time (s)')
        plt.xticks(ticks=ticks, labels=np.round(ticks/fps).astype(int))
    elif fish is None and fps is None:
        ticks = np.arange(0, 16*60, 60)
        plt.xlabel('Frames')
        plt.xticks(ticks=ticks, labels=ticks)
    
    if savefig:
        if fish is not None and len(save_path) == 0:
            plt.savefig(fish.exp_path.joinpath("cluster_average_traces.pdf"), transparent=True)
        elif len(save_path) != 0:
            plt.savefig(save_path.joinpath("cluster_average_traces.pdf"), transparent=True)
        else:
            print("No save_path provided. Cannot save figure.")


def plot_spatial_overlayed(fish, img, clusters=None, vmin=0, vmax=360, savefig=False):
    '''
    Plots spatial components of given clusters overlayed on an image
    clusters: if None, will plot all clusters by default. if given a list, it will plot the clusters in the list. 
    '''
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)

    cmap = get_cmap('Set1')  # type: matplotlib.colors.ListedColormap
    cmap2 = get_cmap('Accent')  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors + cmap2.colors  # type: list

    for i, (cluster, coms) in enumerate(fish.clusters['com_clusters'].items()):
        if clusters is None or (clusters is not None and cluster in clusters):
            
            for j, com in enumerate(coms):
                try:
                    if j == 0:  # add a label for the first component of each cluster
                        plt.scatter(com[0]*(1024 / img.shape[1]*2), com[1]*(1024 / img.shape[0]*2), s=150, color=colors[i], label=cluster)
                    else:
                        plt.scatter(com[0]*(1024 / img.shape[1]*2), com[1]*(1024 / img.shape[0]*2), s=150, color=colors[i])
                
                except IndexError:  # if we run out of colors
                    colors += colors

                    if j == 0:  # add a label for the first component of each cluster
                        plt.scatter(com[0]*(1024 / img.shape[1]*2), com[1]*(1024 / img.shape[0]*2), s=150, color=colors[i], label=cluster)
                    else:
                        plt.scatter(com[0]*(1024 / img.shape[1]*2), com[1]*(1024 / img.shape[0]*2), s=150, color=colors[i])

    plt.legend()
            
    if savefig:
        plt.savefig(fish.exp_path.joinpath("clusters_spatial_overlayed.pdf"), transparent=True)


def plot_spatial_individual(fish, img, clusters=None, vmin=0, vmax=360, distribution=False, savefig=False):
    '''Plots spatial components of given clusters separately on individual images'''
    com_clusters = dict()
    if clusters is None:
        com_clusters = fish.clusters['com_clusters']
    else:
        for cl in clusters:
            com_clusters[cl] = fish.clusters['com_clusters'][cl]

    # The number of rows and figure size depend on the number of clusters
    n_cols = 2
    n_rows = np.ceil(len(com_clusters) / n_cols)
    fig, axes = plt.subplots(int(n_rows), n_cols, sharex=True, figsize=(15, np.ceil(len(com_clusters) / n_cols) * 5))

    cmap = get_cmap('Set1')  # type: matplotlib.colors.ListedColormap
    cmap2 = get_cmap('Accent')  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors + cmap2.colors  # type: list

    for i, (cluster, coms) in enumerate(com_clusters.items()):
        img_row = int(i / n_cols)
        
        axes[img_row, int(i % n_cols)].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[img_row, int(i % n_cols)].title.set_text(cluster)
        
        xcoords = list()  # list of x coordinates if distribution is True
        ycoords = list() # list of y coordinates if distribution is True
        
        for com in coms:
            try:
                axes[img_row, int(i % n_cols)].scatter(com[0]*(1024 / img.shape[1]*2), com[1]*(1024 / img.shape[0]*2), s=50, color=colors[i])
            except IndexError:  # if we run out of colors
                colors += colors
                axes[img_row, int(i % n_cols)].scatter(com[0]*(1024 / img.shape[1]*2), com[1]*(1024 / img.shape[0]*2), s=50, color=colors[i])
            
            xcoords.append(com[0]*(1024 / img.shape[1]))
            ycoords.append(com[1]*(1024 / img.shape[0]))
        
        if distribution:
            divider = make_axes_locatable(axes[img_row, int(i % n_cols)])
            axtop = divider.append_axes("top", size=0.5, pad=0.3, sharex=axes[img_row, int(i % n_cols)])
            axright = divider.append_axes("right", size=0.5, pad=0.4, sharey=axes[img_row, int(i % n_cols)])
            
            if len(xcoords) > 1:
                xdensity = gaussian_kde(xcoords)
                xdensity.covariance_factor = lambda : .25
                xdensity._compute_covariance()

                ydensity = gaussian_kde(ycoords)
                ydensity.covariance_factor = lambda : .25
                ydensity._compute_covariance()

                xs = np.linspace(0, img.shape[1], 200)
                ys = np.linspace(0, img.shape[0], 200)

                axtop.plot(xs, xdensity(xs))
                axtop.xaxis.set_major_locator(ticker.NullLocator())
                axright.plot(ydensity(ys), ys)

                #adjust margins
                axtop.margins(x=0)
                axright.margins(y=0)

    plt.tight_layout()

    if savefig:
        plt.savefig(fish.exp_path.joinpath("clusters_spatial_individual.pdf"), transparent=True)


def find_stimulus_responsive(fish, pre_frame_num=15, post_frame_num=5, peak_threshold=None, min_threshold=None, key=None, normalize=False, 
                             normalize_by_first=False):
    '''Identifies stimulus responsive neurons
    pre_frame_num: number of frames before the pulse. 4 frames is ~3 seconds
    post_frame_num: number of frames after the pulse. 13 frames is ~10 seconds
    peak_threshold: minimum normalized fluorescence intensity for an activated neuron
    min_threshold: maximum normalized fluorescence intensity for an inhibited neuron
    baseline_threshold: maximum average baseline value for activated neurons
    normalize: if True, takes the pre-pulse period as baseline and calculates dF per pulse 
    normalize_by_first: if True, takes the pre-pulse period of the FIRST pulse as baseline and calculates dF per pulse
    '''
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
    inhibited_neurons = list()  # list of neuron indices that are inhibited by the injection on average
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
        # 0 means it's inhibited by the pulse, 1 means it's activated by the pulse
        neuron_pulse_response = list()  

        # Activated neurons: If the peak response in "post" is bigger
        # than 2 times the "pre" standard deviation, the neuron is stimulus 
        # responsive
        activated_thresh = np.median(np.array(avg_trace)[:pre_frame_num]) + (2 * pre_stdev)

        # Inhibited neurons: If the minimum response in "post" is smaller
        # than 2 times the "pre" standard deviation, the neuron is stimulus 
        # responsive
        inhibited_thresh = np.median(np.array(avg_trace)[:pre_frame_num]) - (2 * pre_stdev)

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

        elif check_if_inhibited(avg_trace, inhibited_thresh, pre_frame_num=pre_frame_num, min_threshold=min_threshold):
            responsive = True
            stim_responsive_neurons.append(i)
            inhibited_neurons.append(i)
            print(f'neuron {i} is inhibited')

            for t, trace in enumerate(traces):
                # now let's determine how many of the stimuli individual neurons are responding to
                pre_stdev = np.array(trace)[:pre_frame_num].std()
                inhibited_thresh = np.median(np.array(trace)[:pre_frame_num]) - (2 * pre_stdev)

                if check_if_inhibited(trace, inhibited_thresh, pre_frame_num=pre_frame_num, min_threshold=min_threshold):
                    response_count += 1
                    neuron_pulse_response.append((t+1, 0))
                    print(f'neuron {i} responds to stimulus {t+1} (inhibited)')

        if responsive:
            print(f'neuron {i} responds to {(response_count/len(traces))*100}% of injections\n')
            pulse_responses.append(neuron_pulse_response)

    print(f'{len(stim_responsive_neurons)} out of {len(neurons)} neurons is stimulus responsive: {len(stim_responsive_neurons)/len(neurons)*100}%')
    print(f'number of inhibited neurons: {len(inhibited_neurons)}')
    
    if len(stim_responsive_neurons) != 0:
        print(f'% of inhibited neurons: {len(inhibited_neurons)/len(stim_responsive_neurons)*100}')
    else:
        print(f'% of inhibited neurons: 0%')

    print(f'number of activated neurons: {len(activated_neurons)}')

    if len(stim_responsive_neurons) != 0:
        print(f'% of activated neurons: {len(activated_neurons)/len(stim_responsive_neurons)*100}')
    else:
        print(f'% of activated neurons: 0%')

    return stim_responsive_neurons, activated_neurons, inhibited_neurons, pulse_responses


def check_if_activated(trace, threshold, pre_frame_num=15, peak_threshold=None):
    '''Checks if a neural trace is activated
    pre_frame_num: number of frames before the pulse. 4 frames is ~3 seconds'''
    peak_response = trace[pre_frame_num:].max()
    mdn_baseline = np.median(trace[:pre_frame_num])

    if peak_threshold is None and mdn_baseline != 0:
        peak_threshold = abs(mdn_baseline * 0.2) + mdn_baseline
    elif peak_threshold is None and mdn_baseline == 0:
        # if average baseline is normalized to be 0, set the peak threshold to 20%
        peak_threshold = 0.2

    # Activated neurons: If the peak response in "post" is bigger
    # than the threshold, the neuron is stimulus responsive
    if peak_response > threshold and peak_response > peak_threshold:
        return True
    else:
        False


def check_if_inhibited(trace, threshold, pre_frame_num=15, min_threshold=None):
    '''Checks if a neural trace is inhibited
    pre_frame_num: number of frames before the pulse. 4 frames is ~3 seconds'''
    min_response = trace[pre_frame_num:].min()
    mdn_baseline = np.median(trace[:pre_frame_num])

    if min_threshold is None and mdn_baseline != 0:
        min_threshold = abs(mdn_baseline * 0.2) - mdn_baseline
    elif min_threshold is None and mdn_baseline == 0:
        # if average baseline is normalized to be 0, set the min threshold to 20%
        min_threshold = -0.2

    # Inhibited neurons: If the minimum response in "post" is smaller
    # than the threshold, the neuron is stimulus responsive
    if min_response < threshold and min_response < min_threshold:
        return True
    else:
        False


def plot_neuron_pulse_average(fish, pre_frame_num=15, post_frame_num=5, peak_threshold=None, min_threshold=None, key=None, savefig=False, 
                              n_cols=3, normalize=False, normalize_by_first=False, sharey=True):
    '''Plots the average response of each neuron to the pulses'''
    if key is None:
        key = 'raw_norm_temporal'

    if normalize and normalize_by_first:
        raise ValueError('normalize and normalize_by_first cannot both be True. Pick one method to normalize.')
    
    stim_responsive_neurons, _, _, _ = find_stimulus_responsive(fish, pre_frame_num=pre_frame_num, post_frame_num=post_frame_num, 
                                                                peak_threshold=peak_threshold, min_threshold=min_threshold, key=key,
                                                                normalize=normalize, normalize_by_first=normalize_by_first)

    neurons = list()
    pulse_frames=list()

    for i, row in fish.temporal_df.iterrows():
        neurons.extend(row[key])

        for j in range(len(row[key])):  # add pulse frames for each neuron in each plane
            pulse_frames.append(row['pulse_frames'])

    n_rows = np.ceil(len(neurons) / n_cols)
    fig, axes = plt.subplots(int(n_rows), n_cols, sharex=True, sharey=sharey, figsize=(5*n_cols, np.ceil(len(neurons) / n_cols) * 5), layout='constrained')

    for i, neuron in enumerate(neurons):
        img_row = int(i / n_cols)

        pulses = pulse_frames[i]
        traces = list()

        for p, pulse in enumerate(pulses):
            start_frame = pulse - pre_frame_num  # when the neuron traces will start
            stop_frame = pulse + post_frame_num  # when the neuron traces will end

            trace = neuron[start_frame:stop_frame+1]

            if normalize or (normalize_by_first and p == 0):
                baseline = np.median(neuron[start_frame:pulse])
                trace = (trace - baseline) / baseline
            elif normalize_by_first and p != 0:
                trace = (trace - baseline) / baseline

            traces.append(trace)

            axes[img_row, int(i % n_cols)].plot(trace, 'lightgray')
            
        avg_trace = np.array(traces).mean(axis=0)
        sems = sem(np.array(traces), axis=0)
        x = np.arange(pre_frame_num+post_frame_num+1)

        axes[img_row, int(i % n_cols)].plot(x, avg_trace)
        axes[img_row, int(i % n_cols)].fill_between(x, avg_trace-sems, avg_trace+sems, alpha=0.2)
        axes[img_row, int(i % n_cols)].axvspan(pre_frame_num-1, pre_frame_num, color='red', lw=2, alpha=0.2)
        axes[img_row, int(i % n_cols)].set_title(str(i))

        if i in stim_responsive_neurons:
            axes[img_row, int(i % n_cols)].tick_params(color='red', labelcolor='red', width=5)
            for spine in axes[img_row, int(i % n_cols)].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(5)

    fig.supxlabel('Frame')
    plt.figtext(0.02, 1, f'{len(stim_responsive_neurons)} out of {len(neurons)} neurons is stimulus responsive: {len(stim_responsive_neurons)/len(neurons)*100}%', fontsize=14)    

    if savefig:
        plt.savefig(fish.exp_path.joinpath("neuron_pulse_averages.pdf"), transparent=True, bbox_inches="tight")
    

def find_twitches(fish, plot=False, **kwargs):
    '''Finds twitch events using x and y shifts from motion correction'''
    mes_df = uuid_to_plane(load_mesmerize(fish))

    all_peaks = dict()  # stores all the peaks for each plane

    for i, row in mes_df.iterrows():
        if row.algo == 'mcorr':
            # Get the shifts manually instead of using mesmerize's get_shifts() function because it returns it weird
            x_shifts, y_shifts = np.load(os.path.join(fish.data_paths['mesmerize'], row.outputs['shifts']))
            
            # Calculate the total shifts per frame as the sum of all x and y shifts
            shifts = np.sum(np.abs(x_shifts), axis=1) + np.sum(np.abs(y_shifts), axis=1)
            peaks, _ = find_peaks(shifts, **kwargs)

            plane = get_plane_number(row)
            all_peaks[plane] = peaks

            if plot:
                try:
                    pulse_frames = fish.temporal_df.loc[int(plane), 'pulse_frames']

                    plt.figure(figsize=(10, 5))
                    plt.plot(np.sum(np.abs(x_shifts), axis=1), label='x_shifts')
                    plt.plot(np.sum(np.abs(y_shifts), axis=1), label='y_shifts')
                    plt.plot(shifts, label='total_shifts')
                    plt.plot(peaks, shifts[peaks], "x", label='twitch')
                    plt.title(f'plane {plane} - twitch frames: {peaks} - pulse frames: {pulse_frames}')
                    plt.legend()
                
                except KeyError:
                    # we might not have any components identified in a plane
                    pass

    return all_peaks


def unroll_temporal_df(fish, min_pulses=3, **kwargs):
    '''Expands the temporal_df so that each row is a single neuron rather than a single plane'''
    
    unrolled_df = pd.DataFrame(columns=['fish_id', 'plane', 'neuron', 'raw_temporal', 'temporal', 'raw_norm_temporal', 'norm_temporal',
                                        'raw_dff', 'dff', 'raw_norm_dff', 'norm_dff', 'roi_index', 'com', 'pulse_frames'])
    
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
            unrolled_row['raw_dff'] = row['raw_dff'][j]
            unrolled_row['dff'] = row['dff'][j]
            unrolled_row['raw_norm_dff'] = row['raw_norm_dff'][j]
            unrolled_row['norm_dff'] = row['norm_dff'][j]
            unrolled_row['roi_index'] = row['roi_indices'][j]
            unrolled_row['com'] = row['coms'][j]
            unrolled_row['pulse_frames'] = row['pulse_frames']

            unrolled_df = pd.concat([unrolled_df, pd.DataFrame([unrolled_row])], ignore_index=True)

    stim_responsive, activated, inhibited, pulse_responses = find_stimulus_responsive(fish, **kwargs)
    
    unrolled_df['responsive'] = False
    unrolled_df['activated'] = None
    unrolled_df['inhibited'] = None
    unrolled_df['pulse_response'] = None

    for i, neuron in enumerate(stim_responsive):
        unrolled_df.at[neuron, 'pulse_response'] = pulse_responses[i]

        if len(pulse_responses[i]) >= min_pulses:
            unrolled_df.loc[neuron, 'responsive'] = True

            if neuron in activated:
                unrolled_df.loc[neuron, 'activated'] = True
                unrolled_df.loc[neuron, 'inhibited'] = False
            
            elif neuron in inhibited:
                unrolled_df.loc[neuron, 'inhibited'] = True
                unrolled_df.loc[neuron, 'activated'] = False

    unrolled_df.to_hdf(fish.exp_path.joinpath('unrolled_temporal.h5'), key='unrolled_temporal')

    fish.process_bruker_filestructure()

    return unrolled_df


def determine_baseline_oscillations(unrolled_df, oscillations=5, pre_frame_num=100, post_frame_num=0, show_peak_example=False):
    '''Determine how long the baseline frame duration should be using oscillations
    oscillations: determines the typical frame length of this many oscillations'''

    traces = list()
    x = np.arange(0-pre_frame_num, 0+post_frame_num)

    for _, neuron in unrolled_df.iterrows():
        pulses = neuron['pulse_frames']
        
        for pulse in pulses:
            start_frame = pulse - pre_frame_num  # when the neuron traces will start
            stop_frame = pulse  # when the neuron traces will end

            trace = neuron['raw_norm_temporal'][start_frame:stop_frame]
            traces.append(trace)

    n_peaks = list()
    for i, trace in enumerate(traces):
        peaks, _ = find_peaks(trace)

        if show_peak_example and i == 0:
            plt.plot(trace)
            plt.plot(peaks, trace[peaks], "x")
            plt.ylim(0, 1)
            plt.show()
        
        n_peaks.append(len(peaks))

    n, bins, _ = plt.hist(n_peaks, bins=np.arange(np.array(n_peaks).min(), np.array(n_peaks).max()+2)-0.5, density=True, align='mid')
    plt.xlabel('# of peaks in 100 frames')
    plt.show()

    print(f"ideal number of frames: {oscillations*len(x)/bins[np.argmax(n)]}")

    
def determine_baseline_sem(temporal_df, pre_frame_num=100):
    """
    Determine the optimal baseline frame duration using the standard error of the mean (SEM).
    Calculates the SEM for different baseline frame durations and identifies the optimal duration 
    using the KneeLocator method. Plots the SEM values against the number of frames before injection 
    and marks the ideal number of frames.
    Parameters:
    temporal_df (pd.DataFrame): DataFrame with temporal data containing 'pulse_frames' and 'raw_norm_temporal'.
    pre_frame_num (int): Maximum number of frames before injection to consider for baseline duration. Default is 100.
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
    plt.show()

    print(f"ideal number of frames: {kn.knee}")


def determine_baseline_mad(unrolled_df, pre_frame_num=100):
    '''Determine how long the baseline frame duration should be using the SEM'''

    x = np.arange(1, pre_frame_num+1)

    mads = list()
    for t in x:
        traces = list()
        for _, neuron in unrolled_df.iterrows():
            pulses = neuron['pulse_frames']
            
            for pulse in pulses:
                start_frame = pulse - t  # when the neuron traces will start
                stop_frame = pulse  # when the neuron traces will end

                trace = neuron['raw_norm_dff'][start_frame:stop_frame]
                traces.extend(trace)

        mads.append(median_abs_deviation(traces, axis=None))

    kn = KneeLocator(x, mads, curve='convex', direction='decreasing')

    plt.plot(x, mads)
    plt.vlines(kn.knee, 0, plt.ylim()[1], linestyles='dashed')
    plt.ylim(0)
    plt.xticks(np.arange(0, 101, 5))
    _, labels = plt.xticks()
    for label in labels[1::2]:
        label.set_visible(False)
    plt.xlabel('# of frames before injection')
    plt.ylabel('MAD of trace values')
    plt.show()

    print(f"ideal number of frames: {kn.knee}")


def get_traces(df, pre_frame_num=0, post_frame_num=10, normalize=False, 
               normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, 
               overlay_filter=None, return_col=None):
    '''From a temporal_df, gets desired traces around pulses
    Returns the x-axis (in frames) and list of individual traces
    post_frame_num: Stop frame of the traces (included)
    return_col: alongside the x-axis and traces, it also returns other df column values for the traces. 
                useful for peak and time to peak calculations for suppressed neurons'''
    x = np.arange(0-pre_frame_num, 0+post_frame_num+1)
    traces = list()

    if overlay_filter is not None:
        overlay_filters = list()

    if return_col is not None:
        return_col_list = list()

    for _, neuron in df.iterrows():
        pulses = neuron['pulse_frames']

        if only_responsive:
            responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
            pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, inhibited

            if normalize_by_first:
                baseline = 0

            for i, pulse in enumerate(responsive_pulses):
                if (pulse_activity[i] == 1 and neuron['activated'] == True) or (pulse_activity[i] == 0 and neuron['inhibited'] == True):
                    # if (neuron["fish_id"] == 36 and pulse != 4) or (neuron["fish_id"] != 36):
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

                    if overlay_filter is not None:
                        overlay_filters.append(neuron[overlay_filter])

                    if return_col is not None:
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

                traces.append(trace)

                if overlay_filter is not None:
                    overlay_filters.append(neuron[overlay_filter])

                if return_col is not None:
                    return_col_list.append(neuron[return_col])

    if overlay_filter is not None:
        return x, traces, overlay_filters
    elif return_col is not None:
        return x, traces, return_col_list
    else:
        return x, traces


def plot_pulse_averages(df, filterby, savefig=False, save_path=None, **kwargs):
    '''Plot the pulse averages for each neuron, with individual pulse traces'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
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
        x, traces = get_traces(subdf, **kwargs)

        plt.figure(figsize=(10, 10))

        for trace in traces:
            plt.plot(x, trace, 'lightgray', alpha=0.5)

        avg_trace = np.mean(np.array(traces), axis=0)
        sems = sem(np.array(traces), axis=0)

        try:
            plt.plot (x, avg_trace, zorder=102)
            plt.fill_between(x, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)
            plt.axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)

            # plt.ylim(top=20, bottom=-2)
            plt.ylim(top=2, bottom=-2)

            # if normalize or normalize_by_first:
            #     plt.ylim(top=1, bottom=-0.50)
            # else:
            #     plt.ylim(top=1, bottom=0)
                
            plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)
            
            if savefig:
                plt.savefig(save_path.joinpath("pulse_average_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)

            plt.show()
        
        except ValueError:
            # sometimes there aren't any neurons that belong to the category, so everything is None
            pass


def plot_pulse_averages_overlayed(df, overlay_filter, filterby, color_order=None, overlay_order=None, fps=1, savefig=False, save_path=None, **kwargs):
    '''Plots the pulse averages from different filters overlayed on top'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
    for filter in filterby:
        if filter not in df.columns:
            raise ValueError("Given filter is not a column in the df")
        
    if overlay_filter not in df.columns:
        raise ValueError("Given overlay_filter is not a column in the df")
        
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
        x, traces, overlay_filters = get_traces(subdf, overlay_filter=overlay_filter, **kwargs)

        plt.figure(figsize=(10, 10))

        if overlay_order is not None:  # if you want the overlay_filters to go in a specific order
            for i, of in enumerate(overlay_order):
                tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                avg_trace = np.mean(np.array(tr), axis=0)
                sems = sem(np.array(tr), axis=0)

                if color_order is not None:
                    plt.plot (x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}', color=color_order[i])
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                else:
                    plt.plot (x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}')
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

        else:  # if you don't care about the order, it will just find the unique overlay filters
            for i, of in enumerate(np.unique(overlay_filters)):
                tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                avg_trace = np.mean(np.array(tr), axis=0)
                sems = sem(np.array(tr), axis=0)

                if color_order is not None:
                    plt.plot (x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}', color=color_order[i])
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                else:
                    plt.plot (x/fps, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}')
                    plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

        plt.axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
        plt.legend()

        plt.ylim(top=2.5, bottom=-0.75)

        # if normalize or normalize_by_first:
        #     plt.ylim(top=1, bottom=-0.50)
        # else:
        #     plt.ylim(top=1, bottom=0)
            
        plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

        if savefig:
            plt.savefig(save_path.joinpath("pulse_average_by_" + overlay_filter + "_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)


def calculate_aucs(df, filterby=None, auc_frame_nums=list(), **kwargs):
    '''Calculates the area under the curve for each filter
    auc_frame_nums: indices of traces to retrieve AUCs from. first item should be the start index and the second item should be the stop index (included)'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        aucs = dict()

        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            x, traces = get_traces(subdf, **kwargs)

            if auc_frame_nums is not None:
                x = x[auc_frame_nums[0]:auc_frame_nums[1]+1]
                traces = [trace[auc_frame_nums[0]:auc_frame_nums[1]+1] for trace in traces]

            aucs[" - ".join([str(cond) for cond in conditions])] = [auc(x, trace) for trace in traces]

        aucs = dict([(k,pd.Series(v)) for k,v in aucs.items()])
        print(aucs)
        pd.DataFrame(aucs).to_clipboard()
        return aucs
    
    else:
        x, traces = get_traces(df, **kwargs)

        if auc_frame_nums is not None:
            x = x[auc_frame_nums[0]:auc_frame_nums[1]+1]
            traces = [trace[auc_frame_nums[0]:auc_frame_nums[1]+1] for trace in traces]

        return [auc(x, trace) for trace in traces]


def get_pulse_aucs_perfish(df, filterby=None, activated=-1, only_responsive=True, **kwargs):
    '''Returns the AUCs per pulse.
    If filterby is not None, returns the labels and pulse AUCs.
    activated: if -1, counts all responsive pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_auc_dfs = list()  # store the pulse AUC dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]

            if only_responsive:
                subdf = subdf[subdf['pulse_response'].notnull()]

            all_pulse_aucs = dict()

            for fish in subdf.fish_id.unique():
                aucs = calculate_aucs(subdf[subdf.fish_id == fish], only_responsive=only_responsive, **kwargs)

                if only_responsive:
                    if activated == -1:  # all pulses
                        pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                        pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
                    else:  # only activated (1) or suppressed (0) pulses
                        pr = subdf[subdf.fish_id == fish]['pulse_response']
                        # if fish == 36:
                        #     pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated and inj[0] != 4])
                        # else:
                        pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])
                else:
                    pfs = subdf[subdf.fish_id == fish]['pulse_frames']
                    pulses = np.array([i+1 for pf in pfs for i in range(len(pf))])

                for i, pulse in enumerate(pulses):
                    if pulse not in all_pulse_aucs:
                        all_pulse_aucs[pulse] = list()

                    all_pulse_aucs[pulse].append(aucs[i])

            all_pulse_aucs = dict(sorted(all_pulse_aucs.items()))
            all_pulse_aucs = dict([(k,pd.Series(v)) for k,v in all_pulse_aucs.items()])
            pulse_auc_dfs.append(pd.DataFrame(all_pulse_aucs))

            labels.append(" - ".join([str(cond) for cond in conditions]))

        return labels, pulse_auc_dfs
        
    else:
        notnull_df = df[df['pulse_response'].notnull()]

        all_pulse_aucs = dict()

        for fish in notnull_df.fish_id.unique():
            if activated == -1:  # all pulses
                pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
            else:  # only activated (1) or suppressed (0) pulses
                pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

            aucs = calculate_aucs(notnull_df[notnull_df.fish_id == fish], only_responsive=only_responsive, **kwargs)
		
            for i, pulse in enumerate(pulses):
                if pulse not in all_pulse_aucs:
                    all_pulse_aucs[pulse] = list()

                all_pulse_aucs[pulse].append(aucs[i])

        all_pulse_aucs = dict(sorted(all_pulse_aucs.items()))
        all_pulse_aucs = dict([(k,pd.Series(v)) for k,v in all_pulse_aucs.items()])

        return pd.DataFrame(all_pulse_aucs)


def plot_individual_pulses(df, filterby, pre_frame_num=15, post_frame_num=30, normalize=False, normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, fps=1, savefig=False, save_path=None):
    '''Separates each pulse and plots averages per pulse'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
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

        max_n_pulses = np.max([len(pf) for pf in df['pulse_frames']])

        traces = dict()

        fig, axs = plt.subplots(1, max_n_pulses, figsize=(max_n_pulses*10, 10))
        x = np.arange(0-pre_frame_num, 0+post_frame_num+1)

        for _, neuron in subdf.iterrows():
            pulses = neuron['pulse_frames']

            if only_responsive:
                responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
                pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, inhibited

                if normalize_by_first:
                    baseline = 0
                
                for i, pulse in enumerate(responsive_pulses):
                    if (pulse_activity[i] == 1 and neuron['activated'] == True) or (pulse_activity[i] == 0 and neuron['inhibited'] == True):
                        # if (neuron["fish_id"] == 36 and pulse != 4) or (neuron["fish_id"] != 36):
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

                        if pulse not in traces:
                            traces[pulse] = list()
                        
                        traces[pulse].append(trace)
                        # axs[pulse-1].plot(x/fps, trace, 'lightgray', alpha=0.5)

            else:
                if normalize_by_first:
                    baseline = 0

                for i, pulse in enumerate(pulses):
                    start_frame = pulse - pre_frame_num  # when the neuron traces will start
                    stop_frame = pulse + post_frame_num  # when the neuron traces will end

                    trace = neuron[key][start_frame:stop_frame+1]

                    if normalize:
                        baseline = np.median(neuron[key][start_frame:pulses[pulse-1]])
                        trace = (trace - baseline) / baseline
                    elif normalize_by_first and i == 0:
                        baseline = np.median(neuron[key][start_frame:pulses[pulse-1]])
                        trace = (trace - baseline) / baseline
                    elif normalize_by_first:
                        trace = (trace - baseline) / baseline

                    if i+1 not in traces:
                        traces[i+1] = list()

                    traces[i+1].append(trace)

                    # axs[i].plot(x/fps, trace, 'lightgray', alpha=0.5)

        for pulse in traces:
            avg_trace = np.array(traces[pulse]).mean(axis=0)
            sems = sem(np.array(traces[pulse]), axis=0)

            axs[pulse-1].plot (x/fps, avg_trace, zorder=102)
            axs[pulse-1].fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)
            axs[pulse-1].axvspan(-1/fps, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)

            if normalize or normalize_by_first:
                axs[pulse-1].set_ylim(top=1.5, bottom=-0.5)
            else:
                axs[pulse-1].set_ylim(top=1, bottom=0)

            axs[pulse-1].set_title(pulse, fontsize=36)
                    
        plt.suptitle(" - ".join([str(cond) for cond in conditions]), fontsize=42)

        if savefig and not only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_" + "_".join([str(cond) for cond in conditions]) + "_all.pdf"), transparent=True)
        elif savefig and only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)

        plt.show()
    

def plot_individual_pulses_overlayed(df, overlay_filter, filterby=None, color_order=None, pre_frame_num=15, post_frame_num=30, 
                                     normalize=False, normalize_by_first=False, key='raw_norm_temporal', 
                                     only_responsive=False, fps=1, savefig=False, save_path=None):
    '''Separates each pulse and plots averages from different filters overlayed on top'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
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

            max_n_pulses = np.max([len(pf) for pf in df['pulse_frames']])

            traces = dict()
            overlay_filters = dict()

            fig, axs = plt.subplots(1, max_n_pulses, figsize=(max_n_pulses*10, 10))
            x = np.arange(0-pre_frame_num, 0+post_frame_num+1)

            for _, neuron in subdf.iterrows():
                pulses = neuron['pulse_frames']

                if only_responsive:
                    responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
                    pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, inhibited

                    if normalize_by_first:
                        baseline = 0
                    
                    for i, pulse in enumerate(responsive_pulses):
                        if (pulse_activity[i] == 1 and neuron['activated'] == True) or (pulse_activity[i] == 0 and neuron['inhibited'] == True):
                            # if (neuron["fish_id"] == 36 and pulse != 4) or (neuron["fish_id"] != 36):
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

                            if pulse not in traces:
                                traces[pulse] = list()
                                overlay_filters[pulse] = list()
                            
                            traces[pulse].append(trace)
                            overlay_filters[pulse].append(neuron[overlay_filter])
                            # axs[pulse-1].plot(x, trace, 'lightgray', alpha=0.5)

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

                        if i+1 not in traces:
                            traces[i+1] = list()
                            overlay_filters[i+1] = list()
                        
                        traces[i+1].append(trace)
                        overlay_filters[i+1].append(neuron[overlay_filter])
                        # axs[i].plot(x, trace, 'lightgray', alpha=0.5)

            for pulse in traces:
                for i, of in enumerate(np.unique(overlay_filters[pulse])):
                    tr = np.array(traces[pulse])[np.where(np.array(overlay_filters[pulse]) == of)[0]]
                    avg_trace = np.array(tr).mean(axis=0)
                    # avg_trace = np.median(np.array(tr), axis=0)
                    sems = sem(np.array(tr), axis=0)
                    # lower, median, upper = np.quantile(np.array(tr), [0.025, 0.5, 0.975], axis=0)
                    
                    if color_order is not None:
                        axs[pulse-1].plot (x/fps, avg_trace, zorder=102, label=of, color=color_order[i])
                        # axs[pulse-1].plot (x/fps, median, zorder=102, label=of, color=color_order[i])
                        axs[pulse-1].fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                        # axs[pulse-1].fill_between(x/fps, lower, upper, alpha=0.2, zorder=101, color=color_order[i])
                    else:
                        axs[pulse-1].plot (x/fps, avg_trace, zorder=102, label=of)
                        # axs[pulse-1].plot (x/fps, median, zorder=102, label=of)
                        axs[pulse-1].fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)
                        # axs[pulse-1].fill_between(x/fps, lower, upper, alpha=0.2, zorder=101)
                
                axs[pulse-1].axvspan(-1/fps, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
                axs[pulse-1].legend()

                axs[pulse-1].set_ylim(top=5.5, bottom=-0.6)

                axs[pulse-1].set_title(pulse, fontsize=36)
                
            plt.suptitle(" - ".join([str(cond) for cond in conditions]), fontsize=42)

            if savefig and not only_responsive:
                plt.savefig(save_path.joinpath("individual_pulses_by_" + overlay_filter + "_" + "_".join([str(cond) for cond in conditions]) + "_all.pdf"), transparent=True)
            elif savefig and only_responsive:
                plt.savefig(save_path.joinpath("individual_pulses_by_" + overlay_filter + "_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)

    else:
        max_n_pulses = np.max([len(pf) for pf in df['pulse_frames']])

        traces = dict()
        overlay_filters = dict()

        fig, axs = plt.subplots(1, max_n_pulses, figsize=(max_n_pulses*10, 10))
        x = np.arange(0-pre_frame_num, 0+post_frame_num+1)

        for _, neuron in df.iterrows():
            pulses = neuron['pulse_frames']

            if only_responsive:
                responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
                pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, inhibited

                if normalize_by_first:
                    baseline = 0
                
                for i, pulse in enumerate(responsive_pulses):
                    if (pulse_activity[i] == 1 and neuron['activated'] == True) or (pulse_activity[i] == 0 and neuron['inhibited'] == True):
                        # if (neuron["fish_id"] == 36 and pulse != 4) or neuron["fish_id"] != 36:
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

                        if pulse not in traces:
                            traces[pulse] = list()
                            overlay_filters[pulse] = list()
                        
                        traces[pulse].append(trace)
                        overlay_filters[pulse].append(neuron[overlay_filter])
                        # axs[pulse-1].plot(x, trace, 'lightgray', alpha=0.5)

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

                    if i+1 not in traces:
                        traces[i+1] = list()
                        overlay_filters[i+1] = list()
                    
                    traces[i+1].append(trace)
                    overlay_filters[i+1].append(neuron[overlay_filter])
                    # axs[i].plot(x, trace, 'lightgray', alpha=0.5)

        for pulse in traces:
            for i, of in enumerate(np.unique(overlay_filters[pulse])):
                tr = np.array(traces[pulse])[np.where(np.array(overlay_filters[pulse]) == of)[0]]
                avg_trace = np.array(tr).mean(axis=0)
                sems = sem(np.array(tr), axis=0)
                
                if color_order is not None:
                    axs[pulse-1].plot (x/fps, avg_trace, zorder=102, label=of, color=color_order[i])
                    axs[pulse-1].fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=color_order[i])
                else:
                    axs[pulse-1].plot (x/fps, avg_trace, zorder=102, label=of)
                    axs[pulse-1].fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)
            
            axs[pulse-1].axvspan(-1/fps, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
            axs[pulse-1].legend()

            axs[pulse-1].set_ylim(top=3.25, bottom=-0.8)

            axs[pulse-1].set_title(pulse, fontsize=36)

        if savefig and not only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_by_" + overlay_filter + "_all.pdf"), transparent=True)
        elif savefig and only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_by_" + overlay_filter + ".pdf"), transparent=True)


def plot_pulses_overlayed(df, filterby, fps=1, pre_frame_num=15, post_frame_num=30, normalize=False, normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, savefig=False, save_path=None):
    '''Plots averages from each pulse overlayed on top'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
    for filter in filterby:
        if filter not in df:
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

        traces = dict()

        plt.figure(figsize=(10, 10))
        x = np.arange(0-pre_frame_num, 0+post_frame_num+1)

        for _, neuron in subdf.iterrows():
            pulses = neuron['pulse_frames']

            if only_responsive:
                responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
                pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, inhibited

                if normalize_by_first:
                    baseline = 0
                
                for i, pulse in enumerate(responsive_pulses):
                    if (pulse_activity[i] == 1 and neuron['activated'] == True) or (pulse_activity[i] == 0 and neuron['inhibited'] == True):
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

                        if pulse not in traces:
                            traces[pulse] = list()
                        
                        traces[pulse].append(trace)
                        # axs[pulse-1].plot(x, trace, 'lightgray', alpha=0.5)

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

                    if i+1 not in traces:
                        traces[i+1] = list()
                    
                    traces[i+1].append(trace)
                    # axs[i].plot(x, trace, 'lightgray', alpha=0.5)
        
        pulses = np.array(sorted(traces.keys()))
        colors = ["#cccccc", "#999999", "#666666", "#333333", "#000000"]  # egg water
        # colors = ["#FEC5E0", "#FD9AC8", "#F66EB1", "#97436A", "#321623"]  # glucose
        for i, pulse in enumerate(pulses):
            tr = np.array(traces[pulse])
            avg_trace = np.array(tr).mean(axis=0)
            sems = sem(np.array(tr), axis=0)

            plt.plot (x/fps, avg_trace, zorder=102, label=pulse, color=colors[i])
            plt.fill_between(x/fps, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101, color=colors[i])
        
        plt.axvspan(-1/fps, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
        plt.legend()

        # if normalize or normalize_by_first:
        #     plt.ylim(top=1, bottom=-0.50)
        # else:
        #     plt.ylim(top=1, bottom=0)
            
        plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

        if savefig and not only_responsive:
            plt.savefig(save_path.joinpath("overlayed_pulses_by_" + "_".join([str(cond) for cond in conditions]) + "_all.pdf"), transparent=True)
        elif savefig and only_responsive:
            plt.savefig(save_path.joinpath("overlayed_pulses_by_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)


def get_pulse_traces(df, normalize_by_first=False, normalize=False, pre_frame_num=15, post_frame_num=39, key="raw_norm_temporal"):
    '''Returns the temporal dynamics for each neuron for each pulse to plot pulse traces on Prism'''
    traces = dict()

    for _, neuron in df.iterrows():
        pulses = neuron['pulse_frames']
        responsive_pulses = [pr[0] for pr in neuron['pulse_response']]  # individual pulses that the neuron responded to
        pulse_activity = [pr[1] for pr in neuron['pulse_response']]  # if 1, activated, if 0, inhibited

        if normalize_by_first:
            baseline = 0
        
        for i, pulse in enumerate(responsive_pulses):
            if (pulse_activity[i] == 1 and neuron['activated'] == True) or (pulse_activity[i] == 0 and neuron['inhibited'] == True):
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

                if pulse not in traces:
                    traces[pulse] = list()
                
                traces[pulse].append(trace)

    traces = dict(sorted(traces.items()))
    traces = dict([(k,pd.Series(v)) for k,v in traces.items()])
    return pd.DataFrame(traces)


def get_pulse_response_ratios(df, filterby=None, normalize=False):
    '''Returns the pulse response ratios (e.g., responds to 3 out of 5 injections).
    If filterby is not None, returns the labels and pulse response ratios
    normalize: if True, normalizes the number of neurons to a per fish value'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_response_ratios = list()  # store the pulse response ratios
        counts = list()
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]

            pr = subdf[subdf['pulse_response'].notnull()]['pulse_response']
            lens = np.array([len(i) for i in pr])

            total_pulses = subdf[subdf['pulse_response'].notnull()]['pulse_frames']
            len_total_pulses = np.array([len(i) for i in total_pulses])
            
            x, y = np.unique(np.sort(lens/len_total_pulses*100), return_counts=True)

            if normalize:
                n_fish = len(subdf[subdf['pulse_response'].notnull()]['fish_id'].unique())
                y = y / n_fish
            
            labels.append(" - ".join([str(cond) for cond in conditions]))
            pulse_response_ratios.append(x)
            counts.append(y)
            
        return labels, pulse_response_ratios, counts
        
    else:
        pr = df[df['pulse_response'].notnull()]['pulse_response']
        lens = np.array([len(i) for i in pr])

        total_pulses = df[df['pulse_response'].notnull()]['pulse_frames']
        len_total_pulses = np.array([len(i) for i in total_pulses])

        pulse_response_ratios, counts = np.unique(np.sort(lens/len_total_pulses*100), return_counts=True)

        if normalize:
            n_fish = len(df[df['pulse_response'].notnull()]['fish_id'].unique())
            counts = counts / n_fish

        return pulse_response_ratios, counts


def get_pulse_response_ratios_perfish(df, filterby=None):
    '''Returns the pulse response ratios (e.g., responds to 3 out of 5 injections) for each fish.
    If filterby is not None, returns the labels and pulse response ratios'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_response_ratio_dfs = list()  # store the pulse response ratio dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            notnull_df = subdf[subdf['pulse_response'].notnull()]

            all_pulse_response_ratios = list()
            all_counts = list()

            for fish in notnull_df.fish_id.unique():
                pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                lens = np.array([len(i) for i in pr])

                total_pulses = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                len_total_pulses = np.array([len(i) for i in total_pulses])

                pulse_response_ratios, counts = np.unique(np.sort(lens/len_total_pulses*100), return_counts=True)

                all_pulse_response_ratios.append(np.array(pulse_response_ratios))
                all_counts.append(np.array(counts))

            flattened_prrs = np.array([j for i in all_pulse_response_ratios for j in i])
            xs = np.unique(flattened_prrs)
            prr_dict = {x:list() for x in xs}

            for fish, prr in enumerate(all_pulse_response_ratios):
                for x in xs:
                    try:
                        prr_dict[x].append(all_counts[fish][np.where(prr == x)][0])
                    except IndexError:
                        prr_dict[x].append(np.nan)

            labels.append(" - ".join([str(cond) for cond in conditions]))
            pulse_response_ratio_dfs.append(pd.DataFrame(prr_dict))

        return labels, pulse_response_ratio_dfs
        
    else:
        notnull_df = df[df['pulse_response'].notnull()]

        all_pulse_response_ratios = list()
        all_counts = list()

        for fish in notnull_df.fish_id.unique():
            pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
            lens = np.array([len(i) for i in pr])

            total_pulses = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
            len_total_pulses = np.array([len(i) for i in total_pulses])

            pulse_response_ratios, counts = np.unique(np.sort(lens/len_total_pulses*100), return_counts=True)

            all_pulse_response_ratios.append(np.array(pulse_response_ratios))
            all_counts.append(np.array(counts))

        flattened_prrs = np.array([j for i in all_pulse_response_ratios for j in i])
        xs = np.unique(flattened_prrs)
        prr_dict = {x:list() for x in xs}

        for fish, prr in enumerate(all_pulse_response_ratios):
            for x in xs:
                try:
                    prr_dict[x].append(all_counts[fish][np.where(prr == x)][0])
                except IndexError:
                    prr_dict[x].append(np.nan)

        return pd.DataFrame(prr_dict)


def plot_pulse_response_ratios(df, savefig=False, save_path=None, normalize=False, **kwargs):
    '''Plots the histogram of pulse response ratios'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
    try:  # if the data has a filterby
        labels, pulse_response_ratios, counts = get_pulse_response_ratios(df, normalize=normalize, **kwargs)
        for i, label in enumerate(labels):
            plt.plot(pulse_response_ratios[i], counts[i], label=label, marker='o')
            plt.xticks(pulse_response_ratios[i])
            plt.legend()
    
    except:
        pulse_response_ratios, counts = get_pulse_response_ratios(df, normalize=normalize, **kwargs)
        plt.plot(pulse_response_ratios, counts, marker='o')
        plt.xticks(pulse_response_ratios)
    
    plt.xlabel('pulse response ratio (%)')

    if normalize:
        plt.ylabel('# of neurons per fish')
    else:
        plt.ylabel('# of neurons')
    plt.show()

    if savefig:
        plt.savefig(save_path.joinpath("pulse_response_ratios.pdf"), transparent=True)


def get_pulse_response_counts_perfish(df, filterby=None, activated=-1):
    '''Returns the number of responsive neurons per pulse (e.g., n neurons responded to the 4th injection).
    If filterby is not None, returns the labels and pulse response ratios.
    activated: if -1, counts all responsive pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_response_count_dfs = list()  # store the pulse response count dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            notnull_df = subdf[subdf['pulse_response'].notnull()]

            all_pulse_response_counts = list()

            for fish in notnull_df.fish_id.unique():
                pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']

                if activated == -1:  # all responsive pulses
                    pulses = np.array([inj[0] for i in pr for inj in i])
                else:  # only activated (1) or suppressed (0) pulses
                    pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

                _, counts = np.unique(pulses, return_counts=True)

                all_pulse_response_counts.append(counts)

            max_pulses = np.max([len(fish) for fish in all_pulse_response_counts])
            xs = np.arange(1, max_pulses+1)

            prc_dict = {x:list() for x in xs}

            for fish, prc in enumerate(all_pulse_response_counts):
                for x in xs:
                    try:
                        prc_dict[x].append(prc[x-1])
                    except IndexError:
                        prc_dict[x].append(np.nan)

            labels.append(" - ".join([str(cond) for cond in conditions]))
            pulse_response_count_dfs.append(pd.DataFrame(prc_dict))

        return labels, pulse_response_count_dfs
        
    else:
        notnull_df = df[df['pulse_response'].notnull()]

        all_pulse_response_counts = list()

        for fish in notnull_df.fish_id.unique():
            pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']

            if activated == -1:  # all responsive pulses
                pulses = np.array([inj[0] for i in pr for inj in i])
            else:  # only activated (1) or suppressed (0) pulses
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

            _, counts = np.unique(pulses, return_counts=True)

            all_pulse_response_counts.append(counts)

        max_pulses = np.max([len(fish) for fish in all_pulse_response_counts])
        xs = np.arange(1, max_pulses+1)

        prc_dict = {x:list() for x in xs}

        for fish, prc in enumerate(all_pulse_response_counts):
            for x in xs:
                try:
                    prc_dict[x].append(prc[x-1])
                except IndexError:
                    prc_dict[x].append(np.nan)

        return pd.DataFrame(prc_dict)


def get_pulse_response_counts(df, filterby=None, activated=-1, normalize=False):
    '''Returns the number of responsive neurons per pulse (e.g., n neurons responded to the 4th injection).
    If filterby is not None, returns the labels and pulse response ratios
    activated: if -1, counts all responsive pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.
    normalize: if True, normalizes the number of neurons to a per fish value'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_response_counts = list()  # store the pulse response counts
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            pr = subdf[subdf['pulse_response'].notnull()]['pulse_response']

            if activated == -1:  # all responsive pulses
                pulses = np.array([inj[0] for i in pr for inj in i])
            else:  # only activated (1) or suppressed (0) pulses
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])
            
            _, counts = np.unique(pulses, return_counts=True)

            if normalize:
                n_fish = len(subdf[subdf['pulse_response'].notnull()]['fish_id'].unique())
                counts = counts / n_fish

            labels.append(" - ".join([str(cond) for cond in conditions]))
            pulse_response_counts.append(counts)

        return labels, pulse_response_counts
        
    else:
        pr = df[df['pulse_response'].notnull()]['pulse_response']
        if activated == -1:  # all responsive pulses
            pulses = np.array([inj[0] for i in pr for inj in i])
        else:  # only activated (1) or suppressed (0) pulses
            pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])
        
        _, counts = np.unique(pulses, return_counts=True)

        if normalize:
            n_fish = len(df[df['pulse_response'].notnull()]['fish_id'].unique())
            counts = counts / n_fish

        return counts


def plot_pulse_response_counts(df, savefig=False, save_path=None, normalize=False, **kwargs):
    '''Plots the histogram of responsive neurons per pulse'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
    try:  # if the data has a filterby
        labels, counts = get_pulse_response_counts(df, normalize=normalize, **kwargs)
        for i, label in enumerate(labels):
            x = np.arange(1, len(counts[i])+1)
            plt.plot(np.arange(1, len(counts[i])+1), counts[i], label=label, marker='o')
        plt.legend()
    
    except:
        counts = get_pulse_response_counts(df, normalize=normalize, **kwargs)
        x = np.arange(1, len(counts)+1)
        plt.plot(np.arange(1, len(counts)+1), counts, marker='o')
    
    plt.xticks(x)
    plt.xlabel('pulse')

    if normalize:
        plt.ylabel('# of responsive neurons per fish')
    else:
        plt.ylabel('# of responsive neurons')
    plt.show()

    if savefig:
        plt.savefig(save_path.joinpath("pulse_response_counts.pdf"), transparent=True)


def get_peaks(df, flip_suppressed=True, filterby=None, peak_frame_nums=list(), return_col=None, **kwargs):
    '''Returns a dataframe of peak values for each neuron
    flip_suppressed: for suppressed neurons, finds the minimum value instead of the maximum
    peak_frame_nums: indices of traces to calculate peaks from. first item should be the start index and the second item should be the stop index (included)'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        peaks = dict()

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
                x, traces, inhibited = get_traces(subdf, return_col='inhibited', **kwargs)
            elif return_col is not None:
                x, traces, other_col = get_traces(subdf, return_col=return_col, **kwargs)
                other_col_values[" - ".join([str(cond) for cond in conditions])] = other_col
            else:
                x, traces = get_traces(subdf, **kwargs)

            if len(peak_frame_nums) != 0:
                x = x[peak_frame_nums[0]:peak_frame_nums[1]+1]
                traces = [trace[peak_frame_nums[0]:peak_frame_nums[1]+1] for trace in traces]

            if flip_suppressed:
                # peaks[" - ".join([str(cond) for cond in conditions])] = [abs(trace[np.argmin(np.diff(trace))+1]) if inhibited[i] == True else trace[np.argmax(np.diff(trace))+1] for i, trace in enumerate(traces)]
                peaks[" - ".join([str(cond) for cond in conditions])] = [abs(np.min(trace)) if inhibited[i] == True else np.max(trace) for i, trace in enumerate(traces)]
            else:
                # peaks[" - ".join([str(cond) for cond in conditions])] = [trace[np.argmax(np.diff(trace))+1] for trace in traces]
                peaks[" - ".join([str(cond) for cond in conditions])] = [np.max(trace) for trace in traces]

    else:
        if flip_suppressed:
            x, traces, inhibited = get_traces(df, return_col='inhibited', **kwargs)
        elif return_col is not None:
            x, traces, other_col_values = get_traces(df, **kwargs)
        else:
            x, traces = get_traces(df, **kwargs)
        
        peaks = {
            "peaks": list()
        }

        if len(peak_frame_nums) != 0:
            x = x[peak_frame_nums[0]:peak_frame_nums[1]+1]
            traces = [trace[peak_frame_nums[0]:peak_frame_nums[1]+1] for trace in traces]

        if flip_suppressed:
            # peaks["peaks"] = [abs(trace[np.argmin(np.diff(trace))+1]) if inhibited[i] == True else trace[np.argmax(np.diff(trace))+1] for i, trace in enumerate(traces)]
            peaks["peaks"] = [abs(np.min(trace)) if inhibited[i] == True else np.max(trace) for i, trace in enumerate(traces)]
        else:
            # peaks["peaks"] = [trace[np.argmax(np.diff(trace))+1] for trace in traces]
            peaks["peaks"] = [np.max(trace) for trace in traces]

    peaks = dict([(k,pd.Series(v)) for k,v in peaks.items()])
    # print(peaks)
    pd.DataFrame(peaks).to_clipboard()
    if return_col is not None:
        return peaks, other_col_values
    else:
        return peaks


def get_pulse_peaks_perfish(df, filterby=None, activated=-1, only_responsive=True, flip_suppressed=True, **kwargs):
    '''Returns the peaks per pulse.
    If filterby is not None, returns the labels and pulse peaks.
    activated: if -1, counts all pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_peak_dfs = list()  # store the pulse peak dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            notnull_df = subdf[subdf['pulse_response'].notnull()]

            all_pulse_peaks = dict()

            for fish in notnull_df.fish_id.unique():
                if activated == -1:  # all pulses
                    pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                    pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
                else:  # only activated (1) or suppressed (0) pulses
                    pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                    # if fish == 36:
                    #     pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated and inj[0] != 4])
                    # else:
                    pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

                peaks = get_peaks(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
                
                # print(fish)
                # print(len(peaks["peaks"]))
                # print(len(pulses))
		
                for i, pulse in enumerate(pulses):
                    if pulse not in all_pulse_peaks:
                        all_pulse_peaks[pulse] = list()

                    all_pulse_peaks[pulse].append(peaks["peaks"][i])

            all_pulse_peaks = dict(sorted(all_pulse_peaks.items()))
            all_pulse_peaks = dict([(k,pd.Series(v)) for k,v in all_pulse_peaks.items()])
            pulse_peak_dfs.append(pd.DataFrame(all_pulse_peaks))

            labels.append(" - ".join([str(cond) for cond in conditions]))

        return labels, pulse_peak_dfs
        
    else:
        notnull_df = df[df["responsive"] == True]

        all_pulse_peaks = dict()

        for fish in notnull_df.fish_id.unique():
            if activated == -1:  # all responsive pulses
                pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
            else:  # only activated (1) or suppressed (0) pulses
                pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

            peaks = get_peaks(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
		
            for i, pulse in enumerate(pulses):
                if pulse not in all_pulse_peaks:
                    all_pulse_peaks[pulse] = list()

                all_pulse_peaks[pulse].append(peaks["peaks"][i])

        all_pulse_peaks = dict(sorted(all_pulse_peaks.items()))
        all_pulse_peaks = dict([(k,pd.Series(v)) for k,v in all_pulse_peaks.items()])

        return pd.DataFrame(all_pulse_peaks)


def plot_pulse_peaks(df, savefig=False, save_path=None, normalize=False, **kwargs):
    '''Plots the average peaks per pulse'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")
    
    try:  # if the data has a filterby
        labels, peaks = get_pulse_peaks_perfish(df, normalize=normalize, **kwargs)
        for i, label in enumerate(labels):
            x = np.arange(1, len(peaks[i].columns)+1)
            means = [np.mean(peaks[i][pulse]) for pulse in x]
            sems = [sem(peaks[i][pulse], nan_policy='omit') for pulse in x]
            # plt.plot(x, means, marker='o', label=label)
            plt.errorbar(x, means, yerr=sems, capsize=3, marker='o')
        print(sems)    

        plt.legend()
    
    except:
        peaks = get_pulse_peaks_perfish(df, normalize=normalize, **kwargs)
        x = np.arange(1, len(peaks.columns)+1)
        means = [np.mean(peaks[pulse]) for pulse in x]
        sems = [sem(peaks[pulse], nan_policy='omit') for pulse in x]
        # plt.plot(x, means, marker='o', label=label)
        plt.errorbar(x, means, yerr=sems, capsize=3, marker='o')
    
    plt.xticks(x)
    plt.xlabel('pulse')
    plt.ylabel('peak value (a.u.)')

    plt.show()

    if savefig:
        plt.savefig(save_path.joinpath("pulse_peaks.pdf"), transparent=True)


def get_times_to_peak(df, filterby=None, flip_suppressed=True, fps=1, peak_frame_nums=list(), **kwargs):
    '''Returns a dataframe of times to peak for each neuron
    flip_suppressed: for suppressed neurons, finds the minimum value instead of the maximum
    peak_frame_nums: indices of traces to calculate peaks from. first item should be the start index and the second item should be the stop index (included)'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        times_to_peak = dict()

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
                x, traces, inhibited = get_traces(subdf, return_col='inhibited', **kwargs)
            else:
                x, traces = get_traces(subdf, **kwargs)

            if peak_frame_nums is not None:
                x = x[peak_frame_nums[0]:peak_frame_nums[1]+1]
                traces = [trace[peak_frame_nums[0]:peak_frame_nums[1]+1] for trace in traces]

            if flip_suppressed:
                # times_to_peak[" - ".join([str(cond) for cond in conditions])] = [np.argmin(np.diff(trace))+1/fps if inhibited[i] == True else np.argmax(np.diff(trace))+1/fps for i, trace in enumerate(traces)]
                times_to_peak[" - ".join([str(cond) for cond in conditions])] = [np.argmin(trace)/fps if inhibited[i] == True else np.argmax(trace)/fps for i, trace in enumerate(traces)]
            else:
                times_to_peak[" - ".join([str(cond) for cond in conditions])] = [np.argmax(trace)/fps for trace in traces]
                
        times_to_peak = dict([(k,pd.Series(v)) for k,v in times_to_peak.items()])
        # print(times_to_peak)
        pd.DataFrame(times_to_peak).to_clipboard()
        return times_to_peak
    
    else:
        if flip_suppressed:
            x, traces, inhibited = get_traces(df, return_col='inhibited', **kwargs)
        else:
            x, traces = get_traces(df, **kwargs)

        if peak_frame_nums is not None:
            x = x[peak_frame_nums[0]:peak_frame_nums[1]+1]
            traces = [trace[peak_frame_nums[0]:peak_frame_nums[1]+1] for trace in traces]

        if flip_suppressed:
            times_to_peak = [np.argmin(trace)/fps if inhibited[i] == True else np.argmax(trace)/fps for i, trace in enumerate(traces)]
        else:
            times_to_peak = [np.argmax(trace)/fps for trace in traces]

        return times_to_peak
    

def get_pulse_peaktimes_perfish(df, filterby=None, activated=-1, flip_suppressed=True, only_responsive=True, **kwargs):
    '''Returns the times to peak per pulse.
    If filterby is not None, returns the labels and pulse peaks.
    activated: if -1, counts all responsive pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_peaktime_dfs = list()  # store the pulse time to peak dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            notnull_df = subdf[subdf['pulse_response'].notnull()]

            all_pulse_peaktimes = dict()

            for fish in notnull_df.fish_id.unique():
                if activated == -1:  # all responsive pulses
                    pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                    pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
                else:  # only activated (1) or suppressed (0) pulses
                    pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                    pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

                peaktimes = get_times_to_peak(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
		
                for i, pulse in enumerate(pulses):
                    if pulse not in all_pulse_peaktimes:
                        all_pulse_peaktimes[pulse] = list()

                    all_pulse_peaktimes[pulse].append(peaktimes[i])

            all_pulse_peaktimes = dict(sorted(all_pulse_peaktimes.items()))
            all_pulse_peaktimes = dict([(k,pd.Series(v)) for k,v in all_pulse_peaktimes.items()])
            pulse_peaktime_dfs.append(pd.DataFrame(all_pulse_peaktimes))

            labels.append(" - ".join([str(cond) for cond in conditions]))

        return labels, pulse_peaktime_dfs
        
    else:
        notnull_df = df[df["responsive"] == True]

        all_pulse_peaktimes = dict()

        for fish in notnull_df.fish_id.unique():
            if activated == -1:  # all responsive pulses
                pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
            else:  # only activated (1) or suppressed (0) pulses
                pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

            peaktimes = get_times_to_peak(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
		
            for i, pulse in enumerate(pulses):
                if pulse not in all_pulse_peaktimes:
                    all_pulse_peaktimes[pulse] = list()

                all_pulse_peaktimes[pulse].append(peaktimes["peaks"][i])

        all_pulse_peaktimes = dict(sorted(all_pulse_peaktimes.items()))
        all_pulse_peaktimes = dict([(k,pd.Series(v)) for k,v in all_pulse_peaktimes.items()])

        return pd.DataFrame(all_pulse_peaktimes)


def calculate_time_at_half_maximum(x, trace, suppressed=False):
    """
    Returns the time at half maximum of the given trace.

    Parameters:
    x (array-like): The x values (e.g., time points).
    trace (array-like): The y values (e.g., intensity or response values).

    Returns:
    float: The interpolated time at half maximum.
    """
    # Find the index of the peak
    if not suppressed:
        peak_idx = np.argmax(trace)
    else:
        peak_idx = np.argmin(trace)
    
    # Get the maximum value and half maximum value
    max_value = trace[peak_idx]
    half_max = max_value / 2
    
    # Consider only the segment before the peak
    pre_peak_trace = trace[:peak_idx + 1]
    pre_peak_x = x[:peak_idx + 1]

    # Find where the trace crosses the half maximum before the peak
    if not suppressed:
        above_half_max = pre_peak_trace >= half_max
    else:
        above_half_max = pre_peak_trace <= half_max

    crossing_indices = np.where(np.diff(above_half_max.astype(int)) != 0)[0]

    if len(crossing_indices) == 0:
        # print(trace)
        # print(suppressed)
        return 0

    # Interpolate to find a more accurate time at half maximum
    x1, x2 = pre_peak_x[crossing_indices[0]], pre_peak_x[crossing_indices[0] + 1]
    y1, y2 = pre_peak_trace[crossing_indices[0]], pre_peak_trace[crossing_indices[0] + 1]

    interpolator = interp1d([y1, y2], [x1, x2], kind='linear')
    time_at_half_max = interpolator(half_max)

    return float(time_at_half_max)


def get_times_at_half_maximum(df, filterby=None, flip_suppressed=True, fps=1, peak_frame_nums=list(), **kwargs):
    '''Returns a dataframe of times at half maximum (THM) for each neuron
    flip_suppressed: for suppressed neurons, finds the minimum value instead of the maximum
    peak_frame_nums: indices of traces to calculate peaks from. first item should be the start index and the second item should be the stop index (included)'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        times_at_thm = dict()

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
                x, traces, inhibited = get_traces(subdf, return_col='inhibited', **kwargs)
            else:
                x, traces = get_traces(subdf, **kwargs)

            if peak_frame_nums is not None:
                x = x[peak_frame_nums[0]:peak_frame_nums[1]+1]
                traces = [trace[peak_frame_nums[0]:peak_frame_nums[1]+1] for trace in traces]

            if flip_suppressed:
                times_at_thm[" - ".join([str(cond) for cond in conditions])] = [calculate_time_at_half_maximum(x, trace, suppressed=True)/fps if inhibited[i] == True else calculate_time_at_half_maximum(x, trace, suppressed=False)/fps for i, trace in enumerate(traces)]
            else:
                times_at_thm[" - ".join([str(cond) for cond in conditions])] = [calculate_time_at_half_maximum(x, trace, suppressed=False)/fps for trace in traces]
                
        times_at_thm = dict([(k,pd.Series(v)) for k,v in times_at_thm.items()])
        # print(times_at_thm)
        pd.DataFrame(times_at_thm).to_clipboard()
        return times_at_thm
    
    else:
        if flip_suppressed:
            x, traces, inhibited = get_traces(df, return_col='inhibited', **kwargs)
        else:
            x, traces = get_traces(df, **kwargs)

        if peak_frame_nums is not None:
            x = x[peak_frame_nums[0]:peak_frame_nums[1]+1]
            traces = [trace[peak_frame_nums[0]:peak_frame_nums[1]+1] for trace in traces]

        if flip_suppressed:
            times_at_thm = [calculate_time_at_half_maximum(x, trace, suppressed=True)/fps if inhibited[i] == True else calculate_time_at_half_maximum(x, trace, suppressed=False)/fps for i, trace in enumerate(traces)]
        else:
            times_at_thm = [calculate_time_at_half_maximum(x, trace, suppressed=False)/fps for trace in traces]

        return times_at_thm


def get_pulse_thms_perfish(df, filterby=None, activated=-1, flip_suppressed=True, only_responsive=True, **kwargs):
    '''Returns the times at half maximum per pulse.
    If filterby is not None, returns the labels and pulse peaks.
    activated: if -1, counts all responsive pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_thm_dfs = list()  # store the pulse time to peak dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            notnull_df = subdf[subdf['pulse_response'].notnull()]

            all_pulse_thms = dict()

            for fish in notnull_df.fish_id.unique():
                if activated == -1:  # all responsive pulses
                    pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                    pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
                else:  # only activated (1) or suppressed (0) pulses
                    pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                    pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

                thms = get_times_at_half_maximum(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
		
                for i, pulse in enumerate(pulses):
                    if pulse not in all_pulse_thms:
                        all_pulse_thms[pulse] = list()

                    all_pulse_thms[pulse].append(thms[i])

            all_pulse_thms = dict(sorted(all_pulse_thms.items()))
            all_pulse_thms = dict([(k,pd.Series(v)) for k,v in all_pulse_thms.items()])
            pulse_thm_dfs.append(pd.DataFrame(all_pulse_thms))

            labels.append(" - ".join([str(cond) for cond in conditions]))

        return labels, pulse_thm_dfs
        
    else:
        notnull_df = df[df["responsive"] == True]

        all_pulse_thms = dict()

        for fish in notnull_df.fish_id.unique():
            if activated == -1:  # all responsive pulses
                pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
            else:  # only activated (1) or suppressed (0) pulses
                pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

            thms = get_times_at_half_maximum(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
		
            for i, pulse in enumerate(pulses):
                if pulse not in all_pulse_thms:
                    all_pulse_thms[pulse] = list()

                all_pulse_thms[pulse].append(thms["peaks"][i])

        all_pulse_thms = dict(sorted(all_pulse_thms.items()))
        all_pulse_thms = dict([(k,pd.Series(v)) for k,v in all_pulse_thms.items()])

        return pd.DataFrame(all_pulse_thms)


def get_times_to_decay(df, filterby=None, fps=1, non_sig_cutoff=3, pre_frame_num=15, peak_frame_nums=list(), flip_suppressed=True, **kwargs):
    '''Returns a dataframe of times to decay for each neuron
    non_sig_cutoff: if this number of frames are non-significant consecutively, the response has decayed
    decay_frame_nums: indices of traces to calculate decays from. first item should be the start index and the second item should be the stop index (included)'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        times_to_decay = dict()
        times_to_peak = get_times_to_peak(df, filterby, fps=fps, 
                                        peak_frame_nums=peak_frame_nums, 
                                        pre_frame_num=pre_frame_num, 
                                        flip_suppressed=flip_suppressed, 
                                        **kwargs)

        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            x, traces = get_traces(subdf, pre_frame_num=pre_frame_num, **kwargs)
            times_to_decay[" - ".join([str(cond) for cond in conditions])] = list()
            t_peaks = times_to_peak[" - ".join([str(cond) for cond in conditions])]

            for i, trace in enumerate(traces):
                baseline_distribution = baseline_distribution = trace[:pre_frame_num]
                peak_frame = round(t_peaks[i] * fps)  # when the peak happened, with peak_frame_nums[0] as the start
                start_frame = peak_frame_nums[0] + peak_frame  # the frame where the decay search will start from
                
                final_sig_frame = -1  # keep track of the last significant frame

                added = False  # check if the neuron is added to the dictionary
                                # this is to make sure that the neurons that still haven't decayed are added to the list 
                for t in range(start_frame+1, len(trace)):
                    # The range starts from start_frame+1 because the first frame is the peak frame
                    arr = trace[t]
                    frame = t - start_frame

                    _, p_value = wilcoxon(baseline_distribution-arr) 

                    # if np.mean(arr) > np.mean(baseline_distribution):
                    #     final_sig_frame = frame
                    #     added = True
                    #     times_to_decay[" - ".join([str(cond) for cond in conditions])].append((final_sig_frame/fps))
                    #     break

                    if p_value < 0.05:
                        final_sig_frame = frame

                    elif final_sig_frame != -1 and frame-final_sig_frame == non_sig_cutoff:
                        added = True
                        times_to_decay[" - ".join([str(cond) for cond in conditions])].append((final_sig_frame/fps))
                        break

                if not added and final_sig_frame != -1:
                    times_to_decay[" - ".join([str(cond) for cond in conditions])].append((final_sig_frame/fps))
                elif not added and final_sig_frame == -1:
                    times_to_decay[" - ".join([str(cond) for cond in conditions])].append(1/fps)

        times_to_decay = dict([(k,pd.Series(v)) for k,v in times_to_decay.items()])
        print(times_to_decay) 
        pd.DataFrame(times_to_decay).to_clipboard()
        return times_to_decay
    else:
        x, traces = get_traces(df, pre_frame_num=pre_frame_num, **kwargs)
        t_peaks = get_times_to_peak(df, fps=fps, 
                                        peak_frame_nums=peak_frame_nums, 
                                        pre_frame_num=pre_frame_num, 
                                        flip_suppressed=flip_suppressed, 
                                        **kwargs)
        times_to_decay = list()

        for i, trace in enumerate(traces):
            baseline_distribution = baseline_distribution = trace[:pre_frame_num]
            peak_frame = round(t_peaks[i] * fps)  # when the peak happened, with peak_frame_nums[0] as the start
            start_frame = peak_frame_nums[0] + peak_frame  # the frame where the decay search will start from
            
            final_sig_frame = -1  # keep track of the last significant frame

            added = False  # check if the neuron is added to the dictionary
                            # this is to make sure that the neurons that still haven't decayed are added to the list 
            for t in range(start_frame+1, len(trace)):
                arr = trace[t]
                frame = t - start_frame

                _, p_value = wilcoxon(baseline_distribution-arr) 

                if p_value < 0.05:
                    final_sig_frame = frame

                elif final_sig_frame != -1 and frame-final_sig_frame == non_sig_cutoff:
                    added = True
                    times_to_decay.append((final_sig_frame/fps))
                    break

            if not added and final_sig_frame != -1:
                times_to_decay.append((final_sig_frame/fps))
            elif not added and final_sig_frame == -1:
                times_to_decay.append(1/fps)

        return times_to_decay


def get_pulse_decays_perfish(df, filterby=None, activated=-1, flip_suppressed=True, only_responsive=True, **kwargs):
    '''Returns the times to decay per pulse.
    If filterby is not None, returns the labels and pulse decay times.
    activated: if -1, counts all responsive pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_decay_dfs = list()  # store the pulse peak dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]
            notnull_df = subdf[subdf['pulse_response'].notnull()]

            all_pulse_decays = dict()

            for fish in notnull_df.fish_id.unique():
                if activated == -1:  # all pulses
                    pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                    pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
                else:  # only activated (1) or suppressed (0) pulses
                    pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                    # if fish == 36:
                    #     pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated and inj[0] != 4])
                    # else:
                    pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

                decays = get_times_to_decay(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
		
                for i, pulse in enumerate(pulses):
                    if pulse not in all_pulse_decays:
                        all_pulse_decays[pulse] = list()

                    all_pulse_decays[pulse].append(decays[i])
                    # if (decays[i] > 90) and (pulse == 5):
                    #     print(fish)
                    #     print(pulses)
                    #     print(i)

            all_pulse_decays = dict(sorted(all_pulse_decays.items()))
            all_pulse_decays = dict([(k,pd.Series(v)) for k,v in all_pulse_decays.items()])
            pulse_decay_dfs.append(pd.DataFrame(all_pulse_decays))

            labels.append(" - ".join([str(cond) for cond in conditions]))

        return labels, pulse_decay_dfs
        
    else:
        notnull_df = df[df["responsive"] == True]

        all_pulse_decays = dict()

        for fish in notnull_df.fish_id.unique():
            if activated == -1:  # all pulses
                pf = notnull_df[notnull_df.fish_id == fish]['pulse_frames']
                pulses = np.array([np.arange(len(i)) for i in pf]).flatten()
            else:  # only activated (1) or suppressed (0) pulses
                pr = notnull_df[notnull_df.fish_id == fish]['pulse_response']
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])

            decays = get_times_to_decay(notnull_df[notnull_df.fish_id == fish], flip_suppressed=flip_suppressed, only_responsive=only_responsive,
                                  **kwargs)
		
            for i, pulse in enumerate(pulses):
                if pulse not in all_pulse_decays:
                    all_pulse_decays[pulse] = list()

                all_pulse_decays[pulse].append(decays[i])

        all_pulse_decays = dict(sorted(all_pulse_decays.items()))
        all_pulse_decays = dict([(k,pd.Series(v)) for k,v in all_pulse_decays.items()])

        return pd.DataFrame(all_pulse_decays)


def label_monotonic_neurons(df, flip_suppressed=True, pvalue_threshold=0.05, corr_threshold=0, save_df=Path(), **kwargs):
    '''Labels neurons in the df that have a monotonic relationship with the pulse (i.e., integrating vs habituating neurons)
    min_pulses: the number of pulse responses that the neuron needs to have
    pvalue_threshold: for a neuron to be considered integrating/habituating, the spearman p-value needs to be lower than this
    corr_threshold: for a neuron to be considered integrating/habituating, the correlation value needs to be higher than this'''
    new_df = df[df['responsive'] == True].copy()
    peaks = get_peaks(new_df, flip_suppressed=flip_suppressed, **kwargs)

    df['monotonic'] = None

    count = 0  # count at which peak index you're at
    for i, neuron in df.iterrows():
        if neuron["responsive"] == True:
            n_peaks = len(neuron['pulse_response'])
            pulses = [pulse[0] for pulse in neuron['pulse_response']]
            
            res = spearmanr(peaks["peaks"][count:count+n_peaks], pulses)

            if res.pvalue < pvalue_threshold and res.statistic > corr_threshold:
                if neuron["activated"]:
                    df.loc[i, "monotonic"] = 'integrating'
                elif neuron["inhibited"]:
                    df.loc[i, "monotonic"] = "habituating"

            elif res.pvalue < pvalue_threshold and res.statistic < -corr_threshold:
                if neuron["activated"]:
                    df.loc[i, "monotonic"] = 'habituating'
                elif neuron["inhibited"]:
                    df.loc[i, "monotonic"] = 'integrating'
            
            else:
                df.loc[i, "monotonic"] = 'non-corr'

            # diff = np.diff(peaks["peaks"][count:count+n_peaks])

            # if np.all(diff > 0):
            #     new_df.loc[i, "monotonic"] = 'integrating'
            # elif np.all(diff < 0):
            #     new_df.loc[i, "monotonic"] = 'habituating'
            # else:
            #     new_df.loc[i, "monotonic"] = 'non-corr'

            count += n_peaks

    if len(str(save_df)) != 0:
        df.to_hdf(save_df.joinpath('monotonic_temporal.h5'), key='monotonic', mode='w')

        return df
    

def label_transient_sustained_neurons(df, fps=1, transient_thresh=49, sustained_thresh=153, flip_suppressed=True, save_df=Path(), **kwargs):
	'''Labels neurons in the df that have short or long decays (i.e., transient vs sustained neurons)
		transient_thresh: longest decay time for a neuron to be considered transient (in frames) (threshold included)
		sustained_thresh: shortest decay time for a neuron to be considered sustained (in frames) (threshold included)'''
	new_df = df[df['responsive'] == True].copy()
	times_to_decay = get_times_to_decay(new_df, flip_suppressed=flip_suppressed, **kwargs)

	df['decay_type'] = None

	count = 0  # count at which peak index you're at
	for i, neuron in df.iterrows():
		if neuron["responsive"] == True:
			n_peaks = len(neuron['pulse_response'])
			avg_decay = np.mean(times_to_decay[count:count+n_peaks]) * fps  # converted to frames

			if avg_decay <= transient_thresh:
				df.loc[i, "decay_type"] = "transient"
			elif avg_decay >= sustained_thresh:
				df.loc[i, "decay_type"] = "sustained"
			else:
				df.loc[i, "decay_type"] = "middle"

			count += n_peaks

	if len(str(save_df)) != 0:
		df.to_hdf(save_df.joinpath('monotonic_temporal.h5'), key='monotonic', mode='w')

		return df


def plot_traces(df, filterby=None, fps=1, key='raw_norm_temporal'):
    '''Plots the temporal trace of each neuron in the df'''
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

            traces = list()
            pulse_frames = list()

            for i, sub_row in subdf.iterrows():
                traces.append(sub_row[key])
                pulse_frames.append(sub_row['pulse_frames'])

            fig, axes = plt.subplots(len(traces), 1, sharex=True, sharey=True, figsize=(20, 3*len(traces)))
            for i, t in enumerate(traces):
                try:
                    axes[i].plot(t)
                    if i == 0:
                        axes[i].set_title(" - ".join([str(cond) for cond in conditions]))
                        
                    for pulse in pulse_frames[i]:
                        axes[i].vlines(pulse, np.min(t), np.max(t), color='r')

                except TypeError:  # if there aren't more than 1 fish
                    axes.plot(t)
                    axes.set_title(" - ".join([str(cond) for cond in conditions]))
                        
                    for pulse in pulse_frames[i]:
                        axes.vlines(pulse, np.min(t), np.max(t), color='r')

            ticks = np.arange(0, 16*60*fps, 60*fps)
            plt.xticks(ticks=ticks, labels=np.round(ticks/fps).astype(int))
            plt.xlabel('Time (s)')
            plt.show()

    else:
        traces = list()
        pulse_frames = list()

        for i, row in df.iterrows():
            traces.append(row[key])
            pulse_frames.append(row['pulse_frames'])

        fig, axes = plt.subplots(len(traces), 1, sharex=True, sharey=True, figsize=(20, 3*len(traces)))
        for i, t in enumerate(traces):
            try:
                axes[i].plot(t)
                for pulse in pulse_frames[i]:
                    axes[i].vlines(pulse, 0, 1, color='r')

            except TypeError:  # if there aren't more than 1 fish
                axes.plot(t)
                for pulse in pulse_frames[i]:
                    axes.vlines(pulse, 0, 1, color='r')

        ticks = np.arange(0, 16*60*fps, 60*fps)
        plt.xticks(ticks=ticks, labels=np.round(ticks/fps).astype(int))
        plt.xlabel('Time (s)')
        plt.show()


def get_n_responsive_neurons(df, filterby):
	'''Returns a dataframe of number of responsive neurons per fish'''
	for filter in filterby:
		if filter not in df.columns:
			raise ValueError("Given filter is not a column in the df")
		
	filter_groups = df.groupby(filterby).size().reset_index()

	n_resp_neurons = dict()

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
		n_resp_neurons[label] = list()

		for fish in subdf.fish_id.unique():
			fish_df = subdf[subdf.fish_id == fish]
			n_resp_neurons[label].append(len(fish_df[fish_df.responsive == True]))

	n_resp_neurons = dict([(k,pd.Series(v)) for k,v in n_resp_neurons.items()])
	print(pd.DataFrame(n_resp_neurons))
	pd.DataFrame(n_resp_neurons).to_clipboard()


def get_percentage_metric_neurons(df, metric, value, filterby=None, inverse=False):
    '''Returns a dataframe of %metric neurons per fish
    metric: must be a column in df, not in filterby
    value: the value that the rows need to be in the metric column
    inverse: if True, it will return when the metric is not equal to value'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
            
        filter_groups = df.groupby(filterby).size().reset_index()

        perc_metric_neurons = dict()

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

        perc_metric_neurons = dict([(k,pd.Series(v)) for k,v in perc_metric_neurons.items()])
        print(pd.DataFrame(perc_metric_neurons))
        pd.DataFrame(perc_metric_neurons).to_clipboard()
        return pd.DataFrame(perc_metric_neurons)


def get_pulse_percentage_responsive_perfish(df, filterby=None, activated=-1):
    '''Returns the % of responsive neurons per pulse.
    If filterby is not None, returns the labels and pulse percentages.
    activated: if -1, counts all responsive pulses. if 0, counts only suppressed pulses. if 1, counts only activated pulses.'''
    if filterby is not None:
        for filter in filterby:
            if filter not in df.columns:
                raise ValueError("Given filter is not a column in the df")
        filter_groups = df.groupby(filterby).size().reset_index()

        labels = list()  # store the name of the conditions
        pulse_percentage_dfs = list()  # store the pulse AUC dataframes
            
        for _, row in filter_groups.iterrows():
            conditions = [row[col] for col in filterby]

            filters = list()
            for col, cond in zip(filterby, conditions):
                if isinstance(cond, str):
                    filters.append(f"(df['{col}'] == '{cond}')")
                else:
                    filters.append(f"(df['{col}'] == {cond})")

            subdf = df[eval(" & ".join(filters))]

            all_pulse_percentages = dict()

            for fish in subdf.fish_id.unique():
                fish_df = df[df.fish_id == fish].reset_index()
                pr = subdf[subdf.fish_id == fish]['pulse_response']
                n_pulses = len(fish_df.loc[0, "pulse_frames"])

                if activated == -1:  # all responsive pulses
                    pulses = np.array([inj[0] for i in pr for inj in i])
                else:  # only activated (1) or suppressed (0) pulses
                    pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])
                    
                for pulse in np.arange(1, n_pulses+1):
                    if pulse not in all_pulse_percentages.keys():
                        all_pulse_percentages[pulse] = list()

                    n_responsive_neurons = list(pulses).count(pulse)
                    all_pulse_percentages[pulse].append(n_responsive_neurons / len(fish_df) * 100)

            all_pulse_percentages = dict(sorted(all_pulse_percentages.items()))
            all_pulse_percentages = dict([(k,pd.Series(v)) for k,v in all_pulse_percentages.items()])
            pulse_percentage_dfs.append(pd.DataFrame(all_pulse_percentages))

            labels.append(" - ".join([str(cond) for cond in conditions]))

        return labels, pulse_percentage_dfs

    else:
        all_pulse_percentages = dict()

        for fish in df.fish_id.unique():
            fish_df = df[df.fish_id == fish].reset_index()
            pr = df[df.fish_id == fish]['pulse_response']
            n_pulses = len(df.loc[0, "pulse_frames"])

            if activated == -1:  # all responsive pulses
                pulses = np.array([inj[0] for i in pr for inj in i])
            else:  # only activated (1) or suppressed (0) pulses
                pulses = np.array([inj[0] for i in pr for inj in i if inj[1] == activated])
                
            for pulse in np.arange(1, n_pulses+1):
                if pulse not in all_pulse_percentages.keys():
                    all_pulse_percentages[pulse] = list()

                n_responsive_neurons = list(pulses).count(pulse)
                all_pulse_percentages[pulse].append(n_responsive_neurons / len(fish_df) * 100)

        all_pulse_percentages = dict(sorted(all_pulse_percentages.items()))
        all_pulse_percentages = dict([(k,pd.Series(v)) for k,v in all_pulse_percentages.items()])

        return all_pulse_percentages
    

def plot_grouped_heatmaps(df, filterby, sort=True, key='norm_temporal', savefig=False, save_path=None, colors=None, **kwargs):
    '''Plots heatmaps based on filtered columns.
    filterby: list of column names that the temporal_df will be filtered by
    sort: if True, sorts by peak'''
    if savefig and save_path is None:
        raise ValueError("Enter a save_path to save the figure")

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

        if savefig:
            plt.savefig(save_path.joinpath("heatmap_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)
    

def plot_time_to_peak_distribution(df, step=1, convert_to_frames=False, fps=1, pre_frame_num=15, post_frame_num=39, flip_suppressed=True, save_df=Path(), **kwargs):
	'''Plots the distribution of peak times
	step: bin steps'''
	times_to_peak = get_times_to_peak(df, fps=fps, pre_frame_num=pre_frame_num, post_frame_num=post_frame_num, flip_suppressed=flip_suppressed, **kwargs)

	if convert_to_frames:
		times_to_peak = np.array(times_to_peak) * fps
	
	n, bins, _ = plt.hist(times_to_peak, bins=np.arange(np.array(times_to_peak).min(), np.array(times_to_peak).max(), step), density=False, align='mid')
	
	if convert_to_frames:
		plt.xlabel("Peak frames relative to the injection")
	else:
		plt.xlabel('Peak times relative to the injection (s)')
		
	plt.ylabel('# of neurons')
	if convert_to_frames:
		plt.axvline(5, color="tab:red")
	else:
		plt.axvline(4, color="tab:red")
	plt.show()
	
	return n, bins


def plot_time_to_decay_distribution(df, step=1, convert_to_frames=False, fps=1, pre_frame_num=15, post_frame_num=39, save_df=Path(), **kwargs):
    '''Plots the distribution of peak times
    step: bin steps'''
    times_to_decay = get_times_to_decay(df, fps=fps, pre_frame_num=pre_frame_num, post_frame_num=post_frame_num, **kwargs)

    if convert_to_frames:
        times_to_decay = np.array(times_to_decay) * fps

    fig, ax = plt.subplots()
    n, bins, _ = plt.hist(times_to_decay, bins=np.arange(np.array(times_to_decay).min(), np.array(times_to_decay).max(), step), density=False, align='mid', color="black")

    if convert_to_frames:
        plt.xlabel("Decay frames relative to the injection")
    else:
        plt.xlabel('Decay times relative to the injection (s)')
        
    plt.ylabel('# of neurons')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

    return n, bins