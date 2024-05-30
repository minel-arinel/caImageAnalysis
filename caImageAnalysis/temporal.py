from kneed import KneeLocator
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import colormaps
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
import plotly.express as px
import random
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, median_abs_deviation, sem, t, zscore
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
    # raw_dff = list()
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

                # row.cnmf.run_detrend_dfof()  # uses caiman's detrend_df_f function
                # F_dff = row.cnmf.get_detrend_dfof("good")  # detrended dF/F0 curves
                # raw_dff.append(F_dff[indices])

                fts = pd.read_hdf(fish.data_paths['volumes'][plane]['frametimes'])
                try:
                    pulses = [fts[fts.pulse == pulse].index.values[0] for pulse in fts.pulse.unique() if pulse != fts.loc[0, 'pulse']]
                except:
                    pulses = [0]
                
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


def normalize_temporaldf(fish):
    '''Normalizes both the raw and denoised traces between 0 and 1'''
    fish.temporal_df['norm_temporal'] = None
    fish.temporal_df['raw_norm_temporal'] = None

    for i, row in fish.temporal_df.iterrows():
        norm_temporals = list()
        raw_norm_temporals = list()

        for comp in row.temporal:
            norm_temporal = (comp - min(comp)) / (max(comp) - min(comp))
            norm_temporals.append(norm_temporal)

        for comp in row.raw_temporal:
            raw_norm_temporal = (comp - min(comp)) / (max(comp) - min(comp))
            raw_norm_temporals.append(raw_norm_temporal)

        fish.temporal_df['norm_temporal'][i] = norm_temporals
        fish.temporal_df['raw_norm_temporal'][i] = raw_norm_temporals

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
        plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

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
        plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

        if savefig:
            plt.savefig(save_path.joinpath("tsne_clustering.pdf"), transparent=True)


def pca_clustering(df, filterby=None, colorby=None, key='raw_norm_temporal'):
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
					
			pca = PCA(n_components=3)
			components = pca.fit_transform(traces)

			total_var = pca.explained_variance_ratio_.sum() * 100
			
			if colorby is not None:
				fig = px.scatter_3d(
					components, x=0, y=1, z=2, color=subdf[colorby],
					title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
					labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
				)
			else:
				fig = px.scatter_3d(
					components, x=0, y=1, z=2,
					title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
					labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
				)

			fig.show()

	else:
		traces = np.array(df.loc[:, key])
		traces = np.array([np.array(trace) for trace in traces])
				
		pca = PCA(n_components=3)
		components = pca.fit_transform(traces)

		total_var = pca.explained_variance_ratio_.sum() * 100
		
		if colorby is not None:
			fig = px.scatter_3d(
				components, x=0, y=1, z=2, color=df[colorby],
				title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
				labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
			)
		else:
			fig = px.scatter_3d(
				components, x=0, y=1, z=2,
				title=" - ".join([str(cond) for cond in conditions]) + f' - Total Explained Variance: {total_var:.2f}%',
				labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
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

    if len(save_path) != 0:
       plt.savefig(save_path.joinpath(f"hierarchical_clustering_{max_inter_cluster_dist}.pdf"), transparent=True) 

    # Create a clusters dictionary to store all temporal responses per cluster
    clusters = dict()
    for i, cluster in enumerate(T[sort_inds]):
        if cluster in clusters:
            clusters[cluster].append(data[sort_inds][i])
        else:
            clusters[cluster] = [data[sort_inds][i]]

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


def plot_heatmap(data, fps=1.3039181000348583, pulses=[391, 548, 704, 861, 1017]):
    '''Plots temporal components'''
    # Just simple heatmaps
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(data, cmap='plasma', interpolation='nearest', aspect='auto')
    
    ticks = np.arange(0, 16*60*fps, 60*fps)
    plt.xticks(ticks=ticks, labels=np.round(ticks/fps).astype(int))
    
    for pulse in pulses:
        plt.vlines(pulse, -0.5, len(data)-0.5, color='w', lw=3)
    
    plt.xlabel('Time (s)')
    plt.grid(visible=False)


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


def unroll_temporal_df(fish, **kwargs):
    '''Expands the temporal_df so that each row is a single neuron rather than a single plane'''
    
    unrolled_df = pd.DataFrame(columns=['fish_id', 'plane', 'neuron', 'raw_temporal', 'temporal', 'raw_norm_temporal', 'norm_temporal',
                                        'raw_dff', 'dff', 'raw_norm_dff', 'norm_dff', 'roi_index', 'pulse_frames'])
    
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
            unrolled_row['pulse_frames'] = row['pulse_frames']

            unrolled_df = pd.concat([unrolled_df, pd.DataFrame([unrolled_row])], ignore_index=True)

    stim_responsive, activated, inhibited, pulse_responses = find_stimulus_responsive(fish, **kwargs)
    
    unrolled_df['responsive'] = False
    unrolled_df['activated'] = None
    unrolled_df['inhibited'] = None
    unrolled_df['pulse_response'] = None

    for i, neuron in enumerate(stim_responsive):
        unrolled_df.loc[neuron, 'responsive'] = True

        if neuron in activated:
            unrolled_df.loc[neuron, 'activated'] = True
            unrolled_df.loc[neuron, 'inhibited'] = False
        
        elif neuron in inhibited:
            unrolled_df.loc[neuron, 'inhibited'] = True
            unrolled_df.loc[neuron, 'activated'] = False

        unrolled_df.at[neuron, 'pulse_response'] = pulse_responses[i]

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

    
def determine_baseline_sem(unrolled_df, pre_frame_num=100):
    '''Determine how long the baseline frame duration should be using the SEM'''

    x = np.arange(1, pre_frame_num+1)

    sems = list()
    for t in x:
        traces = list()
        for _, neuron in unrolled_df.iterrows():
            pulses = neuron['pulse_frames']
            
            for pulse in pulses:
                start_frame = pulse - t  # when the neuron traces will start
                stop_frame = pulse  # when the neuron traces will end

                trace = neuron['raw_norm_dff'][start_frame:stop_frame]
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


def get_traces(df, pre_frame_num=0, post_frame_num=10, normalize=False, normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, overlay_filter=None):
    '''From a temporal_df, gets desired traces around pulses
    Returns the x-axis (in frames) and list of individual traces'''
    x = np.arange(0-pre_frame_num, 0+post_frame_num+1)

    if overlay_filter is not None:
        overlay_filters = list()

    traces = list()
    for _, neuron in df.iterrows():
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
                    
                    traces.append(trace)

                    if overlay_filter is not None:
                        overlay_filters.append(neuron[overlay_filter])

        else:
            if normalize_by_first:
                baseline = 0

            for i, pulse in enumerate(pulses):
                start_frame = pulse - pre_frame_num  # when the neuron traces will start
                stop_frame = pulse + post_frame_num  # when the neuron traces will end

                trace = neuron[key][start_frame:stop_frame+1]

                if normalize:
                    baseline = np.median(neuron[key][start_frame:pulse])
                elif normalize_by_first and i == 0:
                    baseline = np.median(neuron[key][start_frame:pulse])

                trace = (trace - baseline) / baseline
                traces.append(trace)

                if overlay_filter is not None:
                    overlay_filters.append(neuron[overlay_filter])

    if overlay_filter is not None:
        return x, traces, overlay_filters
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

            plt.ylim(top=20, bottom=-2)

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


def plot_pulse_averages_overlayed(df, overlay_filter, filterby, overlay_order=None, savefig=False, save_path=None, **kwargs):
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
            for of in overlay_order:
                tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                avg_trace = np.mean(np.array(tr), axis=0)
                sems = sem(np.array(tr), axis=0)

                plt.plot (x, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}')
                plt.fill_between(x, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

        else:  # if you don't care about the order, it will just find the unique overlay filters
            for of in np.unique(overlay_filters):
                tr = np.array(traces)[np.where(np.array(overlay_filters) == of)[0]]
                avg_trace = np.mean(np.array(tr), axis=0)
                sems = sem(np.array(tr), axis=0)

                plt.plot (x, avg_trace, zorder=102, label=f'{of}, n={len(subdf[subdf[overlay_filter] == of])}')
                plt.fill_between(x, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)

        plt.axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
        plt.legend()

        plt.ylim(top=2, bottom=-0.50)

        # if normalize or normalize_by_first:
        #     plt.ylim(top=1, bottom=-0.50)
        # else:
        #     plt.ylim(top=1, bottom=0)
            
        plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

        if savefig:
            plt.savefig(save_path.joinpath("pulse_average_by_" + overlay_filter + "_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)


def calculate_aucs(df, filterby, auc_frame_nums=list(), **kwargs):
    '''Calculates the area under the curve for each filter
    auc_frame_nums: indices of traces to retrieve AUCs from. first item should be the start index and the second item should be the stop index (included)'''
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


def plot_individual_pulses(df, filterby, pre_frame_num=15, post_frame_num=30, normalize=False, normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, savefig=False, save_path=None):
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
                        start_frame = pulses[pulse-1] - pre_frame_num  # when the neuron traces will start
                        stop_frame = pulses[pulse-1] + post_frame_num  # when the neuron traces will end

                        trace = neuron[key][start_frame:stop_frame+1]

                        if normalize:
                            baseline = np.mean(neuron[key][start_frame:pulses[pulse-1]])
                        elif normalize_by_first and i == 0:
                            baseline = np.mean(neuron[key][start_frame:pulses[pulse-1]])

                        trace = trace - baseline

                        if pulse not in traces:
                            traces[pulse] = list()

                        traces[pulse].append(trace)

                        axs[pulse-1].plot(x, trace, 'lightgray', alpha=0.5)

            else:
                if normalize_by_first:
                    baseline = 0

                for i, pulse in enumerate(pulses):
                    start_frame = pulse - pre_frame_num  # when the neuron traces will start
                    stop_frame = pulse + post_frame_num  # when the neuron traces will end

                    trace = neuron[key][start_frame:stop_frame+1]

                    if normalize:
                        baseline = np.mean(neuron[key][start_frame:pulse])
                    elif normalize_by_first and i == 0:
                        baseline = np.mean(neuron[key][start_frame:pulse])

                    trace = trace - baseline

                    if i+1 not in traces:
                        traces[i+1] = list()

                    traces[i+1].append(trace)

                    axs[i].plot(x, trace, 'lightgray', alpha=0.5)

        for pulse in traces:
            avg_trace = np.array(traces[pulse]).mean(axis=0)
            sems = sem(np.array(traces[pulse]), axis=0)

            axs[pulse-1].plot (x, avg_trace, zorder=102)
            axs[pulse-1].fill_between(x, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)
            axs[pulse-1].axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)

            if normalize or normalize_by_first:
                axs[pulse-1].set_ylim(top=1, bottom=-0.50)
            else:
                axs[pulse-1].set_ylim(top=1, bottom=0)

            axs[pulse-1].set_title(pulse, fontsize=36)
                    
        plt.suptitle(" - ".join([str(cond) for cond in conditions]), fontsize=42)

        if savefig and not only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_" + "_".join([str(cond) for cond in conditions]) + "_all.pdf"), transparent=True)
        elif savefig and only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)

        plt.show()
    

def plot_individual_pulses_overlayed(df, overlay_filter, filterby, pre_frame_num=15, post_frame_num=30, normalize=False, normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, savefig=False, save_path=None):
    '''Separates each pulse and plots averages from different filters overlayed on top'''
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
                        start_frame = pulses[pulse-1] - pre_frame_num  # when the neuron traces will start
                        stop_frame = pulses[pulse-1] + post_frame_num  # when the neuron traces will end

                        trace = neuron[key][start_frame:stop_frame+1]

                        if normalize:
                            baseline = np.mean(neuron[key][start_frame:pulses[pulse-1]])
                        elif normalize_by_first and i == 0:
                            baseline = np.mean(neuron[key][start_frame:pulses[pulse-1]])

                        trace = trace - baseline

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
                        baseline = np.mean(neuron[key][start_frame:pulse])
                    elif normalize_by_first and i == 0:
                        baseline = np.mean(neuron[key][start_frame:pulse])

                    trace = trace - baseline

                    if i+1 not in traces:
                        traces[i+1] = list()
                        overlay_filters[i+1] = list()
                    
                    traces[i+1].append(trace)
                    overlay_filters.append(neuron[overlay_filter])
                    # axs[i].plot(x, trace, 'lightgray', alpha=0.5)

        for pulse in traces:
            for of in np.unique(overlay_filters[pulse]):
                tr = np.array(traces[pulse])[np.where(np.array(overlay_filters[pulse]) == of)[0]]
                avg_trace = np.array(tr).mean(axis=0)
                sems = sem(np.array(tr), axis=0)

                axs[pulse-1].plot (x, avg_trace, zorder=102, label=of)
                axs[pulse-1].fill_between(x, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)
            
            axs[pulse-1].axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
            axs[pulse-1].legend()

            if normalize or normalize_by_first:
                axs[pulse-1].set_ylim(top=1, bottom=-0.50)
            else:
                axs[pulse-1].set_ylim(top=1, bottom=0)

            axs[pulse-1].set_title(pulse, fontsize=36)
            
        plt.suptitle(" - ".join([str(cond) for cond in conditions]), fontsize=42)

        if savefig and not only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_by_" + overlay_filter + "_" + "_".join([str(cond) for cond in conditions]) + "_all.pdf"), transparent=True)
        elif savefig and only_responsive:
            plt.savefig(save_path.joinpath("individual_pulses_by_" + overlay_filter + "_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)


def plot_pulses_overlayed(df, filterby, pre_frame_num=15, post_frame_num=30, normalize=False, normalize_by_first=False, key='raw_norm_temporal', only_responsive=False, savefig=False, save_path=None):
    '''Plots averages from each pulse overlayed on top'''
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
                            baseline = np.mean(neuron[key][start_frame:pulses[pulse-1]])
                        elif normalize_by_first and i == 0:
                            baseline = np.mean(neuron[key][start_frame:pulses[pulse-1]])

                        trace = trace - baseline

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
                        baseline = np.mean(neuron[key][start_frame:pulse])
                    elif normalize_by_first and i == 0:
                        baseline = np.mean(neuron[key][start_frame:pulse])

                    trace = trace - baseline

                    if i+1 not in traces:
                        traces[i+1] = list()
                    
                    traces[i+1].append(trace)
                    # axs[i].plot(x, trace, 'lightgray', alpha=0.5)

        for pulse in sorted(traces.keys()):
            # avg_trace = np.array(traces[pulse]).mean(axis=0)
            avg_trace = np.median(traces[pulse], axis=0)
            # sems = sem(np.array(traces[pulse]), axis=0)
            sems = t.interval(0.95, len(np.array(traces[pulse]))-1, loc=np.median(traces[pulse], axis=0))

            plt.plot (x, avg_trace, zorder=102, label=pulse)
            plt.fill_between(x, avg_trace-sems, avg_trace+sems, alpha=0.2, zorder=101)
        
        plt.axvspan(-1, 0, color='red', lw=2, alpha=0.2, ec=None, zorder=100)
        plt.legend()

        if normalize or normalize_by_first:
            plt.ylim(top=1, bottom=-0.50)
        else:
            plt.ylim(top=1, bottom=0)
            
        plt.title(" - ".join([str(cond) for cond in conditions]), fontsize=18)

        if savefig and not only_responsive:
            plt.savefig(save_path.joinpath("overlayed_pulses_by_" + "_".join([str(cond) for cond in conditions]) + "_all.pdf"), transparent=True)
        elif savefig and only_responsive:
            plt.savefig(save_path.joinpath("overlayed_pulses_by_" + "_".join([str(cond) for cond in conditions]) + ".pdf"), transparent=True)