import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import random
import scipy.cluster.hierarchy as sch
from scipy.stats import sem, gaussian_kde

from .mesm import load_mesmerize, load_rois, uuid_to_plane
from .utils import save_pickle


def save_temporal(fish):
    '''Saves the temporal components of final ROIs as a temporal.h5 file'''
    mes_df = uuid_to_plane(load_mesmerize(fish))
    final_rois = load_rois(fish)

    planes = []
    temporal = []
    roi_indices = []
    pulse_frames = []

    for i, row in mes_df.iterrows():
        if row.algo == 'cnmf':

            name = row['item_name']
            if name not in final_rois.keys():
                continue

            plane = name[name.rfind('_')+1:]
            planes.append(int(plane))

            fts = pd.read_hdf(fish.data_paths['volumes'][plane]['frametimes'])
            pulses = [fts[fts.pulse == pulse].index.values[0] for pulse in fts.pulse.unique() if pulse != 0]
            pulse_frames.append(pulses)

            temp = row.cnmf.get_temporal('good')
            indices = final_rois[name]
            temporal.append(temp[indices])
            roi_indices.append(indices)

    temporal_df = pd.DataFrame({'plane': planes,
                                'temporal': temporal,
                                'roi_indices': roi_indices,
                                'pulse_frames': pulse_frames})
    temporal_df.sort_values(by=['plane'], ignore_index=True, inplace=True)
    temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    fish.process_filestructure()


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

    fish.process_filestructure()

    return new_temp


def compute_dff(fish):
    '''Computes the dF/F signal for each component'''
    fish.temporal_df['dff'] = None

    for i, row in fish.temporal_df.iterrows():
        pulse_0 = row.pulse_frames[0]
        dffs = []

        for comp in row.temporal:
            baseline = comp[:pulse_0]
            f0 = np.median(baseline)
            dff = (comp - f0)/abs(f0)
            dffs.append(dff)

        fish.temporal_df['dff'][i] = dffs

    fish.temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    return fish.temporal_df


def normalize_dff(fish):
    '''Normalizes each dF/F signal between 0 and 1'''
    fish.temporal_df['norm_dff'] = None

    for i, row in fish.temporal_df.iterrows():
        norm_dffs = []

        for comp in row.dff:
            norm_dff = (comp - min(comp)) / (max(comp) - min(comp))
            norm_dffs.append(norm_dff)

        fish.temporal_df['norm_dff'][i] = norm_dffs

    fish.temporal_df.to_hdf(fish.exp_path.joinpath('temporal.h5'), key='temporal')

    return fish.temporal_df


def cluster_temporal(fish, max_inter_cluster_dist, sort=True, savefig=False):
    '''
    Clusters temporal responses using hierarchical and flat clustering
    
    max_inter_cluster_dist: threshold for maximum inter-cluster distance allowed
    sort: sorts the cluster keys in ascending order if True
    '''
    # Hierarchical clustering of temporal responses
    d = sch.distance.pdist(fish.vol_temporal)
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
    plt.hlines(max_inter_cluster_dist, 0, len(d), color='r')
    plt.title(f'Clustering: max_inter_cluster_dist={max_inter_cluster_dist}')
    plt.show()

    if savefig:
        plt.savefig(fish.exp_path.joinpath("hierarchical_clustering.pdf"), transparent=True)
    
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


def plot_temporal(fish, plane, indices=None, heatmap=False, key=None):
    '''Plots individual temporal components of a plane'''
    row = fish.temporal_df[fish.temporal_df.plane == plane].iloc[0]

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
            for pulse in row.pulse_frames:
                axs[i].vlines(pulse, t.min(), t.max(), colors='r')

        plt.show()

    else:
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(temporal, cmap='plasma', interpolation='nearest')
        for pulse in fish.get_pulse_frames():
            plt.vlines(pulse, 0, len(temporal)-1, color='r')
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
        for pulse in fish.get_pulse_frames():
            ax1.vlines(pulse, -0.5, len(data)-0.5, color='r')
        ax1.title.set_text(title)

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
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(data, cmap='plasma', interpolation='nearest')
        for pulse in fish.get_pulse_frames():
            plt.vlines(pulse, 0, len(data)-1, color='r')
        plt.title(title)

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
        for pulse in fish.get_pulse_frames():
            axes[i].vlines(pulse, 0, 1, color='r')

    if savefig:    
        plt.savefig(fish.exp_path.joinpath("cluster_representative_traces.pdf"), transparent=True)
    

def plot_average_traces(fish, clusters, savefig=False):
    '''Given a clusters dictionary, plots the mean temporal trace per cluster'''
    fig, axes = plt.subplots(len(clusters), 1, sharex=True, sharey=True, figsize=(20, 20))

    for i, (cluster, temp) in enumerate(clusters.items()):
        t = np.mean(temp, axis=0)
        err = sem(temp)
        x = np.linspace(0, len(t), len(t))

        axes[i].plot(t)
        axes[i].fill_between(x, t-err, t+err, alpha=0.2)
        axes[i].title.set_text(cluster)
        for pulse in fish.get_pulse_frames():
            axes[i].vlines(pulse, 0, 1, color='r')

    if savefig:
        plt.savefig(fish.exp_path.joinpath("cluster_average_traces.pdf"), transparent=True)


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
                        plt.scatter(com[0]*2, com[1]*2, s=150, color=colors[i], label=cluster)
                    else:
                        plt.scatter(com[0]*2, com[1]*2, s=150, color=colors[i])
                
                except IndexError:  # if we run out of colors
                    colors += colors

                    if j == 0:  # add a label for the first component of each cluster
                        plt.scatter(com[0]*2, com[1]*2, s=150, color=colors[i], label=cluster)
                    else:
                        plt.scatter(com[0]*2, com[1]*2, s=150, color=colors[i])

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

    if distribution:
        # x-axis distribution plots require additional rows
        n_rows = np.ceil(len(com_clusters) / n_cols) * 2
    else:
        n_rows = np.ceil(len(com_clusters) / n_cols)
    
    fig, axes = plt.subplots(int(n_rows), n_cols, sharex=True, figsize=(10, np.ceil(len(com_clusters) / n_cols) * 5), 
                            height_ratios=[10, 1] * int(n_rows/2))

    cmap = get_cmap('Set1')  # type: matplotlib.colors.ListedColormap
    cmap2 = get_cmap('Accent')  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors + cmap2.colors  # type: list

    for i, (cluster, coms) in enumerate(com_clusters.items()):
        
        if distribution:
            if i % 2 == 0:
                img_row = i
                dist_row = i + 1
            else:
                img_row = i - 1
                dist_row = i
        else:
            img_row = int(i / n_cols)
        
        axes[img_row, int(i % n_cols)].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[img_row, int(i % n_cols)].title.set_text(cluster)
        
        coords = list()  # list of x coordinates if distribution is True
        
        for com in coms:
            try:
                axes[img_row, int(i % n_cols)].scatter(com[0]*2, com[1]*2, s=50, color=colors[i])
            except IndexError:  # if we run out of colors
                colors += colors
                axes[img_row, int(i % n_cols)].scatter(com[0]*2, com[1]*2, s=50, color=colors[i])
            
            if distribution:
                coords.append(com[0]*2)
        
        if len(coords) > 1:
            density = gaussian_kde(coords)
            density.covariance_factor = lambda : .25
            density._compute_covariance()

            xs = np.linspace(0, img.shape[1], 200)

            axes[dist_row, int(i % n_cols)].plot(xs, density(xs))

    plt.subplots_adjust(hspace=0.2)

    if savefig:
        plt.savefig(fish.exp_path.joinpath("clusters_spatial_individual.pdf"), transparent=True)