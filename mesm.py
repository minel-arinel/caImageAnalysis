from mesmerize_core import *
import os
from utils import *
from fish import Fish
import pandas as pd
from itertools import product
import numpy as np
from copy import deepcopy
from pathlib import Path
from fastplotlib import ImageWidget, Plot
from fastplotlib.graphics.line_slider import LineSlider
from fastplotlib.graphics.text import TextGraphic
from ipywidgets import FloatSlider, FloatText, Label, HBox, VBox, link, Layout
from collections import OrderedDict
from sklearn.neighbors import BallTree as tree
import matplotlib.pyplot as plt
from math import ceil
import pickle


def load_mesmerize(fish):
    '''Loads mesmerize-batch df'''
    set_parent_raw_data_path(fish.data_paths['postgavage_path'])
    batch_path = get_parent_raw_data_path().joinpath("mesmerize-batch/batch.pickle")

    if os.path.exists(batch_path):
        print('Batch exists. Loading batch.pickle')
        df = load_batch(batch_path)
    else:
        print('Batch does not exist. Creating batch.pickle')
        df = create_batch(batch_path)
        fish.data_paths['mesmerize'] = batch_path

    return df


def run_mesmerize(df):
    '''Runs mesmerize on a given series or batch dataframe'''
    if isinstance(df, pd.core.series.Series):
        row = df
        process = row.caiman.run(backend='local')

    elif isinstance(df, pd.core.frame.DataFrame):
        for i, row in df.iterrows():
            if row.outputs is not None and row.outputs['success']:
                continue

            process = row.caiman.run(backend='local')

            # on Windows you MUST reload the batch dataframe after every iteration because it uses the `local` backend.
            # this is unnecessary on Linux & Mac
            # "DummyProcess" is used for local backend so this is automatic
            if process.__class__.__name__ == "DummyProcess":
                df = df.caiman.reload_from_disk()


def add_mcorr(fish, img_paths, default=None, grid=False, **params):
    '''Runs motion correction with different parameters on Mesmerize
    grid: if True, creates a grid from the cartesian product of the parameters'''
    df = load_mesmerize(fish)

    if not isinstance(img_paths, list):
        raise TypeError('img_paths needs to be list of strings')

    if default is None:
        default = \
            {
                'main':
                    {
                        'strides': (40, 40),
                        'overlaps': (15, 15),
                        'max_shifts': (20, 20),
                        'max_deviation_rigid': 20,
                        'pw_rigid': True,
                    },
            }

    if grid:
        param_grid = product(*params.values())
        for row in param_grid:
            new_params = deepcopy(default)

            for i, val in enumerate(row):
                if list(params.keys())[i] in ['strides', 'overlaps', 'max_shifts']:
                    mesval = [val, val]
                else:
                    mesval = val

                new_params['main'][list(params.keys())[i]] = mesval

            for img in img_paths:
                df.caiman.add_item(algo='mcorr', item_name=Path(img).parent.name,
                                   input_movie_path=img, params=new_params)
    else:
        if len(params) > 0:
            for p in params:
                for val in params[p]:
                    new_params = deepcopy(default)
                    if p in ['strides', 'overlaps', 'max_shifts']:
                        mesval = [val, val]
                    else:
                        mesval = val

                    new_params['main'][p] = mesval

                    for img in img_paths:
                        df.caiman.add_item(algo='mcorr', item_name=Path(img).parent.name,
                                           input_movie_path=img, params=new_params)
        else:
            for img in img_paths:
                df.caiman.add_item(algo='mcorr', item_name=Path(img).parent.name,
                                   input_movie_path=img, params=default)

    return df


def add_cnmf(fish, img_paths, default=None, grid=False, **params):
    '''Runs CNMF with different parameters on Mesmerize
    grid: if True, creates a grid from the cartesian product of the parameters'''
    df = load_mesmerize(fish)

    if not isinstance(img_paths, list):
        raise TypeError('img_paths needs to be list of strings')

    transient = 1  # in seconds; 1 for GCaMP8m, 1.5 for GCaMP6s

    if default is None:
        default = \
            {
                'main':
                    {
                        'fr': 30, # framerate, very important!
                        'p': 1,
                        'nb': 2,
                        'merge_thr': 0.85,
                        'rf': 15,
                        'stride': 6, # "stride" for cnmf, "strides" for mcorr
                        'K': 6,
                        'gSig': [5, 5],
                        'ssub': 1,
                        'tsub': 1,
                        'method_init': 'greedy_roi',
                        'min_SNR': 2.0,
                        'rval_thr': 0.85,
                        'use_cnn': True,
                        'min_cnn_thr': 0.8,
                        'cnn_lowest': 0.1,
                        'decay_time': transient,
                    },
                'refit': True, # If `True`, run a second iteration of CNMF
            }

    if grid:
        param_grid = product(*params.values())
        for row in param_grid:
            new_params = deepcopy(default)

            for i, val in enumerate(row):
                if list(params.keys())[i] == 'gSig':
                    mesval = [val, val]
                else:
                    mesval = val

                new_params['main'][list(params.keys())[i]] = mesval

            for img in img_paths:
                name = Path(img).parent.name
                if name.rfind('_') == -1:  # for running on mesmerize outputs
                    plane_name = df[df.uuid == name].item_name.values[0]
                else:
                    plane_name = name

                plane = plane_name[plane_name.rfind('_')+1:]
                fps = calculate_fps(fish.data_paths['volumes'][plane]['frametimes'])
                new_params['main']['fr'] = fps

                df.caiman.add_item(algo='cnmf', item_name=name,
                                   input_movie_path=img, params=new_params)
    else:
        if len(params) > 0:
            for p in params:
                for val in params[p]:
                    new_params = deepcopy(default)
                    if p == 'gSig':
                        mesval = [val, val]
                    else:
                        mesval = val

                    new_params['main'][p] = mesval

                    for img in img_paths:
                        name = Path(img).parent.name
                        if name.rfind('_') == -1:  # for running on mesmerize outputs
                            name = df[df.uuid == name].item_name.values[0]

                        plane = name[name.rfind('_')+1:]
                        fps = calculate_fps(fish.data_paths['volumes'][plane]['frametimes'])
                        new_params['main']['fr'] = fps

                        df.caiman.add_item(algo='cnmf', item_name=name,
                                           input_movie_path=img, params=new_params)
        else:
            for img in img_paths:
                name = Path(img).parent.name
                if name.rfind('_') == -1:  # for running on mesmerize outputs
                    name = df[df.uuid == name].item_name.values[0]

                plane = name[name.rfind('_')+1:]
                fps = calculate_fps(fish.data_paths['volumes'][plane]['frametimes'])
                default['main']['fr'] = fps

                df.caiman.add_item(algo='cnmf', item_name=name,
                                   input_movie_path=img, params=default)

    return df


def add_volume(fish, algo):
    '''Runs mesmerize on the experiment volume with the given algo
    algo: 'mcorr' or 'cnmf' '''
    _df = load_mesmerize(fish)
    df = _df[_df.algo == algo]

    planes = []
    for i in fish.data_paths['volumes']:
        if algo == 'mcorr':
            planes.append(fish.data_paths['volumes'][i]['image'])
        elif algo == 'cnmf':
            planes.append(fish.data_paths['volumes'][i]['mcorr'])
        else:
            raise ValueError('algo needs to be either \'mcorr\' or \'cnmf\'')

    if len(df) == 1:
        test_path = df.iloc[0].caiman.get_input_movie_path()
        if test_path in planes:
            planes.remove(test_path)
    else:
        raise ValueError('Mesmerize dataframe should have a single entry')

    params = df.iloc[0].params

    if algo == 'mcorr':
        vol_df = add_mcorr(fish, planes, default=params)
    elif algo == 'cnmf':
        vol_df = add_cnmf(fish, planes, default=params)

    return vol_df


def visualize_mesmerize(df, algo, keys=None, roi_idxs=None, contrs='good'):
    '''Visualize results from a mesmerize dataframe
    keys: indices to visualize
    conts: for cnmf, which contours to visualize (can be 'all', 'good', or 'none' '''
    df = df[df['algo'] == algo]

    if algo == 'mcorr':
        movies = [df.iloc[0].caiman.get_input_movie()]
        subplot_names = ["raw"]
        for i, row in df.iterrows():
            movies.append(row.mcorr.get_output())
            subplot_names.append(f"index: {i}")

    elif algo == 'cnmf':
        items_contours = list()
        subplot_names = list()
        movies = list()
        for i, row in df.iterrows():
            if keys is None or (keys is not None and i in keys):
                movie = row.caiman.get_input_movie()
                movies.append(movie)

                contours, coms = row.cnmf.get_contours(contrs, swap_dim=False)
                if roi_idxs is not None:
                    contours = [contours[idx] for idx in roi_idxs]
                # flip the contours along the y-axis for visualization
                y_max = movie.shape[1]
                for contour in contours:
                    contour[:, 1] = y_max - contour[:, 1]
                # you can also pass "bad", "all" integer indices or a list/array of indices
                items_contours.append(contours)
                subplot_names.append(f"index: {i}")

    else:
        raise ValueError('algo must be \'mcorr\' or \'cnmf\'')

    # create the widget
    iw = visualize_images(imgs=movies, names=subplot_names)

    if algo == 'cnmf':
        for i, subplot in enumerate(iw.plot):  # enumerate gives the iteration number
            if i < len(items_contours):
                contours = items_contours[i]
                subplot.add_line_collection(contours, colors="w", alpha=0.7, name="contours")

    return iw


def uuid_to_plane(df):
    '''Changes the item_names with a uuid to the plane name'''
    for _, row in df.iterrows():
        if row.item_name.rfind('_') == -1:  # for running on mesmerize outputs
            row.item_name = df[df.uuid == row.item_name].item_name.values[0]

    return df


def visualize_temporal(fish, row):
    '''Visualizes spatial components of a single movie and their temporal responses'''
    if not isinstance(row, pd.core.series.Series):
        raise ValueError('Input must be a pandas Series')
    else:
        # get the motion corrected input movie as a memmap
        _cnmf_movie = row.caiman.get_input_movie()
        cnmf_movie = np.flip(_cnmf_movie, axis=1)

        # we can get the contours of the spatial components
        contours, coms = row.cnmf.get_contours("all", swap_dim=False)

        # flip the contours along the y-axis for visualization
        y_max = cnmf_movie.shape[1]
        for contour in contours:
            contour[:, 1] = y_max - contour[:, 1]

        # and temporal components
        temporal = row.cnmf.get_temporal("all")

        ixs_good = row.cnmf.get_good_components()
        ixs_bad = row.cnmf.get_bad_components()

        # for the image data and contours
        iw_cnmf = ImageWidget(cnmf_movie, vmin_vmax_sliders=True, cmap="gnuplot2")

        # add good contours to the plot within the widget
        contours_graphic = iw_cnmf.plot.add_line_collection(contours, colors="cyan", name="contours")
        contours_graphic[ixs_good].colors = "cyan"
        contours_graphic[ixs_bad].colors = "magenta"

        # temporal plot
        plot_temporal = Plot()

        temporal_graphic = plot_temporal.add_line_collection(temporal, colors="cyan", name="temporal")
        temporal_graphic[ixs_good].colors = "cyan"
        temporal_graphic[ixs_bad].colors = "magenta"

        # injection line
        name = row['item_name']
        plane = name[name.rfind('_')+1:]
        inj_frame = fish.data_paths['volumes'][plane]['inj_frame']

        xs = [inj_frame] * 2
        line = np.dstack([xs, [temporal.min(), temporal.max()]])[0]
        plot_temporal.add_line(data=line, thickness=3, colors="red", name="injection")

        # a vertical line that is synchronized to the image widget "t" (timepoint) slider
        _ls = LineSlider(x_pos=0, bounds=(temporal.min(), temporal.max()), slider=iw_cnmf.sliders["t"])
        plot_temporal.add_graphic(_ls)

        return plot_temporal, iw_cnmf


def compeval_sliders():
    '''Generates the evaluation sliders'''
    # low thresholds
    lt = OrderedDict(
        rval_lowest=(-1.0, -1.0, 1.0), # (val, min, max)
        SNR_lowest=(0.5, 0., 100),
        cnn_lowest=(0.1, 0., 1.0),
    )

    # high thresholds
    ht = OrderedDict(
        rval_thr=(0.8, 0., 1.0),
        min_SNR=(2.5, 0., 100),
        min_cnn_thr=(0.9, 0., 1.0),
    )

    lw = list()
    for k in lt:
        kwargs = dict(value=lt[k][0], min=lt[k][1], max=lt[k][2], step=0.01, description=k)
        slider = FloatSlider(**kwargs)
        entry = FloatText(**kwargs, layout=Layout(width="150px"))

        link((slider, "value"), (entry, "value"))

        lw.append(HBox([slider, entry]))

    hw = list()
    for k in ht:
        kwargs = dict(value=ht[k][0], min=ht[k][1], max=ht[k][2], step=0.01, description=k)
        slider = FloatSlider(**kwargs)
        entry = FloatText(**kwargs, layout=Layout(width="150px"))

        link((slider, "value"), (entry, "value"))

        hw.append(HBox([slider, entry]))

    label_eval = Label(value="")

    return lw, hw, label_eval


def visualize_compeval(fish, row):
    '''visualize_temporal() but with component evaluation metrics'''

    plot_l, iw = visualize_temporal(fish, row)
    lw, hw, label_eval = compeval_sliders()

    def get_eval_params():
        '''Gets the values from the GUI'''
        _eval_params = [{w.children[0].description: w.children[0].value for w in ws} for ws in [lw, hw]]
        return {**_eval_params[0], **_eval_params[1]}

    global eval_params
    eval_params = get_eval_params()

    @iw.plot.renderer.add_event_handler("resize")
    def update_with(*args):
        w = iw.plot.canvas.get_logical_size()[0]
        h = plot_l.canvas.get_logical_size()
        plot_l.canvas.set_logical_size(w, h)

    def update_eval(p):
        '''Animation function'''
        global eval_params

        new_eval_params = get_eval_params()

        if new_eval_params == eval_params:
            return
        eval_params = new_eval_params

        label_eval.value = "Please wait running eval..."
        # run eval
        row.cnmf.run_eval(new_eval_params)
        label_eval.value = ""

        # get the new indices after eval
        good_ixs = row.cnmf.get_good_components()
        bad_ixs = row.cnmf.get_bad_components()

        # if mode == "colors":
        #     # make sure all of them are present
        #     p["contours"][:].present = True
        #
        #     # set colors
        #     p["contours"][:].colors.block_events(True)
        #     p["contours"][good_ixs].colors = "cyan"
        #     p["contours"][bad_ixs].colors = "magenta"
        #     p["contours"][:].colors.block_events(False)

        # elif mode == "present":

        # make them all cyan
        p["contours"][:].colors = "cyan"

        # set present=True for good
        p["contours"][:].colors.block_events(True)
        p["contours"][good_ixs].present = True
        p["contours"][bad_ixs].present = False
        p["contours"][:].colors.block_events(False)

    iw.plot.add_animations(update_eval)

    return [
                plot_l,
                iw,
                label_eval,
                Label(value="Low Thresholds"),
                *lw,
                Label(value="High Thresholds"),
                *hw
            ]


def visualize_compeval_volume(fish):
    '''TODO: Get this to work. Currently gets stuck at evaluating'''
    '''Component evaluation on the volume'''
    mes_df = load_mesmerize(fish)
    df = mes_df[mes_df.algo == 'cnmf']

    iw = visualize_mesmerize(mes_df, 'cnmf', contrs='all')

    lw, hw, label_eval = compeval_sliders()

    def get_eval_params():
        '''Gets the values from the GUI'''
        _eval_params = [{w.children[0].description: w.children[0].value for w in ws} for ws in [lw, hw]]
        return {**_eval_params[0], **_eval_params[1]}

    global eval_params
    eval_params = get_eval_params()

    def update_eval(p):
        '''Animation function'''
        global eval_params

        new_eval_params = get_eval_params()

        if new_eval_params == eval_params:
            return
        eval_params = new_eval_params

        label_eval.value = "Please wait running eval..."

        for i, row in df.iterrows():
            # run eval
            row.cnmf.run_eval(new_eval_params)
            label_eval.value = ""

            # get the new indices after eval
            good_ixs = row.cnmf.get_good_components()
            bad_ixs = row.cnmf.get_bad_components()

            sub_p = iw.plot[f'index: {i}']

            # make them all cyan
            sub_p["contours"][:].colors = "cyan"

            # set present=True for good
            sub_p["contours"][:].colors.block_events(True)
            sub_p["contours"][good_ixs].present = True
            sub_p["contours"][bad_ixs].present = False
            sub_p["contours"][:].colors.block_events(False)

    iw.plot.add_animations(update_eval)

    return [
                iw,
                label_eval,
                Label(value="Low Thresholds"),
                *lw,
                Label(value="High Thresholds"),
                *hw
            ]


def interactive_temporal(plot_temporal, iw_cnmf):
    '''Maps click events of contours to temporal graphs'''
    plot_temporal.auto_scale()
    plot_temporal.camera.scale.x = 0.85

    temporal_graphic = plot_temporal['temporal']
    temporal_graphic[:].present = False

    image_graphic = iw_cnmf.plot["image"]
    contours_graphic = iw_cnmf.plot['contours']

    # link image to contours
    image_graphic.link(
        "click",
        target=contours_graphic,
        feature="colors",
        new_data="w",
        callback=euclidean
    )

    # thickness of contour
    contours_graphic.link("colors", target=contours_graphic, feature="thickness", new_data=5)

    # toggle temporal component when contour changes color
    contours_graphic.link("colors", target=temporal_graphic, feature="present", new_data=True)

    # autoscale temporal plot to the current temporal component
    temporal_graphic[:].present.add_event_handler(plot_temporal.auto_scale)


def euclidean(source, target, event, new_data):
    '''maps click events to contour'''
    # calculate coms of line collection
    indices = np.array(event.pick_info["index"])

    coms = list()

    for contour in target.graphics:
        coors = contour.data()[~np.isnan(contour.data()).any(axis=1)]
        com = coors.mean(axis=0)
        coms.append(com)

    # euclidean distance to find closest index of com
    indices = np.append(indices, [0])

    ix = int(np.linalg.norm((coms - indices), axis=1).argsort()[0])

    target._set_feature(feature="colors", new_data=new_data, indices=ix)

    return None


def visualize_diff(df):
    '''Visualizes the difference from mean for each mesmerize motion correction output'''
    iw = visualize_mesmerize(df, 'mcorr')

    means = [df.iloc[0].caiman.get_projection("mean")]
    for i, row in df.iterrows():
        means.append(row.caiman.get_projection("mean"))

    subtract_means = {}
    for i in range(len(means)):
        subtract_means[i] = lambda x: x - means[i]
    iw.frame_apply = subtract_means

    for sp in iw.plot:
        sp.graphics[0].cmap = "jet"

    return iw


def clean_mesmerize(df, keep_rows, keep_algo=None):
    '''Removes rows from mesmerize dataframe except the row indices in keep_rows'''
    if not isinstance(keep_rows, list):
        raise ValueError('keep_rows should be a list of row indices')

    if keep_algo is not None:
        keep_rows.extend(list(df[df.algo == keep_algo].index))

    final_rows = []
    for ind in keep_rows:
        final_rows.append(df.iloc[ind].uuid)

    try:
        for i, row in df.iterrows():
            if row.uuid not in final_rows:
                df.caiman.remove_item(row.uuid)

    except PermissionError:
        print('On Windows removing items will raise a PermissionError if you have the memmap file open')
        print('Restart the kernel and re-run the function')

    return df


def save_params(fish, params):
    '''Saves the params dict as a 'compeval_params.pickle' file under the mesmerize-batch folder'''
    path = fish.data_paths['mesmerize'].joinpath('compeval_params.pickle')
    save_pickle(params, path)


def load_params(fish):
    '''Loads the params pickle file under the mesmerize-batch folder as a dict'''
    path = fish.data_paths['mesmerize'].joinpath('compeval_params.pickle')
    return load_pickle(path)


def comp_eval2(fish, row, xy_cutoff=25, t_cutoff=100, dist_cutoff=5, intermediate_plot=True):
    '''Runs secondary component evaluation
    xy_cutoff: removes components with a com within the cutoff from each side
    t_cutoff: removes components with a max temporal peak less than the cutoff
    dist_cutoff: removes component with a com within the dist_cutoff of another component
                removes the com with the lowest temporal peak value
    intermediate_plot: plots for individual steps'''
    name = row['item_name']
    plane = name[name.rfind('_')+1:]
    inj_frame = fish.data_paths['volumes'][plane]['inj_frame']
    roi_ixs = row.cnmf.get_good_components()
    contours, coms = row.cnmf.get_contours('good', swap_dim=False)
    coms = np.array(coms)
    temporal = row.cnmf.get_temporal('good')

    movie = row.caiman.get_input_movie()
    t, y, x = movie.shape

    plt.imshow(movie[0])
    for com in coms:
        plt.scatter(com[0], com[1])
    plt.title(f'{row.item_name}: Before')
    plt.show()

    # Remove ROIs at the borders
    good_ixs = []

    if intermediate_plot:
        plt.imshow(movie[0])

    for i, com in enumerate(coms):
        if com[0] >= xy_cutoff and com[1] >= xy_cutoff:
            if com[0] < x-xy_cutoff and com[1] < y-xy_cutoff:
                good_ixs.append(i)
                if intermediate_plot:
                    plt.scatter(com[0], com[1])
    if intermediate_plot:
        plt.xlim(0, x)
        plt.ylim(y, 0)
        plt.title(f'{row.item_name}: After xy_cutoff')
        plt.show()

    good_ixs = np.array(good_ixs)

    # Remove ROIs with small temporal peaks
    if intermediate_plot:
        fig = plt.figure(2, figsize=(10, 1))
        for t in temporal[good_ixs]:
            if t.max() < t_cutoff:
                plt.scatter(t.max(), 1)
        plt.title(f'{row.item_name}: Peak t of components below t_cutoff')
        plt.show()

    better_ixs = []

    if intermediate_plot:
        plt.imshow(movie[0])
    for ix in good_ixs:
        if temporal[ix].max() > t_cutoff:
            better_ixs.append(ix)
            if intermediate_plot:
                plt.scatter(coms[ix][0], coms[ix][1])
    if intermediate_plot:
        plt.xlim(0, x)
        plt.ylim(y, 0)
        plt.title(f'{row.item_name}: After t_cutoff')
        plt.show()

    better_ixs = np.array(better_ixs)

    # Remove ROIs with close centers of mass
    res = tree(coms[better_ixs], metric='euclidean')
    dists, inds = res.query(coms[better_ixs], 2)

    bad_inds = []
    if intermediate_plot:
        fig = plt.figure(3, figsize=(10, 1))
    for i, d in enumerate(dists):
        if d[1] < dist_cutoff:
            bad_inds.append(inds[i])
            if intermediate_plot:
                plt.scatter(d[1], 1)
    if intermediate_plot:
        plt.title(f'{row.item_name}: Pairwise distance of components below dist_cutoff')
        plt.show()

    bad_inds = np.array(bad_inds)
    uniq_inds = np.unique(np.ndarray.flatten(bad_inds))

    if intermediate_plot:
        plt.imshow(movie[0])
        for ind in uniq_inds:
            plt.scatter(coms[better_ixs][ind][0], coms[better_ixs][ind][1])
        plt.xlim(0, x)
        plt.ylim(y, 0)
        plt.title(f'{row.item_name}: Close CoM components')
        plt.show()

    for i, inds in enumerate(bad_inds):
        if inds[1] < inds[0]:
            bad_inds[i] = np.flip(inds)

    cells, cnt = np.unique(bad_inds, axis=0, return_counts=True)
    close_cells = []
    for i, c in enumerate(cnt):
        if c == 2:
            close_cells.append(cells[i])

    remove_inds = []
    for pair in close_cells:
        if temporal[pair[0]].max() > temporal[pair[1]].max():
            remove_inds.append(pair[1])
        else:
            remove_inds.append(pair[0])

    best_ixs = np.copy(better_ixs)
    for ind in remove_inds:
        best_ixs = np.delete(best_ixs, np.where(best_ixs == better_ixs[ind]))

    # Final ROIs
    plt.imshow(movie[0])
    for com in coms[best_ixs]:
        plt.scatter(com[0], com[1])
    plt.xlim(0, x)
    plt.ylim(y, 0)
    plt.title(f'{row.item_name}: Final ROIs')
    plt.show()

    fig = plt.figure(4, figsize=(10, temporal[best_ixs].shape[0]))
    gs = fig.add_gridspec(temporal[best_ixs].shape[0], hspace=0)
    axs = gs.subplots(sharex=True)
    for i, t in enumerate(temporal[best_ixs]):
        axs[i].plot(t)
        axs[i].vlines(inj_frame, t.min(), t.max(), colors='r')
    plt.title(f'{row.item_name}: Final temporal components')
    plt.show()

    fig = plt.figure(3, figsize=(20, 20))
    temp = temporal[best_ixs]
    plt.imshow(temp, cmap='plasma', interpolation='nearest')
    plt.vlines(inj_frame, 0, 71, color='r')
    plt.title(f'{row.item_name}: Temporal heatmap')
    plt.show()

    actual_ixs = roi_ixs[best_ixs]

    return actual_ixs


def compeval2_volume(fish, xy_cutoff=25, t_cutoff=100, dist_cutoff=5):
    '''Runs compeval2 on the entire volume with given parameters'''
    mes_df = uuid_to_plane(load_mesmerize(fish))
    cnmf_df = mes_df[mes_df.algo == 'cnmf']

    final_rois = dict()

    for i, row in cnmf_df.iterrows():
        ixs = comp_eval2(fish, row, xy_cutoff=xy_cutoff, t_cutoff=t_cutoff, dist_cutoff=dist_cutoff)
        final_rois[row.item_name] = ixs

    return final_rois


def remove_xy(fish, indices=None, xy_cutoff=25):
    '''Removes components with a com within the cutoff from each side'''
    mes_df = uuid_to_plane(load_mesmerize(fish))
    cnmf_df = mes_df[mes_df.algo == 'cnmf'].reset_index()

    good_rois = dict()

    fig = plt.figure(figsize=(10, 20), constrained_layout=True)
    gs = fig.add_gridspec(len(cnmf_df), 2)
    axs = gs.subplots()
    for i, row in cnmf_df.iterrows():
        movie = row.caiman.get_input_movie()
        t, y, x = movie.shape
        axs[i, 0].imshow(movie[0])
        axs[i, 1].imshow(movie[0])
        axs[i, 0].set_title(f'{row.item_name}: Before')
        _, coms = row.cnmf.get_contours('good', swap_dim=False)
        coms = np.array(coms)

        if indices is None:
            ixs = np.arange(coms.shape[0])
        else:
            ixs = indices[row.item_name]

        _ixs = []

        for ix in ixs:
            com = coms[ix]
            axs[i, 0].scatter(com[0], com[1])
            if com[0] >= xy_cutoff and com[1] >= xy_cutoff:
                if com[0] < x-xy_cutoff and com[1] < y-xy_cutoff:
                    _ixs.append(ix)
                    axs[i, 1].scatter(com[0], com[1])

        axs[i, 0].set_xlim([0, x])
        axs[i, 0].set_ylim([y, 0])
        axs[i, 1].set_xlim([0, x])
        axs[i, 1].set_ylim([y, 0])
        axs[i, 1].set_title(f'{row.item_name}: After xy_cutoff')
        good_rois[row.item_name] = np.array(_ixs)
    plt.show()

    params = load_params(fish)
    params['xy_cutoff'] = xy_cutoff
    save_params(fish, params)

    return good_rois


def remove_low_t(fish, indices=None, t_cutoff=100):
    '''Removes components with a max temporal peak less than the cutoff'''
    mes_df = uuid_to_plane(load_mesmerize(fish))
    cnmf_df = mes_df[mes_df.algo == 'cnmf'].reset_index()

    good_rois = dict()

    fig = plt.figure(figsize=(10, 20), constrained_layout=True)
    gs = fig.add_gridspec(len(cnmf_df), 2)
    axs = gs.subplots()
    for i, row in cnmf_df.iterrows():
        movie = row.caiman.get_input_movie()
        t, y, x = movie.shape
        axs[i, 0].imshow(movie[0])
        axs[i, 1].imshow(movie[0])
        axs[i, 0].set_title(f'{row.item_name}: Before')
        _, coms = row.cnmf.get_contours('good', swap_dim=False)
        coms = np.array(coms)
        temporal = row.cnmf.get_temporal('good')

        if indices is None:
            ixs = np.arange(coms.shape[0])
        else:
            ixs = indices[row.item_name]

        _ixs = []

        for ix in ixs:
            com = coms[ix]
            axs[i, 0].scatter(com[0], com[1])
            if temporal[ix].max() > t_cutoff:
                _ixs.append(ix)
                axs[i, 1].scatter(com[0], com[1])

        axs[i, 0].set_xlim([0, x])
        axs[i, 0].set_ylim([y, 0])
        axs[i, 1].set_xlim([0, x])
        axs[i, 1].set_ylim([y, 0])
        axs[i, 1].set_title(f'{row.item_name}: After t_cutoff')
        good_rois[row.item_name] = np.array(_ixs)
    plt.show()

    params = load_params(fish)
    params['t_cutoff'] = t_cutoff
    save_params(fish, params)

    return good_rois


def plot_t_distribution(row, indices, t_cutoff=100):
    '''Plots the distribution of peak t values below the cutoff'''
    temporal = row.cnmf.get_temporal('good')[indices]
    fig = plt.figure(2, figsize=(10, 1))
    for t in temporal:
        if t.max() < t_cutoff:
            plt.scatter(t.max(), 1)
    plt.title(f'{row.item_name}: Peak t of components below t_cutoff')
    plt.show()


def remove_close_dist(fish, indices=None, dist_cutoff=100):
    '''Removes components with a com within the dist_cutoff of another component
    Removes the com with the lowest temporal peak value'''
    mes_df = uuid_to_plane(load_mesmerize(fish))
    cnmf_df = mes_df[mes_df.algo == 'cnmf'].reset_index()

    good_rois = dict()

    fig = plt.figure(figsize=(10, 20), constrained_layout=True)
    gs = fig.add_gridspec(len(cnmf_df), 2)
    axs = gs.subplots()
    for i, row in cnmf_df.iterrows():
        movie = row.caiman.get_input_movie()
        t, y, x = movie.shape
        axs[i, 0].imshow(movie[0])
        axs[i, 1].imshow(movie[0])
        axs[i, 0].set_title(f'{row.item_name}: Before')
        _, coms = row.cnmf.get_contours('good', swap_dim=False)
        coms = np.array(coms)
        temporal = row.cnmf.get_temporal('good')

        if indices is None:
            ixs = np.arange(coms.shape[0])
        else:
            ixs = indices[row.item_name]

        if len(ixs) >= 2:
            res = tree(coms[ixs], metric='euclidean')
            dists, inds = res.query(coms[ixs], 2)

            bad_ixs = []
            for j, d in enumerate(dists):
                if d[1] < dist_cutoff:
                    bad_ixs.append(inds[j])
            bad_ixs = np.array(bad_ixs)

            for j, inds in enumerate(bad_ixs):
                if inds[1] < inds[0]:
                    bad_ixs[j] = np.flip(inds)

            cells, cnt = np.unique(bad_ixs, axis=0, return_counts=True)
            close_cells = []
            for j, c in enumerate(cnt):
                if c == 2:
                    close_cells.append(cells[j])

            remove_inds = []
            for pair in close_cells:
                if temporal[pair[0]].max() > temporal[pair[1]].max():
                    remove_inds.append(pair[1])
                else:
                    remove_inds.append(pair[0])

            _ixs = np.copy(ixs)
            for ind in remove_inds:
                _ixs = np.delete(_ixs, np.where(_ixs == ixs[ind]))

            for ix in ixs:
                com = coms[ix]
                axs[i, 0].scatter(com[0], com[1])
                if ix in _ixs:
                    axs[i, 1].scatter(com[0], com[1])

            good_rois[row.item_name] = np.array(_ixs)

        else:
            for ix in ixs:
                com = coms[ix]
                axs[i, 0].scatter(com[0], com[1])
                axs[i, 1].scatter(com[0], com[1])
            good_rois[row.item_name] = np.array(ixs)


        axs[i, 0].set_xlim([0, x])
        axs[i, 0].set_ylim([y, 0])
        axs[i, 1].set_xlim([0, x])
        axs[i, 1].set_ylim([y, 0])
        axs[i, 1].set_title(f'{row.item_name}: After dist_cutoff')

    plt.show()

    params = load_params(fish)
    params['dist_cutoff'] = dist_cutoff
    save_params(fish, params)

    return good_rois


def save_rois(fish, _rois):
    '''Saves the ROIs dict as a 'final_rois.pickle' file under the mesmerize-batch folder'''
    rois = dict()
    for key, val in _rois.items():
        if len(val) != 0:
            rois[key] = val

    path = fish.data_paths['mesmerize'].joinpath('final_rois.pickle')
    save_pickle(rois, path)


def load_rois(fish):
    '''Loads the final_rois pickle file under the mesmerize-batch folder as a dict'''
    path = fish.data_paths['mesmerize'].joinpath('final_rois.pickle')
    return load_pickle(path)


def save_temporal(fish):
    '''Saves the temporal components of final ROIs as a temporal.h5 file'''
    mes_df = uuid_to_plane(load_mesmerize(fish))
    final_rois = load_rois(fish)

    planes = []
    temporal = []
    roi_indices = []
    inj_frames = []

    for i, row in mes_df.iterrows():
        if row.algo == 'cnmf':

            name = row['item_name']
            if name not in final_rois.keys():
                continue

            plane = name[name.rfind('_')+1:]
            inj = fish.data_paths['volumes'][plane]['inj_frame']

            planes.append(int(plane))
            inj_frames.append(inj)

            temp = row.cnmf.get_temporal('good')
            indices = final_rois[name]

            temporal.append(temp[indices])
            roi_indices.append(indices)

    temporal_df = pd.DataFrame({'plane': planes,
                                'temporal': temporal,
                                'roi_indices': roi_indices,
                                'inj_frame': inj_frames})
    temporal_df.sort_values(by=['plane'], ignore_index=True, inplace=True)
    temporal_df.to_hdf(fish.data_paths['postgavage_path'].joinpath('temporal.h5'), key='temporal')


def plot_single_rois(row, indices):
    '''Plots individual ROIs of a given plane'''
    fig_height = 720
    n_cols = 5
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(800*px, 2400*px), constrained_layout=True)
    gs = fig.add_gridspec(ceil(len(indices)/n_cols), n_cols)
    axs = gs.subplots()

    movie = row.caiman.get_input_movie()

    t, y, x = movie.shape
    _, coms = row.cnmf.get_contours('good', swap_dim=False)
    coms = np.array(coms)
    temporal = row.cnmf.get_temporal('good')

    if len(indices) > n_cols:
        for i, ind in enumerate(indices):
            axs[int(i / n_cols), int(i % n_cols)].imshow(movie[0])
            axs[int(i / n_cols), int(i % n_cols)].set_title(f'Index: {ind}')

            axs[int(i / n_cols), int(i % n_cols)].scatter(coms[ind][0], coms[ind][1], s=2, c='r')

            axs[int(i / n_cols), int(i % n_cols)].set_xlim([0, x])
            axs[int(i / n_cols), int(i % n_cols)].set_ylim([y, 0])

    else:
    # if there is a single row of ROIs
        for i, ind in enumerate(indices):
            axs[int(i % n_cols)].imshow(movie[0])
            axs[int(i % n_cols)].set_title(f'Index: {ind}')

            axs[int(i % n_cols)].scatter(coms[ind][0], coms[ind][1], s=2, c='r')

            axs[int(i % n_cols)].set_xlim([0, x])
            axs[int(i % n_cols)].set_ylim([y, 0])

    plt.show()
