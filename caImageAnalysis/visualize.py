from collections import OrderedDict
from fastplotlib import ImageWidget, Plot
from fastplotlib.graphics.line_slider import LineSlider
from ipywidgets import FloatSlider, FloatText, Label, HBox, link, Layout
import matplotlib.pyplot as plt
from mesmerize_core import *
import numpy as np
import pandas as pd
from tifffile import memmap


def visualize_images(imgs=list(), path=None, names=None):
    '''Visualize a list of images within the same plot using ImageWidget'''
    names = names
    add_names = False

    if names is None:
        add_names = True
        names = []

    if path is not None:
        img = memmap(path)
        imgs.append(img)

    if isinstance(imgs, list):
        for i, img in enumerate(imgs):
            img_rot = np.flip(img, axis=1)
            imgs[i] = img_rot
            if add_names:
                names.append(f'index: {i}')
    else:
        if add_names:
            names.append('image')

    iw = ImageWidget(imgs, names=names, vmin_vmax_sliders=True, cmap="gnuplot2")
    return iw


def visualize_volumes(fish, names=None):
    '''Visualize individual planes of a given fish'''
    planes = []

    for i in fish.data_paths['volumes'].keys():
        img = memmap(fish.data_paths['volumes'][i]['image'])
        planes.append(img)

    iw = visualize_images(imgs=planes, names=names)
    return iw


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

    for subplot in iw.plot:
        subplot.graphics[0].cmap = "jet"

    return iw


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
        contours_graphic[ixs_good].colors = 'cyan'
        contours_graphic[ixs_bad].colors = 'magenta'

        # temporal plot
        plot_temporal = Plot()

        temporal_graphic = plot_temporal.add_line_collection(temporal, colors="cyan", name="temporal")
        temporal_graphic[ixs_good].colors = 'cyan'
        temporal_graphic[ixs_bad].colors = 'magenta'

        # voltage output lines
        name = row['item_name']
        plane = name[name.rfind('_')+1:]
        fts = pd.read_hdf(fish.data_paths['volumes'][plane]['frametimes'])
        
        for pulse in fts.pulse.unique():
            if pulse != 0:
                pulse_frame = fts[fts.pulse == pulse].index.values[0]

                xs = [pulse_frame] * 2
                line = np.dstack([xs, [temporal.min(), temporal.max()]])[0]
                plot_temporal.add_line(data=line, thickness=3, colors='red', name=f'pulse_{pulse}')

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
    '''Maps click events to contour'''
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
