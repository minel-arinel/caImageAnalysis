import caiman as cm
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
import glob
import logging
import numpy as np
import os
import pandas as pd
import pickle

from caImageAnalysis import BrukerFish, VolumeFish
from caImageAnalysis.mesm import load_mesmerize
from caImageAnalysis.utils import calculate_fps


def caiman_mcorr(fish, plane=None, **opts_dict):
    '''Motion correct a plane using NoRMCorre'''
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.WARNING)

    if plane is not None:
        img_path = [fish.data_paths['volumes'][str(plane)]['image']]
    elif 'rotated' in fish.data_paths.keys():
        img_path = [fish.data_paths['rotated']]
    else:
        img_path = [fish.data_paths['raw_image']]
    
    m_orig = cm.load_movie_chain(img_path)

    if fish.volumetric:
        fps = calculate_fps(fish.data_paths['volumes'][str(plane)]['frametimes'])
    else:
        fps = calculate_fps(fish.data_paths['frametimes'])

    transient = 1  # in seconds; 1 for GCaMP8m, 1.5 for GCaMP6s

    # dataset dependent parameters
    opts_dict['fnames'] = img_path
    opts_dict['fr'] = fps  # imaging rate in frames per second
    opts_dict['decay_time'] = transient  # length of a typical transient in seconds

    opts = params.CNMFParams(params_dict=opts_dict)

    # start a cluster for parallel processing
    # (if a cluster already exists it will be closed and a new session will be opened)
    try:
        cm.stop_server(dview=dview)
    except:
        pass
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=None,
                                                     single_thread=False)

    # first we create a motion correction object with the parameters specified
    mc = MotionCorrect(img_path, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # Run piecewise-rigid motion correction using NoRMCorre
    mc.motion_correct(save_movie=True)
    m_els = cm.load(mc.fname_tot_els)
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # maximum shift to be used for trimming against NaNs

    # MEMORY MAPPING
    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0, dview=dview) # exclude borders
    print(fname_new)
    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    #load frames in python format (T x X x Y)

    # restart cluster to clean up memory
    cm.stop_server(dview=dview)

    with open(fish.exp_path.joinpath('opts.pkl'), 'wb') as fp:
        pickle.dump(opts_dict, fp)

    fish.data_paths['opts'] = fish.exp_path.joinpath('opts.pkl')
    
    if isinstance(fish, VolumeFish):
        fish.process_volumetric_filestructure()
    elif isinstance(fish, BrukerFish):
        fish.process_filestructure()

    return images


def caiman_cnmf(fish, plane=None, **opts_dict):
    '''TODO: Implement it for non-volumetric recordings'''
    '''Run CNMF on a plane for source extraction'''
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.WARNING)

    if plane is not None:
        img_path = [fish.data_paths['volumes'][str(plane)]['image']]
    elif 'rotated' in fish.data_paths.keys():
        img_path = [fish.data_paths['rotated']]
    else:
        img_path = [fish.data_paths['raw_image']]
    
    if plane is not None and 'mesmerize' in fish.data_paths.keys():
        mes_df = load_mesmerize(fish)
        input_movie_path = f'img_stack_{plane}/image.tif'

        if input_movie_path in mes_df.input_movie_path.values:
            idx = mes_df.input_movie_path.eq(input_movie_path).idxmax()
            row = mes_df.iloc[idx]
            input_movie_path = str(mes_df.paths.resolve(row.input_movie_path))
            uuid = row.uuid

            try:
                cm.stop_server(dview=dview)
            except:
                pass
            c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                                n_processes=None,
                                                                single_thread=False)

            img_path = cm.save_memmap(
                [input_movie_path], base_name=f"{uuid}_cnmf-memmap_", order="C", dview=dview
            )

            _opts = row.params['main']

        else:
            raise ValueError(f'Plane {plane} is not in the Mesmerize dataframe')
    
    else:
        if plane is not None:
            img_path = fish.data_paths['volumes'][str(plane)]['C_frames']
        else:
            img_path = str(fish.data_paths['C_frames'])
        
        with open(fish.data_paths['opts'], 'rb') as fp:
            _opts = pickle.load(fp)

    Yr, dims, T = cm.load_memmap(img_path, mode='r+')
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    _opts['fnames'] = [str(img_path)]

    for key in opts_dict.keys():
        _opts[key] = opts_dict[key]

    opts = params.CNMFParams(params_dict=_opts)

    try:
        cm.stop_server(dview=dview)
    except:
        pass
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=None,
                                                     single_thread=False)

    cnm = cnmf.CNMF(n_processes=n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    cnm = cnm.refit(images, dview=dview)

    # COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    # Extract DF/F values
    cnm.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    # Select only high quality components
    cnm.estimates.select_components(use_object=True)

    results_path = str(fish.exp_path.joinpath('analysis_results.hdf5'))
    cnm.save(results_path)

    with open(fish.exp_path.joinpath('opts.pkl'), 'wb') as fp:
        pickle.dump(_opts, fp)

    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return cnm, images


def plot_components(cnm, images, key=None):
    '''Plots components after CNMF
    key: 'good' for accepted components, 'bad' for rejected components'''
    Cn = cm.local_correlations(images.transpose(1,2,0))
    Cn[np.isnan(Cn)] = 0

    if key is None:
        # plot contours of found components
        cnm.estimates.plot_contours_nb(img=Cn, idx=cnm.estimates.idx_components)

    elif key == 'good':
        # accepted components
        cnm.estimates.nb_view_components(img=Cn,
                                         idx=cnm.estimates.idx_components,
                                         cmap='gray',
                                         thr=0.8,
                                         denoised_color='red')
    elif key == 'bad':
        # rejected components
        if len(cnm.estimates.idx_components_bad) > 0:
            cnm.estimates.nb_view_components(img=Cn,
                                             idx=cnm.estimates.idx_components_bad,
                                             cmap='gray',
                                             thr=0.8,
                                             denoised_color='red')
        else:
            print("No components were rejected.")
    else:
        raise ValueError('Key not accepted. Enter \'good\' for accepted components, \'bad\' for rejected components.')
