import logging
from fish import Fish
import caiman as cm
from utils import calculate_fps
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.utils.visualization import plot_contours
from mesm import load_mesmerize
import numpy as np
import pickle


def caiman_mcorr(fish, plane=None, **opts_dict):
    '''TODO: Implement it for non-volumetric recordings'''
    '''Motion correct a plane using NoRMCorre'''
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.WARNING)

    if plane is not None:
        img_path = [fish.data_paths['volumes'][str(plane)]['image']]
        m_orig = cm.load_movie_chain(img_path)
    else:
        raise ValueError('Give a plane index to run motion correction on')

    fps = calculate_fps(fish.data_paths['volumes'][str(plane)]['frametimes'])
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

    with open(fish.data_paths['postgavage_path'].joinpath('opts.pkl'), 'wb') as fp:
        pickle.dump(opts_dict, fp)

    fish.data_paths['opts'] = fish.data_paths['postgavage_path'].joinpath('opts.pkl')
    fish.process_volumetric_filestructure()

    return images


def caiman_cnmf(fish, plane=None, **opts_dict):
    '''TODO: Implement it for non-volumetric recordings'''
    '''Run CNMF on a plane for source extraction'''
    logging.basicConfig(format=
                        "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        # filename="/tmp/caiman.log",
                        level=logging.WARNING)

    if plane is not None:
        if 'mesmerize' in fish.data_paths.keys():
            mes_df = load_mesmerize(fish)
            input_movie_path = f'img_stack_{plane}\image.tif'

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
            img_path = fish.data_paths['volumes'][str(plane)]['C_frames']
            with open(fish.data_paths['opts'], 'rb') as fp:
                _opts = pickle.load(fp)

        Yr, dims, T = cm.load_memmap(img_path)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
    else:
        raise ValueError('Give a plane index to run motion correction on')

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

    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    with open(fish.data_paths['postgavage_path'].joinpath('opts.pkl'), 'wb') as fp:
        pickle.dump(_opts, fp)

    cm.stop_server(dview=dview)

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
