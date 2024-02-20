from .core import r_tsf_net
from .utils import *
from .feature import gen_default_ts_feature_params
#from sympy import sieve

def get_config(dataset, lr_magnif=1):

    if dataset == "daphnet":
        extract_imu_tensor_func = extract_imu_tensor_func_daphnet
    elif dataset.startswith("pamap2-separation"):
        extract_imu_tensor_func = extract_imu_tensor_func_pamap2_separation
    elif dataset.startswith("pamap2"):
        extract_imu_tensor_func = extract_imu_tensor_func_pamap2
    elif dataset.startswith('opportunity_real-task_c'):
        extract_imu_tensor_func = extract_imu_tensor_func_oppotunity_task_c
    elif dataset.startswith('opportunity-separation'):
        extract_imu_tensor_func = extract_imu_tensor_func_oppotunity_separation
    elif dataset.startswith('opportunity'):
        extract_imu_tensor_func = extract_imu_tensor_func_oppotunity
    elif dataset.startswith('ucihar'):
        extract_imu_tensor_func = extract_imu_tensor_func_ucihar
    elif dataset.startswith('wisdm'):
        extract_imu_tensor_func = extract_imu_tensor_func_wisdm
    elif dataset.startswith('m_health'):
        extract_imu_tensor_func = extract_imu_tensor_func_m_health
    elif dataset.startswith('real_world-separation'):
        extract_imu_tensor_func = extract_imu_tensor_func_real_world_separation
    elif dataset.startswith('real_world'):
        extract_imu_tensor_func = extract_imu_tensor_func_real_world
    elif dataset.startswith('mighar'):
        extract_imu_tensor_func = extract_imu_tensor_func_mighar
    #elif dataset == 'ispl':
    else:
        raise NotImplementedError(f"No extract_imu_tensor_func implementation for {dataset}")

    def build_base_configs(short_win, long_win):
        ext_tsf_params = {}
        ext_tsf_params['rms'] = True
        ext_tsf_params['abs_energy'] = True
        ext_tsf_params['abs_sum_of_changes'] = True
        ext_tsf_params['count_above_tb_end'] = True
        ext_tsf_params['count_above_tb_start'] = True
        ext_tsf_params['max'] = True
        ext_tsf_params['min'] = True
        ext_tsf_params['number_crossing_75p'] = True
        ext_tsf_params['number_crossing_25p'] = True
        ext_tsf_params['mean_change'] = True
        ext_tsf_params['fft_amp_agg_skew'] = True
        ext_tsf_params['fft_amp_ratio_agg_mean'] = True

        ext_tsf_params['autocorralation_mean'] = True
        ext_tsf_params['autocorralation_kurt'] = True

        ext_tsf_params_long= ext_tsf_params.copy()

        ext_tsf_params_map = {
                short_win: ext_tsf_params,
                long_win: ext_tsf_params_long,
                }

        for win_len in ext_tsf_params_map:
            ext_p = ext_tsf_params_map[win_len]
            ext_p['autocorralation'] = np.arange(1, win_len//2, step=1)

        config = {
                'learning_rate': 0.001 * lr_magnif,
                'regularization_rate': 0.00001,
                'extract_imu_tensor_func': extract_imu_tensor_func,
                'use_orig_input': False,
                'no_rms_for_rot': False,
                'dropout_rate': 0.5,
                ####################
                'depth':2, 'base_kn': 32,
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 32,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 16,
                ####################
                'rot_depth':4, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 32,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 16,
                ###################
                'ax_weight_depth':1, 'ax_weight_base_kn': 16,
                'tsf_weight_depth':4, 'tsf_weight_base_kn': 32,
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 32,
                'rot_tsf_weight_depth':4, 'rot_tsf_weight_base_kn': 32,
                ########################
                'tagging': True,
                #########################
                'tsf_block_sizes': list(ext_tsf_params_map.keys()),
                'tsf_block_strides': list(ext_tsf_params_map.keys()),
                'tsf_block_ext_tsf_params': list(ext_tsf_params_map.values()),
                'rot_tsf_block_sizes': list(ext_tsf_params_map.keys()),
                'rot_tsf_block_strides': list(ext_tsf_params_map.keys()),
                'rot_tsf_block_ext_tsf_params': list(ext_tsf_params_map.values()),
                #######################
                'version':5,
                'imu_rot_num': 4,
                'rot_out_num': [0,1,2,3],
                ######################
                'btm_head_num': 1,
                'btm_no_out_activation': False,
                #########################
                'concatenate_blocks_with_same_size_and_stride': False,
                'use_add_for_blk_concatenate': False,
                }

        return config 



    if dataset.startswith('pamap2-separation'):
        config = build_base_configs(64, 256)
        # not tuned.
        # just a copy of pamap2 
        config |= {
                'depth':1, 'base_kn': 128,
                'tsf_mixer_depth':2, 'tsf_mixer_base_kn': 128,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':3, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':2, 'rot_tsf_mixer_base_kn': 128,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 128,
                ###################
                'ax_weight_depth':3, 'ax_weight_base_kn': 16,
                'tsf_weight_depth':3, 'tsf_weight_base_kn': 32,
                'rot_ax_weight_depth':1, 'rot_ax_weight_base_kn': 16,
                'rot_tsf_weight_depth':3, 'rot_tsf_weight_base_kn': 32,
                }

    elif dataset.startswith('pamap2'):
        config = build_base_configs(64, 256)
        config |= {
                'depth':1, 'base_kn': 128,
                'tsf_mixer_depth':2, 'tsf_mixer_base_kn': 128,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':3, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':2, 'rot_tsf_mixer_base_kn': 128,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 128,
                ###################
                'ax_weight_depth':3, 'ax_weight_base_kn': 16,
                'tsf_weight_depth':3, 'tsf_weight_base_kn': 32,
                'rot_ax_weight_depth':1, 'rot_ax_weight_base_kn': 16,
                'rot_tsf_weight_depth':3, 'rot_tsf_weight_base_kn': 32,
                }

    elif dataset.startswith('opportunity-real'):
        config = build_base_configs(16, 32)
        # not tuned.
        # just a copy of opportunity
        config |= {
                'depth':3, 'base_kn': 128,
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 16,
                'block_mixer_depth': 4, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':1, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 16,
                'rot_block_mixer_depth':4, 'rot_block_mixer_base_kn': 128,
                #################
                'ax_weight_depth':4, 'ax_weight_base_kn': 128,
                'tsf_weight_depth':1, 'tsf_weight_base_kn': 64,
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16,
                'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 64,
                }

    elif dataset.startswith('opportunity-separation'):

        config = build_base_configs(16, 32)

        # not tuned.
        # just a copy of opportunity
        config |= {
                'depth':3, 'base_kn': 128,
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 16,
                'block_mixer_depth': 4, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':1, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 16,
                'rot_block_mixer_depth':4, 'rot_block_mixer_base_kn': 128,
                #################
                'ax_weight_depth':4, 'ax_weight_base_kn': 128,
                'tsf_weight_depth':1, 'tsf_weight_base_kn': 64,
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16,
                'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 64,
                }

    elif dataset.startswith('opportunity'):

        config = build_base_configs(16, 32)
        config |= {
                'depth':3, 'base_kn': 128,
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 16,
                'block_mixer_depth': 4, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':1, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 16,
                'rot_block_mixer_depth':4, 'rot_block_mixer_base_kn': 128,
                #################
                'ax_weight_depth':4, 'ax_weight_base_kn': 128,
                'tsf_weight_depth':1, 'tsf_weight_base_kn': 64,
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16,
                'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 64,
                }

        # for iSPL Benchmark setting
        #config = build_base_configs(15)
        #config |= {
        #        'depth':3, 'base_kn': 128,
        #        'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 16,
        #        'block_mixer_depth': 4, 'block_mixer_base_kn': 128,
        #        ####################
        #        'rot_depth':1, 'rot_base_kn': 16,
        #        'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 16,
        #        'rot_block_mixer_depth':4, 'rot_block_mixer_base_kn': 128,
        #        #################
        #        'ax_weight_depth':4, 'ax_weight_base_kn': 128,
        #        'tsf_weight_depth':1, 'tsf_weight_base_kn': 64,
        #        'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16,
        #        'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 64,
        #        }

    elif dataset.startswith('ucihar'):
        config = build_base_configs(32, 128)
        config |= {
                'depth':1, 'base_kn': 32,
                'tsf_mixer_depth':2, 'tsf_mixer_base_kn': 128,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 64,
                ####################
                'rot_depth':3, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':2, 'rot_tsf_mixer_base_kn': 128,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 64,
                ###################
                'ax_weight_depth':4, 'ax_weight_base_kn': 16,
                'tsf_weight_depth':3, 'tsf_weight_base_kn': 128,
                'rot_ax_weight_depth':4, 'rot_ax_weight_base_kn': 128,
                'rot_tsf_weight_depth':3, 'rot_tsf_weight_base_kn': 128,
                }

    elif dataset.startswith('daphnet'):

        config = build_base_configs(32, 192)
        config |= {
                'depth':1, 'base_kn': 128,
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 128,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 32,
                ####################
                'rot_depth':2, 'rot_base_kn': 64,
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 128,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 32,
                ########################
                'ax_weight_depth':3, 'ax_weight_base_kn': 128,
                'tsf_weight_depth':1, 'tsf_weight_base_kn': 16,
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16,
                'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 16,
                }

    elif dataset.startswith('wisdm'):

        config = build_base_configs(16, 32)

        # not tuned.
        # it is the copy of ucihar
        config |= {
                'depth':1, 'base_kn': 32,
                'tsf_mixer_depth':2, 'tsf_mixer_base_kn': 128,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 64,
                ####################
                'rot_depth':3, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':2, 'rot_tsf_mixer_base_kn': 128,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 64,
                ###################
                'ax_weight_depth':4, 'ax_weight_base_kn': 16,
                'tsf_weight_depth':3, 'tsf_weight_base_kn': 128,
                'rot_ax_weight_depth':4, 'rot_ax_weight_base_kn': 128,
                'rot_tsf_weight_depth':3, 'rot_tsf_weight_base_kn': 128,
                }

    elif dataset.startswith('m_health'):

        config = build_base_configs(32, 128)

        config |= {
                ####################
                'depth':2, 'base_kn': 32,
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 32,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 16,
                ####################
                'rot_depth':4, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 32,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 16,
                ###################
                'ax_weight_depth':1, 'ax_weight_base_kn': 16,
                'tsf_weight_depth':4, 'tsf_weight_base_kn': 32,
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 32,
                'rot_tsf_weight_depth':4, 'rot_tsf_weight_base_kn': 32,
                }

    elif dataset.startswith('real_world'):

        config = build_base_configs(32, 128)

        # not tuned.
        # it is the copy of pamap2
        config |= {
                'depth':3, 'base_kn': 16,
                'tsf_mixer_depth':3, 'tsf_mixer_base_kn': 16,
                'block_mixer_depth': 2, 'block_mixer_base_kn': 16,
                ####################
                'rot_depth':3, 'rot_base_kn': 32,
                'rot_tsf_mixer_depth':3, 'rot_tsf_mixer_base_kn': 16,
                'rot_block_mixer_depth':2, 'rot_block_mixer_base_kn': 16,
                ###################
                'ax_weight_depth':3, 'ax_weight_base_kn': 64,
                'tsf_weight_depth':4, 'tsf_weight_base_kn': 32,
                'rot_ax_weight_depth':4, 'rot_ax_weight_base_kn': 128,
                'rot_tsf_weight_depth':4, 'rot_tsf_weight_base_kn': 32,
                }

    elif dataset.startswith('mighar'):
        config = build_base_configs(64, 256)

        config |= {
                'depth':1, 'base_kn': 128,
                'tsf_mixer_depth':2, 'tsf_mixer_base_kn': 128,
                'block_mixer_depth': 1, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':3, 'rot_base_kn': 16,
                'rot_tsf_mixer_depth':2, 'rot_tsf_mixer_base_kn': 128,
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 128,
                ###################
                'ax_weight_depth':3, 'ax_weight_base_kn': 16,
                'tsf_weight_depth':3, 'tsf_weight_base_kn': 32,
                'rot_ax_weight_depth':1, 'rot_ax_weight_base_kn': 16,
                'rot_tsf_weight_depth':3, 'rot_tsf_weight_base_kn': 32,
                }

    else:
        raise NotImplementedError(f"No preconfiged parameters for {dataset}")

    return config

def get_optim_config(dataset, trial, lr_magnif=1):
    config = get_config(dataset, lr_magnif)

    #######################################################
    # PLEASE EDIT HERE
    #######################################################
    is_tsf_selection = False
    #is_tsf_selection = True

    #######################################################
    # TSF selection
    #######################################################
    if is_tsf_selection:
        ext_tsf_params = gen_default_ts_feature_params()

        params_for_true = ['autocorralation_mean', 'autocorralation_kurt']
        ext_tsf_params = gen_default_ts_feature_params()

        for key in ext_tsf_params:
            if key in params_for_true:
                ext_tsf_params[key] = True
                continue

            ext_tsf_params[key] = trial.suggest_int(key, low=0, high=1) == 1

        ext_tsf_params_long = ext_tsf_params.copy()
        config['tsf_block_ext_tsf_params'] = [ext_tsf_params, ext_tsf_params_long],
        config['rot_tsf_block_ext_tsf_params'] = [ext_tsf_params, ext_tsf_params_long],

        def setup_ac(short_win, long_win):
            for elem in ['tsf_block_sizes',
                         'tsf_block_strides',
                         'rot_tsf_block_sizes',
                         'rot_tsf_block_strides']:
                config[elem] = [short_win, long_win]
            ext_tsf_params['autocorralation'] = np.arange(1, short_win/2, step=1)
            ext_tsf_params_long['autocorralation'] = np.arange(1, long_win/2, step=1)

        if dataset.startswith('ucihar'):
            setup_ac(32,128)
        elif dataset.startswith('opportunity'):
            setup_ac(16,32)
        elif dataset.startswith('pamap2'):
            setup_ac(64,256)
        elif dataset.startswith('daphnet'):
            setup_ac(32, 192)
        elif dataset.startswith('wisdm'):
            setup_ac(32, 128)
        else:
            raise NotImplementedError("Please add new implementation here")
    #####################################################

    else:

    #######################################################
    # param selection
    #######################################################
        base_kn = 32*2**trial.suggest_int('base_kn', low=-1, high=2)
        depth = trial.suggest_int('depth', low=1, high=4)
        rot_base_kn = 32*2**trial.suggest_int('rot_base_kn', low=-1, high=2)
        rot_depth = trial.suggest_int('rot_depth', low=1, high=4)

        tsf_mixer_base_kn = 32*2**trial.suggest_int('tsf_mixer_base_kn', low=-1, high=2)
        tsf_mixer_depth = trial.suggest_int('tsf_mixer_depth', low=1, high=4)
        block_mixer_base_kn = 32*2**trial.suggest_int('block_mixer_base_kn', low=-1, high=2)
        block_mixer_depth= trial.suggest_int('block_mixer_depth', low=1, high=4)
        ax_weight_base_kn = 32*2**trial.suggest_int('ax_weight_base_kn', low=-1, high=2)
        ax_weight_depth = trial.suggest_int('ax_weight_depth', low=1, high=4)
        rot_ax_weight_base_kn = 32*2**trial.suggest_int('rot_ax_weight_base_kn', low=-1, high=2)
        rot_ax_weight_depth = trial.suggest_int('rot_ax_weight_depth', low=1, high=4)
        tsf_weight_base_kn = 32*2**trial.suggest_int('tsf_weight_base_kn', low=-1, high=2)
        tsf_weight_depth = trial.suggest_int('tsf_weight_depth', low=1, high=4)

        config['base_kn'] = base_kn
        config['depth'] = depth
        config['tsf_mixer_base_kn'] = tsf_mixer_base_kn
        config['tsf_mixer_depth'] = tsf_mixer_depth
        config['block_mixer_base_kn'] = block_mixer_base_kn
        config['block_mixer_depth'] = block_mixer_depth
        config['rot_base_kn'] = rot_base_kn
        config['rot_depth'] = rot_depth
        config['rot_tsf_mixer_base_kn'] = tsf_mixer_base_kn
        config['rot_tsf_mixer_depth'] = tsf_mixer_depth
        config['rot_block_mixer_base_kn'] = block_mixer_base_kn
        config['rot_block_mixer_depth'] = block_mixer_depth
        config['ax_weight_base_kn'] = ax_weight_base_kn
        config['ax_weight_depth'] = ax_weight_depth
        config['rot_ax_weight_base_kn'] = rot_ax_weight_base_kn
        config['rot_ax_weight_depth'] = rot_ax_weight_depth
        config['tsf_weight_base_kn'] = tsf_weight_base_kn
        config['tsf_weight_depth'] = tsf_weight_depth
        config['rot_tsf_weight_base_kn'] = tsf_weight_base_kn
        config['rot_tsf_weight_depth'] = tsf_weight_depth
    #######################################################

    return config


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return r_tsf_net(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)

def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config
