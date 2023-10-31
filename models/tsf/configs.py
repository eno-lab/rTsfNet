from .core import r_tsf_net
from .utils import *
from .feature import gen_default_ts_feature_params
#from sympy import sieve

def get_config(dataset, lr_magnif=1):

    if dataset == "daphnet":
        extract_imu_tensor_func = extract_imu_tensor_func_daphnet
    elif dataset.startswith("pamap2"):
        extract_imu_tensor_func = extract_imu_tensor_func_pamap2
    elif dataset.startswith('opportunity_real-task_c'):
        extract_imu_tensor_func = extract_imu_tensor_func_oppotunity_task_c
    elif dataset.startswith('opportunity'):
        extract_imu_tensor_func = extract_imu_tensor_func_oppotunity
    elif dataset.startswith('ucihar'):
        extract_imu_tensor_func = extract_imu_tensor_func_ucihar
    elif dataset.startswith('wisdm'):
        extract_imu_tensor_func = extract_imu_tensor_func_wisdm
    #elif dataset == 'ispl':
    else:
        raise NotImplementedError(f"No extract_imu_tensor_func implementation for {dataset}")


    if dataset.startswith('pamap2'):
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

        ext_tsf_params['autocorralation'] = np.arange(1, 32, step=1)

        ext_tsf_params256  = ext_tsf_params.copy()
        ext_tsf_params256['autocorralation'] = np.arange(1, 128, step=1)

        config = {
                'learning_rate': 0.001 * lr_magnif, 
                'regularization_rate': 0.00001, 
                'extract_imu_tensor_func': extract_imu_tensor_func,
                'use_orig_input': False,
                'no_rms_for_rot': False,
                'dropout_rate': 0.5, 
                ####################
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
                ########################
                'tagging': True,
                ########################
                'tsf_block_sizes': [64,256],
                'tsf_block_strides': [64,256],
                'tsf_block_ext_tsf_params': [ext_tsf_params,ext_tsf_params256],
                'rot_tsf_block_sizes': [64,256],
                'rot_tsf_block_strides': [64,256],
                'rot_tsf_block_ext_tsf_params': [ext_tsf_params,ext_tsf_params256],
                ########################
                'version':5,
                'imu_rot_num': 4,
                'rot_out_num': [0,1,2,3],
                ######################
                'btm_head_num': 1,
                'btm_no_out_activation': False, 
                #########################
                }

    elif dataset.startswith('opportunity-real'):

        # not tuned.
        # just a copy of opportunity
        ext_tsf_params = {}
        ext_tsf_params['abs_energy'] = True
        ext_tsf_params['count_above_tb_start'] = True
        ext_tsf_params['count_above_tb_end'] = True
        ext_tsf_params['abs_sum_of_changes'] = True
        ext_tsf_params['mean_change'] = True
        ext_tsf_params['min'] = True
        ext_tsf_params['max'] = True
        ext_tsf_params['number_crossing_25p'] = True
        ext_tsf_params['number_crossing_75p'] = True
        ext_tsf_params['rms'] = True
        ext_tsf_params['fft_amp_ratio_agg_mean'] = True
        ext_tsf_params['fft_amp_agg_skew'] = True
        ext_tsf_params['autocorralation'] = np.arange(1, 8, step=1)
        ext_tsf_params['autocorralation_mean'] = True
        ext_tsf_params['autocorralation_kurt'] = True

        ext_tsf_params32  = ext_tsf_params.copy()
        ext_tsf_params32['autocorralation'] = np.arange(1, 16, step=1)

        config = {
                'learning_rate': 0.001 * lr_magnif, 
                'regularization_rate': 0.00001, 
                'extract_imu_tensor_func': extract_imu_tensor_func,
                'use_orig_input': False,
                'no_rms_for_rot': False,
                'dropout_rate': 0.5, 
                'depth':3, 'base_kn': 128, 
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 16, 
                'block_mixer_depth': 4, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':1, 'rot_base_kn': 16, 
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 16, 
                'rot_block_mixer_depth':4, 'rot_block_mixer_base_kn': 128, 
                ########################
                'tagging': True, 
                ####################
                'imu_rot_num': 4,
                'tsf_block_sizes': [16, 32],
                'tsf_block_strides': [16, 32],
                'tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params32],
                'rot_tsf_block_sizes': [16, 32],
                'rot_tsf_block_strides': [16, 32],
                'rot_tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params32],
                #####################
                'version': 5,
                'rot_out_num': [0,1,2,3],  
                #################
                'ax_weight_depth':4, 'ax_weight_base_kn': 128, 
                'tsf_weight_depth':1, 'tsf_weight_base_kn': 64, 
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16, 
                'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 64, 
                }

    elif dataset.startswith('opportunity'):
        ####################
        ext_tsf_params = {}
        ext_tsf_params['abs_energy'] = True
        ext_tsf_params['count_above_tb_start'] = True
        ext_tsf_params['count_above_tb_end'] = True
        ext_tsf_params['abs_sum_of_changes'] = True
        ext_tsf_params['mean_change'] = True
        ext_tsf_params['min'] = True
        ext_tsf_params['max'] = True
        ext_tsf_params['number_crossing_25p'] = True
        ext_tsf_params['number_crossing_75p'] = True
        ext_tsf_params['rms'] = True
        ext_tsf_params['fft_amp_ratio_agg_mean'] = True
        ext_tsf_params['fft_amp_agg_skew'] = True
        ext_tsf_params['autocorralation'] = np.arange(1, 8, step=1)
        ext_tsf_params['autocorralation_mean'] = True
        ext_tsf_params['autocorralation_kurt'] = True

        ext_tsf_params32  = ext_tsf_params.copy()
        ext_tsf_params32['autocorralation'] = np.arange(1, 16, step=1)

        config = {
                'learning_rate': 0.001 * lr_magnif, 
                'regularization_rate': 0.00001, 
                'extract_imu_tensor_func': extract_imu_tensor_func,
                'use_orig_input': False,
                'no_rms_for_rot': False,
                'dropout_rate': 0.5, 
                'depth':3, 'base_kn': 128, 
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 16, 
                'block_mixer_depth': 4, 'block_mixer_base_kn': 128,
                ####################
                'rot_depth':1, 'rot_base_kn': 16, 
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 16, 
                'rot_block_mixer_depth':4, 'rot_block_mixer_base_kn': 128, 
                ########################
                'tagging': True,
                ########################
                'imu_rot_num': 4,
                'tsf_block_sizes': [16, 32],
                'tsf_block_strides': [16, 32],
                'tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params32],
                'rot_tsf_block_sizes': [16, 32],
                'rot_tsf_block_strides': [16, 32],
                'rot_tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params32],
                ####################
                # for ispl
                #'imu_rot_num': 4,
                #'tsf_block_sizes': [15],
                #'tsf_block_strides': [15],
                #'tsf_block_ext_tsf_params': [ext_tsf_params],
                #'rot_tsf_block_sizes': [15],
                #'rot_tsf_block_strides': [15],
                #'rot_tsf_block_ext_tsf_params': [ext_tsf_params],
                #####################
                'version': 5,
                'rot_out_num': [0,1,2,3],  
                #################
                'ax_weight_depth':4, 'ax_weight_base_kn': 128, 
                'tsf_weight_depth':1, 'tsf_weight_base_kn': 64, 
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16, 
                'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 64, 
                }

    elif dataset.startswith('ucihar'):
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
        ext_tsf_params['autocorralation'] = np.arange(1, 16, step=1) 
        ext_tsf_params128 = ext_tsf_params.copy()
        ext_tsf_params128['autocorralation'] = np.arange(1, 64, step=1)

        config = {
                'learning_rate': 0.001 * lr_magnif, 
                'regularization_rate': 0.00001, 
                'extract_imu_tensor_func': extract_imu_tensor_func,
                'no_rms_for_rot': False,
                'dropout_rate': 0.5, 
                #####################
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
                ########################
                'tagging': True,
                ##########################
                'concatenate_blocks_with_same_size_and_stride': False,
                'tsf_block_sizes': [32,128],
                'tsf_block_strides': [32,128],
                'tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params128],
                'rot_tsf_block_sizes': [32,128],
                'rot_tsf_block_strides': [32,128],
                'rot_tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params128],
                #'use_add_for_blk_concatenate': True,
                #######################
                'btm_head_num': 1,
                'btm_no_out_activation': False, 
                #########################
                'version': 5,
                'rot_out_num': [0,1,2,3],
                'imu_rot_num': 4,
                'use_orig_input': False,
                # for no rot
                #'imu_rot_num': 0,
                #'use_orig_input': True,
                }

    elif dataset.startswith('daphnet'):

        #########
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
        ext_tsf_params['autocorralation'] = np.arange(1, 16, step=1) 
        ext_tsf_params192 = ext_tsf_params.copy()
        ext_tsf_params192['autocorralation'] = np.arange(1, 86, step=1)

        config = {
                'learning_rate': 0.001 * lr_magnif, 
                'regularization_rate': 0.00001, 
                'extract_imu_tensor_func': extract_imu_tensor_func,
                'use_orig_input': False,
                'no_rms_for_rot': False,
                'dropout_rate': 0.5, 
                'depth':1, 'base_kn': 128, 
                'tsf_mixer_depth':1, 'tsf_mixer_base_kn': 128, 
                'block_mixer_depth': 1, 'block_mixer_base_kn': 32,
                ####################
                'rot_depth':2, 'rot_base_kn': 64, 
                'rot_tsf_mixer_depth':1, 'rot_tsf_mixer_base_kn': 128, 
                'rot_block_mixer_depth':1, 'rot_block_mixer_base_kn': 32, 
                ########################
                'tagging': True,  
                #########################
                #'few_norm': True,
                #'few_acti': True,
                ####################
                'tsf_block_sizes': [32,192],
                'tsf_block_strides': [32,192],
                'tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params192],
                'rot_tsf_block_sizes': [32,192,],
                'rot_tsf_block_strides': [32,192],
                'rot_tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params192],
                ######################
                'version': 5,
                'rot_out_num': [0,1,2,3],
                'imu_rot_num': 4,
                ###################
                'ax_weight_depth':3, 'ax_weight_base_kn': 128, 
                'tsf_weight_depth':1, 'tsf_weight_base_kn': 16, 
                'rot_ax_weight_depth':3, 'rot_ax_weight_base_kn': 16, 
                'rot_tsf_weight_depth':1, 'rot_tsf_weight_base_kn': 16, 
                }

    elif dataset.startswith('wisdm'):

        # not tuned.
        # it is the copy of ucihar
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

        ext_tsf_params['autocorralation'] = np.arange(1, 8, step=1) 
        ext_tsf_params64= ext_tsf_params.copy()
        ext_tsf_params64['autocorralation'] = np.arange(1, 32, step=1)

        config = {
                'learning_rate': 0.001 * lr_magnif, 
                'regularization_rate': 0.00001, 
                'extract_imu_tensor_func': extract_imu_tensor_func,
                'no_rms_for_rot': False,
                'dropout_rate': 0.5, 
                ####################
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
                ########################
                'tagging': True, 
                ##########################
                #'few_norm': True,
                #'few_acti': True,
                ###################
                'imu_rot_num': 4,
                'tsf_block_sizes': [16,64],
                'tsf_block_strides': [16,64],
                'tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params64],
                'rot_tsf_block_sizes': [16,64],
                'rot_tsf_block_strides': [16,64],
                'rot_tsf_block_ext_tsf_params': [ext_tsf_params, ext_tsf_params64],
                ###################
                'version': 5,
                'rot_out_num': [0,1,2,3],
                'imu_rot_num': 4,
                'use_orig_input': False,
                ###################
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

        if dataset.startswith('ucihar'):
            for elem in ['tsf_block_sizes', 
                         'tsf_block_strides', 
                         'rot_tsf_block_sizes', 
                         'rot_tsf_block_strides']:
                config[elem] = [32, 128]
            ext_tsf_params['autocorralation'] = np.arange(1, 16, step=1) 
            ext_tsf_params_long['autocorralation'] = np.arange(1, 64, step=1) 
        elif dataset.startswith('opportunity'):
            for elem in ['tsf_block_sizes', 
                         'tsf_block_strides', 
                         'rot_tsf_block_sizes', 
                         'rot_tsf_block_strides']:
                config[elem] = [16, 32]
                #config[elem] = [15]
            ext_tsf_params['autocorralation'] = np.arange(1, 8, step=1) 
            ext_tsf_params_long['autocorralation'] = np.arange(1, 16, step=1) 
            #ext_tsf_params['autocorralation'] = np.arange(1, 8, step=1) 
        elif dataset.startswith('pamap2'):
            for elem in ['tsf_block_sizes', 
                         'tsf_block_strides', 
                         'rot_tsf_block_sizes', 
                         'rot_tsf_block_strides']:
                config[elem] = [64, 128]
            ext_tsf_params['autocorralation'] = np.arange(1, 32, step=1) 
            ext_tsf_params_long['autocorralation'] = np.arange(1, 128, step=1) 
        elif dataset.startswith('daphnet'):
            for elem in ['tsf_block_sizes', 
                         'tsf_block_strides', 
                         'rot_tsf_block_sizes', 
                         'rot_tsf_block_strides']:
                config[elem] = [32, 192]
            ext_tsf_params['autocorralation'] = np.arange(1, 16, step=1) 
            ext_tsf_params_long['autocorralation'] = np.arange(1, 96, step=1) 
        elif dataset.startswith('wisdm'):
            for elem in ['tsf_block_sizes', 
                         'tsf_block_strides', 
                         'rot_tsf_block_sizes', 
                         'rot_tsf_block_strides']:
                config[elem] = [16, 64] 
            ext_tsf_params['autocorralation'] = np.arange(1, 8, step=1) 
            ext_tsf_params_long['autocorralation'] = np.arange(1, 32, step=1) 
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
