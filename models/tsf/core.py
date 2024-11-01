from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, Concatenate, Activation, Dense, \
                                    Flatten, Dropout, LeakyReLU, LayerNormalization,\
                                    Reshape, Add
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np

from .feature import ext_tsf, gen_default_ts_feature_params, BlockedTsfMixer, MultiheadBlockedTsfMixer
from .mh3dr import Multihead3dRotation
from .utils import Tag


__default_ext_tsf_params = gen_default_ts_feature_params()

def r_tsf_net(x_shape,
           n_classes,
           out_loss,
           out_activ,
           extract_imu_tensor_func = None, # should return "tensor except imu_data, tensor[batch, time_ix, imu_num x feature_num x 3], imu_num, feature_num"
           activation_func = None,
           use_orig_input=True,
           imu_rot_num=2,
           learning_rate=0.01,
           ts_normalize = None,
           base_kn = 128,
           dropout_rate=0.5,
           depth=3,
           tsf_key_num = 128,
           tsf_mixer_depth=None,
           tsf_mixer_base_kn=128,
           block_mixer_depth = 2,
           block_mixer_base_kn = 32,
           rot_base_kn=128,
           rot_depth=3,
           rot_tsf_key_num = 128,
           rot_tsf_mixer_depth=1,
           rot_tsf_mixer_base_kn=128,
           rot_block_mixer_depth=3,
           rot_block_mixer_base_kn=64,
           rot_tsf_mixer_for_imu_depth=0,
           rot_tsf_mixer_for_imu_base_kn=32,
           no_rms = False,
           no_rms_for_rot = False,
           ax_weight_depth=1,
           ax_weight_base_kn=256,
           tsf_weight_depth=1,
           tsf_weight_base_kn=256,
           rot_ax_weight_depth=1,
           rot_ax_weight_base_kn=256,
           rot_tsf_weight_depth=1,
           rot_tsf_weight_base_kn=256,
           normalizer = None,
           rot_normalizer = None,
           regularization_rate=0.000001,
           residual_num = 0,
           tsf_block_sizes = [64],
           tsf_block_strides = [64],
           tsf_block_ext_tsf_params = [__default_ext_tsf_params],
           rot_tsf_block_sizes = [64],
           rot_tsf_block_strides = [64],
           rot_tsf_block_ext_tsf_params= [__default_ext_tsf_params],
           use_mh_attention = False,
           mha_head_num = 2,
           mha_key_dim = 32,
           few_norm = False,
           few_acti = False,
           tagging = False,
           rot_parallel_n = 1,
           rot_out_num = None,
           version = 1,
           concatenate_blocks_with_same_size_and_stride = False,
           use_add_for_blk_concatenate = False,
           btm_no_out_activation = False,
           btm_head_num = 1,
           metrics=['accuracy'],
           do_compile=True):
    """ Requires 3D data: [n_samples, n_timesteps, n_features]"""

    def get_activation():
        if activation_func is None:
            return LeakyReLU()
        else:
            return Lambda(activation_func)

    def get_normalizer():
        if normalizer is None:
            return LayerNormalization()
        else:
            return normalizer()

    if tsf_block_sizes is None:
        tsf_block_sizes = [x.shape[1]]

    if tsf_block_strides is None:
        tsf_block_strides = tsf_block_sizes

    _input_shape = x_shape[1:]

    inputs = Input(_input_shape)
    x = inputs

    if extract_imu_tensor_func is None: # TODO remove. for keeping the compatibility for old rot_tsf_mlps on main.py
        extract_imu_tensor_func = extract_imu_tensor_func_pamap2

    x_sensor_ids = None
    _extracted_features = extract_imu_tensor_func(x, version)
    if len(_extracted_features) == 4:
        x_except_imu, x_tags_except_imu, x_imu, x_tags = _extracted_features
    elif len(_extracted_features) == 5:
        x_except_imu, x_tags_except_imu, x_imu, x_tags, x_sensor_ids = _extracted_features
    else:
        raise ValueError("Invalid extracted feature number. Is not it implemented well yet?")

    imu_rotted = []
    imu_rms = []
    imu_rms_tags = []
    tags = []

    if imu_rot_num > 0:
        for _x_imu_list, _x_tags_list in zip(x_imu, x_tags):
            mh3dr = Multihead3dRotation(head_nums=imu_rot_num,
                                        depth=rot_depth,
                                        base_kn=rot_base_kn,
                                        dropout_rate=dropout_rate,
                                        tsf_mixer_depth = rot_tsf_mixer_depth,
                                        tsf_mixer_base_kn = rot_tsf_mixer_base_kn,
                                        block_mixer_depth = rot_block_mixer_depth,
                                        block_mixer_base_kn = rot_block_mixer_base_kn,
                                        normalizer = rot_normalizer,
                                        activation_func = activation_func,
                                        tsf_block_sizes= rot_tsf_block_sizes,
                                        tsf_block_strides = rot_tsf_block_strides,
                                        tsf_block_ext_tsf_params=rot_tsf_block_ext_tsf_params,
                                        ax_weight_depth = rot_ax_weight_depth,
                                        ax_weight_base_kn = rot_ax_weight_base_kn,
                                        tsf_weight_depth = rot_tsf_weight_depth,
                                        tsf_weight_base_kn = rot_tsf_weight_base_kn,
                                        few_norm=few_norm,
                                        few_acti=few_acti,
                                        tagging = tagging,
                                        parallel_n = rot_parallel_n,
                                        rot_out_num= rot_out_num,
                                        version=version,
                                        concatenate_blocks_with_same_size_and_stride = concatenate_blocks_with_same_size_and_stride,
                                        use_add_for_blk_concatenate = use_add_for_blk_concatenate,
                                        )

            for _x_imu, _x_tags in zip(_x_imu_list, _x_tags_list):
                if _x_imu.shape[2] % 3 != 0:
                    raise ValueError(f"{_x_imu.shape[2] % 3 =} should be 0: {_x_imu.shape=}")

                xx = tf.math.square(_x_imu, _x_imu)
                xx = Reshape((xx.shape[1], int(xx.shape[2]/3), 3))(xx)
                _rms = tf.sqrt(tf.reduce_mean(xx, 3))

                _rms_tag = np.copy(_x_tags[::3,:])
                _rms_tag[:,2] = Tag.L2

                imu_rms.append(_rms)
                imu_rms_tags.append(_rms_tag)

                if imu_rot_num > 0:

                    extended_imu = mh3dr([_x_imu,  None if no_rms_for_rot else _rms],
                                         [_x_tags, None if no_rms_for_rot else _rms_tag])
                    imu_rotted.extend(extended_imu)

                    for _ in range(len(extended_imu)):
                        tags.append(_x_tags)


    if not no_rms:
        imu_rotted.extend(imu_rms)
        tags.extend(imu_rms_tags)

    if use_orig_input:
        if version==1:
            imu_rotted.extend(x_imu)
            tags.extend(x_tags)
        elif version >= 2:
            for _imu, _tags in zip(x_imu, x_tags):
                imu_rotted.extend(_imu)
                tags.extend(_tags)

    if x_except_imu is not None:
        imu_rotted.extend(x_except_imu)
        tags.extend(x_tags_except_imu)

    if len(imu_rotted) == 1:
        x = imu_rotted[0]
        tags = tags[0]
    else:
        x = Concatenate(axis=2)(imu_rotted) # batch, time, sensor_axes
        tags = np.concatenate(tags)

    if ts_normalize is not None:
        x = ts_normalize(x)

    x_list = []
    _x_for_ext_tsf = x

    blk_params = {}
    for size, stride, ext_tsf_params in zip(tsf_block_sizes, tsf_block_strides, tsf_block_ext_tsf_params):
        if size not in blk_params:
            blk_params[size] = {}

        if stride not in blk_params[size]:
            blk_params[size][stride] = []

        blk_params[size][stride].append(ext_tsf_params)

    if concatenate_blocks_with_same_size_and_stride:
        def tsf_list_and_blk_id_generator():
            for block_id, blk_size in enumerate(blk_params):
                for blk_stride in blk_params[blk_size]:
                    block_id += 1

                    _tsf_list = []
                    for ext_tsf_params in blk_params[blk_size][blk_stride]:
                        _tsf_list.append(ext_tsf(_x_for_ext_tsf, block_size=blk_size, stride=blk_stride, ext_tsf_params=ext_tsf_params)) # [[b f tsf], .. ], len = len/block_size

                    tsf_list = []
                    for _i in range(len(_tsf_list[0])):
                        _blk_tsf_list = [_tsf[_i] for _tsf in _tsf_list]

                        if len(_blk_tsf_list) == 1:
                            tsf_list.append(_blk_tsf_list[0])
                        else:
                            tsf_list.append(Concatenate(axis=2)(_blk_tsf_list))

                    yield tsf_list, block_id

    else:
        def tsf_list_and_blk_id_generator():
            for size, stride, ext_tsf_params, block_id in zip(tsf_block_sizes, tsf_block_strides, tsf_block_ext_tsf_params, range(1, len(tsf_block_sizes)+1)):
                yield ext_tsf(_x_for_ext_tsf, block_size=size, stride=stride, ext_tsf_params=ext_tsf_params), block_id # [[b f tsf], .. ], len = len/block_size

    for tsf_list, block_id in tsf_list_and_blk_id_generator():
        if len(tsf_list) == 1:
            x = tf.expand_dims(tsf_list[0], 1)
        else:
            x = Concatenate(axis=1)([tf.expand_dims(tsf, 1) for tsf in tsf_list])

        if tagging:
            #_tags = np.copy(tags) # [axes num, [sensor_location_id, sensor type (e.g., ACC), axis (e.g., X)]]
            _tags = np.concatenate([tags, np.repeat(block_id, tags.shape[0]).reshape(tags.shape[0], 1)], axis=-1) # add block ids

            _ones = tf.ones_like(x)
            while _ones.shape[-1] < _tags.shape[-1]:
                _ones = tf.concat([_ones, _ones], axis=-1)
            #tags = tags.reshape((1,) + tags.shape)
            x = tf.concat([x, _ones[:,:,:,0:_tags.shape[-1]] * _tags.astype('float32')], axis=-1)

            # add sensor ids
            if x_sensor_ids is not None:
                # the shape of x_sensor_ids: batch, 1, 1
                assert x_sensor_ids.shape == (x.shape[0], 1, 1)
                # the shape of x: batch, blocks, sensor_axes, tsf
                _tmp = tf.expand_dims(x_sensor_ids, 3) # batch, 1, 1, 1
                _tmp = tf.repeat(_tmp, x.shape[1], axis=1) # batch, blocks, 1, 1
                _tmp = tf.repeat(_tmp, x.shape[2], axis=2) # batch, blocks, sensor_axes, 1
                x = tf.concat([x, _tmp], axis=-1)

        x = MultiheadBlockedTsfMixer(
                head_num = btm_head_num,
                tsf_mixer_depth=tsf_mixer_depth,
                tsf_mixer_base_kn=tsf_mixer_base_kn,
                block_mixer_depth=block_mixer_depth,
                block_mixer_base_kn=block_mixer_base_kn,
                ax_weight_depth=ax_weight_depth,
                ax_weight_base_kn=ax_weight_base_kn,
                tsf_weight_depth=tsf_weight_depth,
                tsf_weight_base_kn=tsf_weight_base_kn,
                normalizer=normalizer,
                activation_func=activation_func,
                dropout_rate=dropout_rate,
                few_norm=few_norm,
                few_acti=few_acti,
                version=version,
                no_out_activation = btm_no_out_activation,
                )(x)

        x_list.append(x)

    if use_add_for_blk_concatenate:
        x = Add()(x_list) if len(x_list) > 1 else x_list[0]
        x = get_normalizer()(x)
        x = get_activation()(x)
    else:
        x = Concatenate(axis=1)(x_list) if len(x_list) > 1 else x_list[0]

    x = Flatten()(x)

    # mixing all
    for d in range(depth-1, -1, -1):
        x = Dense(base_kn*(2**d))(x)
        if residual_num > 0:
            _x = x
            for _i in range(residual_num):
                x = get_normalizer()(x)
                x = get_activation()(x)
                x = Dense(base_kn*(2**d))(x)

            x = get_normalizer()(x)
            x = _x + x
            x = get_activation()(x)
        else:
            if not few_norm or d == depth-1:
                x = get_normalizer()(x)
            if not few_acti or d == 0:
                x = get_activation()(x)
        x = Dropout(dropout_rate)(x)

    x = Dense(n_classes, kernel_regularizer=l2(regularization_rate))(x)
    x = Activation(out_activ, dtype='float32')(x)
    m = Model(inputs, x)

    if do_compile:
        m.compile(loss=out_loss,
                  optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
                  weighted_metrics=metrics)

    return m

