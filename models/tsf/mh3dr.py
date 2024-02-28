from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, \
        Permute, Concatenate, GlobalAveragePooling2D, Reshape, Flatten, Dense, Dropout,\
        TimeDistributed, LayerNormalization, UnitNormalization, Lambda, Add
from tensorflow.keras.models import Sequential
import math
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np

from .feature import ext_tsf, gen_default_ts_feature_params, BlockedTsfMixer, MultiheadBlockedTsfMixer
from .utils import AbstructLayer


__tfc_to_eye = None
__tfc_to_N = None
__tfc_to_cNN1 = None
__tfc_to_cNN2 = None


def _setup_constants():
    global __tfc_to_eye, __tfc_to_N, __tfc_to_cNN1, __tfc_to_cNN2

    if __tfc_to_eye is None:
        __tfc_to_eye = tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    if __tfc_to_N is None:
        __tfc_to_N = tf.constant([[0,0,0,0,0,-1,0,1,0],
                                  [0,0,1,0,0,0,-1,0,0],
                                  [0,-1,0,1,0,0,0,0,0]], dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    if __tfc_to_cNN1 is None:
        __tfc_to_cNN1 = tf.constant([[1,1,1,0,0,0,0,0,0],
                                     [0,0,0,1,1,1,0,0,0],
                                     [0,0,0,0,0,0,1,1,1]], dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    if __tfc_to_cNN2 is None:
        __tfc_to_cNN2 = tf.constant([[1,0,0,1,0,0,1,0,0],
                                     [0,1,0,0,1,0,0,1,0],
                                     [0,0,1,0,0,1,0,0,1]], dtype=tf.keras.mixed_precision.global_policy().compute_dtype)


__default_ext_tsf_params = gen_default_ts_feature_params()

def _imu_rot(in_x, param):
    """
        un: [n1, n2, n3], where -1 < n1,2,3 < 1 and L2(n) == 1
        theta: -1 < p < 1
    """

    _setup_constants()

    x = param
    # calculate rotation matrix
    un = UnitNormalization()(x[:,0:3])
    theta = tf.expand_dims(x[:,3], 1) * math.pi

    c = tf.cos(theta)
    cEye = c * __tfc_to_eye

    N = un @ __tfc_to_N

    sN = N * tf.sin(theta)

    cNN1 = un @ __tfc_to_cNN1
    cNN2 = un @ __tfc_to_cNN2

    NN = cNN1 * cNN2
    cNN = NN * (1. - c)

    rM = Reshape((3,3))(cEye + cNN + sN)
    
    p = Permute((2,1))
    chunks = []

    for chunk in tf.split(in_x, num_or_size_splits=in_x.shape[-1]//3, axis=2):
        chunks.append(p( rM @ p(chunk )))
    
    return chunks[0] if len(chunks) == 1 else  Concatenate()(chunks)


@tf.keras.saving.register_keras_serializable('tsf')
class Multihead3dRotation(AbstructLayer):

    def __init__(self,
                 head_nums, 
                 depth=2, 
                 base_kn=64, 
                 tsf_mixer_depth=3,
                 tsf_mixer_base_kn=64,
                 block_mixer_depth=3,
                 block_mixer_base_kn=64,
                 tsf_block_sizes=[64], 
                 tsf_block_strides=[64], 
                 tsf_block_ext_tsf_params = None, 
                 ax_weight_depth = 2,
                 ax_weight_base_kn = 64,
                 tsf_weight_depth = 2,
                 tsf_weight_base_kn = 32,
                 few_norm = False,
                 few_acti = False,
                 tagging = False,
                 return_list=True,
                 parallel_n = 1,
                 rot_out_num = None,
                 concatenate_blocks_with_same_size_and_stride = False,
                 use_add_for_blk_concatenate = False,
                 btm_head_num = 1,
                 btm_no_out_activation = False,
                 **kwargs):

        super(Multihead3dRotation, self).__init__(**kwargs)
        _setup_constants()
        self.head_nums = head_nums
        self.depth = depth
        self.base_kn = base_kn
        self.tsf_mixer_depth = tsf_mixer_depth
        self.tsf_mixer_base_kn = tsf_mixer_base_kn
        self.block_mixer_depth = block_mixer_depth
        self.block_mixer_base_kn = block_mixer_base_kn
        self.tsf_block_sizes = tsf_block_sizes
        self.tsf_block_strides = tsf_block_strides
        self.tsf_block_ext_tsf_params = tsf_block_ext_tsf_params
        self.few_norm = few_norm
        self.few_acti = few_acti
        self.tagging = tagging
        self.return_list = return_list
        self.parallel_n = parallel_n
        self.ax_weight_depth = ax_weight_depth
        self.ax_weight_base_kn = ax_weight_base_kn
        self.tsf_weight_depth = tsf_weight_depth
        self.tsf_weight_base_kn = tsf_weight_base_kn
        self.concatenate_blocks_with_same_size_and_stride = concatenate_blocks_with_same_size_and_stride
        self.use_add_for_blk_concatenate = use_add_for_blk_concatenate
        self.btm_head_num = btm_head_num
        self.btm_no_out_activation = btm_no_out_activation

        self.rot_out_num = rot_out_num if rot_out_num is not None else [i for i in range(head_nums)]
        if not np.in1d(self.rot_out_num, [i for i in range(head_nums)]).all():
            print(f'{self.rot_out_num=}')
            raise ValueError(f"Elements of rot_out_num should be in 0 and {head_nums}")


        if self.tsf_block_ext_tsf_params is None:
            def_params = gen_default_ts_feature_params()
            self.tsf_block_ext_tsf_params = [def_params for i in len(self.tsf_block_sizes)]

    def build(self, input_shape):
        x_input_shape = input_shape[0]

        if len(input_shape) > 1:
            extra_x_input_shape = input_shape[1]

        if self.tsf_block_sizes is None:
            self.tsf_block_sizes = [x_input_shape[-2]]

        if self.tsf_block_strides is None:
            self.tsf_block_strides = tsf_block_sizes

        total_c = x_input_shape[-1]
        if total_c % 3 != 0:
            raise ValueError("input_shape[-1] of x must be a multiple of 3.")


        if not self.concatenate_blocks_with_same_size_and_stride:
            blk_count = len(self.tsf_block_sizes)
        else:
            blk_count = len(self.tsf_block_ext_tsf_params)

        
        self.blocked_tsf_mixer_list = [
                MultiheadBlockedTsfMixer(
                    tsf_mixer_depth = self.tsf_mixer_depth,
                    tsf_mixer_base_kn = self.tsf_mixer_base_kn,
                    block_mixer_depth = self.block_mixer_depth,
                    block_mixer_base_kn = self.block_mixer_base_kn,
                    normalizer = self.normalizer, 
                    activation_func = self.activation_func,
                    dropout_rate = self.dropout_rate,
                    ax_weight_depth = self.ax_weight_depth,
                    ax_weight_base_kn = self.ax_weight_base_kn,
                    tsf_weight_depth = self.tsf_weight_depth,
                    tsf_weight_base_kn = self.tsf_weight_base_kn,
                    few_norm = self.few_norm,
                    few_acti = self.few_acti,
                    name=self._gen_name('multihead_block_tsf_mixer'),
                    version=self.version,
                    head_num = self.btm_head_num,
                    no_out_activation = self.btm_no_out_activation,
                    ) for _ in range(blk_count)]


        self.rot_param_gen_parallels = []
        for i in range(self.parallel_n):
            rot_param_gen_list = []
            self.rot_param_gen_parallels.append(rot_param_gen_list)
            for j in range(self.head_nums):
                s = Sequential(Flatten(name=self._gen_name('flatten')), name=self._gen_name('sequence'))
                for d in range(self.depth-1, -1, -1):
                    s.add(Dense(self.base_kn*(2**d), name=self._gen_name('dense')))
                    if not self.few_norm or d == self.depth-1:
                        s.add(self.get_normalizer())
                    if not self.few_acti or d == 0:
                        s.add(self.get_activation())
                    s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))

                # it seems better than liner, leakyReLU, ReLU, etc.
                s.add(Dense(4, activation='tanh', kernel_regularizer=l2(1e-5), name=self._gen_name('dense'))) 
                rot_param_gen_list.append(s)

        if self.use_add_for_blk_concatenate:
            self.add_norm_activ = Sequential([self.get_normalizer(), self.get_activation()])


    def get_config(self):
        config = super(Multihead3dRotation, self).get_config()
        config.update({'head_nums': self.head_nums})
        config.update({'depth': self.depth})
        config.update({'base_kn': self.base_kn})
        config.update({'tsf_mixer_depth': self.tsf_mixer_depth})
        config.update({'tsf_mixer_base_kn': self.tsf_mixer_base_kn})
        config.update({'block_mixer_depth': self.block_mixer_depth})
        config.update({'block_mixer_base_kn': self.block_mixer_base_kn})
        config.update({'tsf_block_sizes': self.tsf_block_sizes})
        config.update({'tsf_block_strides': self.tsf_block_strides})
        config.update({'tsf_block_ext_tsf_params': self.tsf_block_ext_tsf_params})
        config.update({'few_norm': self.few_norm})
        config.update({'few_acti': self.few_acti})
        config.update({'tagging': self.tagging})
        config.update({'return_list': self.return_list})
        config.update({'parallel_n': self.parallel_n})
        config.update({'rot_out_num': self.rot_out_num})
        config.update({'ax_weight_depth': self.ax_weight_depth})
        config.update({'ax_weight_base_kn': self.ax_weight_base_kn})
        config.update({'tsf_weight_depth': self.tsf_weight_depth})
        config.update({'tsf_weight_base_kn': self.tsf_weight_base_kn})
        config.update({'concatenate_blocks_with_same_size_and_stride': self.concatenate_blocks_with_same_size_and_stride})
        config.update({'use_add_for_blk_concatenate': self.use_add_for_blk_concatenate})
        config.update({'btm_head_num': self.btm_head_num})
        config.update({'btm_no_out_activation': self.btm_no_out_activation})
        return config

    def call(self, x, tags):
        """ 
        x: [x, extra_x] 
            x must be a shape: [batch, time_ix, features].
            extra_x is optional
        tags: [x_tags, extra_x_tags] (extra_x_tags is optional, is necessary if extra_x is provided)
        """

        assert 1 <= len(x) <= 2
        assert len(x) == len(tags)


        in_x = x[0]
        extra_x = None if len(x) == 1 else x[1]
        x_tags = tags[0]
        extra_x_tags = None if len(tags) == 1 else tags[1]

        if ((extra_x is not None and extra_x_tags is None)
           or (extra_x is None and extra_x_tags is not None)):
            raise ValueError("Both extra_x and extra_x_tags must be provided.")

        x = in_x
        if extra_x is not None:
            x = Concatenate()([x, extra_x])
            x_tags = np.concatenate([x_tags, extra_x_tags])

        _x_for_ext_tsf = x

        # extract TSF for each block setup
        tsf_list_list = []
        if self.concatenate_blocks_with_same_size_and_stride:
            # concatenate TSFs of each block position of each block size and stride
            blk_params = {}
            for size, stride, ext_tsf_params in zip(
                    self.tsf_block_sizes, 
                    self.tsf_block_strides, 
                    self.tsf_block_ext_tsf_params, 
                    ):

                if size not in blk_params:
                    blk_params[size] = {}

                if stride not in blk_params[size]:
                    blk_params[size][stride] = []

                blk_params[size][stride].append(ext_tsf_params)

            for blk_size in blk_params:
                for blk_stride in blk_params[blk_size]:

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

                        tsf_list_list.append(tsf_list)

        else:
            # simple
            for size, stride, ext_tsf_params in zip(self.tsf_block_sizes, self.tsf_block_strides, self.tsf_block_ext_tsf_params):
                tsf_list_list.append(ext_tsf(_x_for_ext_tsf, block_size=size, stride=stride, ext_tsf_params=ext_tsf_params))# [[b f tsf], .. ], len = len/block_size


        x_list = [] # feature after TSF mixer will be stored
        # handling TSFs
        for tsf_list, blocked_tsf_mixer in zip(tsf_list_list, self.blocked_tsf_mixer_list):

            if len(tsf_list) == 1:
                x = tf.expand_dims(tsf_list[0], 1)
            else:
                x = Concatenate(axis=1)([tf.expand_dims(tsf, 1) for tsf in tsf_list])

            if self.tagging:
                _ones = tf.ones_like(x)
                while _ones.shape[-1] < x_tags.shape[-1]:
                    _ones = tf.concat([_ones, _ones], axis=-1)
                x = Concatenate()([x, _ones[:,:,:,0:x_tags.shape[-1]] * x_tags.astype('float32')])

            x = blocked_tsf_mixer(x)
            x_list.append(x)

        # select aggregation method
        if self.use_add_for_blk_concatenate:
            _x = Add()(x_list) if len(x_list) > 1 else x_list[0]
            _x = self.add_norm_activ(_x)
        else:
            _x = Concatenate(axis=1)(x_list) if len(x_list) > 1 else x_list[0]

        # generate rotation parameters
        rot_params = []
        for rot_param_gen_list in self.rot_param_gen_parallels:
            _rot_params = []
            for i, rot_param_gen in enumerate(rot_param_gen_list):
                x = rot_param_gen(_x)

                if i > 0:
                    x = x + _rot_params[-1]
                _rot_params.append(x)

            # append export points
            for rn in self.rot_out_num:
                rot_params.append(_rot_params[rn])

        # calc rotated x for each output points
        imu_rotted = [_imu_rot(in_x, param) for param in rot_params]

        if self.return_list:
            return imu_rotted
        else:
            return imu_rotted[0] if len(imu_rotted) == 1 else Concatenate(axis=2)(imu_rotted)

