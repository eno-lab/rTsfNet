from keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, \
        Permute, Concatenate, GlobalAveragePooling2D, Reshape, Flatten, Dense, Dropout,\
        TimeDistributed, LayerNormalization, UnitNormalization, Lambda, Add
from keras.models import Sequential
from keras.regularizers import l2
import keras
import numpy as np

from .feature import ext_tsf, gen_default_ts_feature_params, BlockedTsfMixer, MultiheadBlockedTsfMixer, _sqrt, divide_no_nan
from .utils import AbstructLayer

from keras.utils import register_keras_serializable

def _imu_rot(in_x, param, no_cos_limit=False):
    # quotanion version
    param = keras.ops.tanh(param)

    if not no_cos_limit:
        p0 = keras.ops.divide(keras.ops.add(param[:,0:1], 1), 2) # cos(theta/2) limited to 0-1, meaning theta is limited to 0 to pi. 
        param = keras.ops.concatenate([p0, param[:,1:4]], axis=-1) # cos limited to 0-1

    uq = keras.ops.normalize(param)
    uqSqSum = keras.ops.sum(keras.ops.square(uq), axis=-1, keepdims=True)
    isZero = keras.ops.where(uqSqSum  == 0., 1., 0.)
    isNotZero = keras.ops.subtract(1., isZero)
    w = keras.ops.concatenate([isZero, keras.ops.zeros_like(uq[:,1:4])], axis=-1)
    param = keras.ops.multiply(uq, isNotZero) # sum(sq(q1234)) == 0 to 0 
    param = keras.ops.add(uq, w) # sum(q1234) == 0 to 1, 0, 0, 0, meaning no rotation

    q_sq = keras.ops.square(uq)
    q_01 = keras.ops.multiply(uq[:,0:1], uq[:,1:2])
    q_02 = keras.ops.multiply(uq[:,0:1], uq[:,2:3])
    q_03 = keras.ops.multiply(uq[:,0:1], uq[:,3:4])
    q_12 = keras.ops.multiply(uq[:,1:2], uq[:,2:3])
    q_13 = keras.ops.multiply(uq[:,1:2], uq[:,3:4])
    q_23 = keras.ops.multiply(uq[:,2:3], uq[:,3:4])

    e = [None for _ in range(9)]
    e[0] = keras.ops.add(q_sq[:,0:1], q_sq[:,1:2])
    e[0] = keras.ops.subtract(e[0], q_sq[:,2:3])
    e[0] = keras.ops.subtract(e[0], q_sq[:,3:4])
    e[1] = keras.ops.multiply(2, keras.ops.subtract(q_12, q_03))
    e[2] = keras.ops.multiply(2, keras.ops.add(q_02, q_13))
    e[3] = keras.ops.multiply(2, keras.ops.add(q_03, q_12))
    e[4] = keras.ops.add(q_sq[:,0:1], q_sq[:,2:3])
    e[4] = keras.ops.subtract(e[4], q_sq[:,1:2])
    e[4] = keras.ops.subtract(e[4], q_sq[:,3:4])
    e[5] = keras.ops.multiply(2, keras.ops.subtract(q_23, q_01))
    e[6] = keras.ops.multiply(2, keras.ops.subtract(q_13, q_02))
    e[7] = keras.ops.multiply(2, keras.ops.add(q_01, q_23))
    e[8] = keras.ops.add(q_sq[:,0:1], q_sq[:,3:4])
    e[8] = keras.ops.subtract(e[8], q_sq[:,1:2])
    e[8] = keras.ops.subtract(e[8], q_sq[:,2:3])

    _tmp = keras.ops.concatenate(e, axis=-1)
    rM = keras.ops.reshape(_tmp, (-1,3,3))
    
    chunks = []

    for chunk in keras.ops.split(in_x, in_x.shape[-1]//3, axis=2):
        chunk = keras.ops.transpose(chunk, (0, 2, 1))
        chunk = keras.ops.matmul(rM, chunk)
        chunk = keras.ops.transpose(chunk, (0, 2, 1))
        chunks.append(chunk)
 
    return chunks[0] if len(chunks) == 1 else keras.ops.concatenate(chunks, axis=-1)


def _imu_rot_dmc(in_x, param, use_as_sin_value=False, no_axis_limit=False, no_theta_tanh=False):

    ####################################
    if not use_as_sin_value:
        un = keras.ops.normalize(param[:,0:3])
        if not no_axis_limit:
            pZ = keras.ops.divide(keras.ops.add(un[:,2:3], 1), 2)
            un = keras.ops.concatenate([un[:,0:2], pZ], axis=-1)

        _filter = keras.ops.sum(keras.ops.square(un), axis=-1, keepdims=True)
        theta = keras.ops.multiply(param[:,3:4], _filter) # if xyz=000, theta to 0
        if not no_theta_tanh:
            theta = keras.ops.tanh(theta) # これを考えるべき
        theta = keras.ops.multiply(theta, np.pi)
        # the cos/sin function do not work with this code, at least tf 2.17.1.
        sin = keras.ops.sin(theta) 
        cos = keras.ops.cos(theta) 
        ####################################
    else:
        ####################
        # sin version 
        un = keras.ops.normalize(param[:,0:3])
        _filter = keras.ops.sum(keras.ops.square(un), axis=-1, keepdims=True)
        sin = keras.ops.multiply(param[:,3:4], _filter) # if xyz=000, theta to 0
        sin = keras.ops.tanh(divide_no_nan(sin, _filter)) # limit sin from 0 to 1
        cos = _sqrt(keras.ops.subtract(1, keras.ops.square(sin))) # theta limited to -90 to 90
        ####################

    oneSubCos = keras.ops.subtract(1., cos)
    unSin = keras.ops.multiply(un, sin)
    unSq = keras.ops.multiply(keras.ops.square(un), oneSubCos)
    unXY = keras.ops.multiply(keras.ops.multiply(un[:,0:1], un[:,1:2]), oneSubCos)
    unYZ = keras.ops.multiply(keras.ops.multiply(un[:,1:2], un[:,2:3]), oneSubCos)
    unZX = keras.ops.multiply(keras.ops.multiply(un[:,2:3], un[:,0:1]), oneSubCos)

    e = []
    e.append(keras.ops.add(unSq[:,0:1], cos))
    e.append(keras.ops.subtract(unXY, unSin[:,2:3]))
    e.append(keras.ops.add(unZX, unSin[:,1:2]))
    e.append(keras.ops.add(unXY, unSin[:,2:3]))
    e.append(keras.ops.add(unSq[:,1:2], cos))
    e.append(keras.ops.subtract(unYZ, unSin[:,0:1]))
    e.append(keras.ops.subtract(unZX, unSin[:,1:2]))
    e.append(keras.ops.add(unYZ, unSin[:,0:1]))
    e.append(keras.ops.add(unSq[:,2:3], cos))

    _tmp = keras.ops.concatenate(e, axis=-1)
    rM = keras.ops.reshape(_tmp, (-1,3,3))
    
    chunks = []

    for chunk in keras.ops.split(in_x, in_x.shape[-1]//3, axis=2):
        chunk = keras.ops.transpose(chunk, (0, 2, 1))
        chunk = keras.ops.matmul(rM, chunk)
        chunk = keras.ops.transpose(chunk, (0, 2, 1))
        chunks.append(chunk)
 
    return chunks[0] if len(chunks) == 1 else keras.ops.concatenate(chunks, axis=-1)


def _imu_rot_old(in_x, param):
    """
        un: [n1, n2, n3], where -1 < n1,2,3 < 1 and L2(n) == 1
        theta: -1 < p < 1
    """
    __tfc_to_eye = keras.ops.convert_to_tensor(
             [1,0,0,0,1,0,0,0,1], dtype=keras.mixed_precision.global_policy().compute_dtype)
    __tfc_to_N = keras.ops.convert_to_tensor(
            [[0, 0,0,0,0,-1, 0,1,0],
             [0, 0,1,0,0, 0,-1,0,0],
             [0,-1,0,1,0, 0, 0,0,0]], dtype=keras.mixed_precision.global_policy().compute_dtype)
    __tfc_to_cNN1 = keras.ops.convert_to_tensor(
            [[1,1,1,0,0,0,0,0,0],
             [0,0,0,1,1,1,0,0,0],
             [0,0,0,0,0,0,1,1,1]], dtype=keras.mixed_precision.global_policy().compute_dtype)
    __tfc_to_cNN2 = keras.ops.convert_to_tensor(
            [[1,0,0,1,0,0,1,0,0],
             [0,1,0,0,1,0,0,1,0],
             [0,0,1,0,0,1,0,0,1]], dtype=keras.mixed_precision.global_policy().compute_dtype)

    w = keras.ops.where(keras.ops.sum(keras.ops.square(param[:,0:3]), axis=-1, keepdims=True) == 0., 1., 0.)
    w = keras.ops.concatenate([w, w, w], axis=-1)
    rot_axes = keras.ops.add(param[:,0:3], w) # xyz = 0, 0, 0 to 1, 1, 1
    rot_axes = param[:,0:3]

    #rot_axes_size = _sqrt(keras.ops.sum(keras.ops.square(rot_axes), axis=-1, keepdims=True))
    #un = divide_no_nan(rot_axes, rot_axes_size)
    un = keras.ops.normalize(rot_axes)
    #un = UnitNormalization()(rot_axes)
    sincos = keras.ops.tanh(param[:,3:5])
    sin = sincos[:,0:1] # direct estimation
    cos = sincos[:,1:2]

    c = cos 
    cEye = keras.ops.multiply(c, __tfc_to_eye)

    N = keras.ops.matmul(un, __tfc_to_N)

    sN = keras.ops.multiply(N, sin)

    cNN1 = keras.ops.matmul(un, __tfc_to_cNN1)
    cNN2 = keras.ops.matmul(un, __tfc_to_cNN2)

    NN = keras.ops.multiply(cNN1, cNN2)
    cNN = keras.ops.multiply(NN,  keras.ops.subtract(1., c))

    _tmp = keras.ops.add(cEye, cNN)
    _tmp = keras.ops.add(_tmp, sN)
    rM = keras.ops.reshape(_tmp, (-1,3,3))
    
    chunks = []

    for chunk in keras.ops.split(in_x, in_x.shape[-1]//3, axis=2):
        chunk = keras.ops.transpose(chunk, (0, 2, 1))
        chunk = keras.ops.matmul(rM, chunk)
        chunk = keras.ops.transpose(chunk, (0, 2, 1))
        chunks.append(chunk)
 
    #param = keras.ops.sum(keras.ops.square(param), axis=1, keepdims=True)
    
    #return keras.ops.add(in_x, keras.ops.expand_dims(param, 1))
    return chunks[0] if len(chunks) == 1 else keras.ops.concatenate(chunks, axis=-1)
    #return keras.ops.add(in_x, rotated)


@register_keras_serializable('tsf')
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
                 rot_kind = 0,
                 regularization_rate = 1e-5,
                 **kwargs):

        super(Multihead3dRotation, self).__init__(**kwargs)
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
        self.regularization_rate = regularization_rate

        self.rot_kind = rot_kind

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
                    regularization_rate = self.regularization_rate,
                    ) for _ in range(blk_count)]


        self.rot_param_gen_parallels = []
        for i in range(self.parallel_n):
            rot_param_gen_list = []
            self.rot_param_gen_parallels.append(rot_param_gen_list)
            for j in range(self.head_nums):
                if j > max(self.rot_out_num):
                    break

                s = Sequential(name=self._gen_name('sequence'))
                s.add(Flatten(name=self._gen_name('flatten')))
                for d in range(self.depth-1, -1, -1):
                    s.add(Dense(self.base_kn*(2**d), name=self._gen_name('dense')))
                    if not self.few_norm or d == self.depth-1:
                        s.add(self.get_normalizer())
                    if not self.few_acti or d == 0:
                        s.add(self.get_activation())
                    s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))

                # tanh seems better than liner, leakyReLU, ReLU, etc.
                #s.add(Dense(5, activation='tanh', kernel_regularizer=l2(1e-5), name=self._gen_name('dense'))) 
                s.add(Dense(4, activation='tanh', kernel_regularizer=l2(self.regularization_rate), name=self._gen_name('dense'))) 
                #s.add(Dense(4, activation='tanh', kernel_regularizer=l2(1e-5), name=self._gen_name('dense'))) 
                #s.add(Dense(4, kernel_regularizer=l2(1e-6), name=self._gen_name('dense'))) 
                #s.add(Dense(4, kernel_regularizer=l2(1e-5), name=self._gen_name('dense'))) 
                #s.add(Dense(4, activation='tanh', name=self._gen_name('dense'))) 
                #s.add(Dense(4, name=self._gen_name('dense'))) 
                #s.add(Dense(9, name=self._gen_name('dense')))  # いまいち
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
            x = keras.ops.concatenate([x, extra_x], axis=-1)
            x_tags = keras.ops.concatenate([x_tags, extra_x_tags])

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
                            tsf_list.append(keras.ops.concatenate(_blk_tsf_list, axis=2))

                        tsf_list_list.append(tsf_list)

        else:
            # simple
            for size, stride, ext_tsf_params in zip(self.tsf_block_sizes, self.tsf_block_strides, self.tsf_block_ext_tsf_params):
                tsf_list_list.append(ext_tsf(_x_for_ext_tsf, block_size=size, stride=stride, ext_tsf_params=ext_tsf_params))# [[b f tsf], .. ], len = len/block_size


        x_list = [] # feature after TSF mixer will be stored
        # handling TSFs
        for tsf_list, blocked_tsf_mixer in zip(tsf_list_list, self.blocked_tsf_mixer_list):

            if len(tsf_list) == 1:
                x = keras.ops.expand_dims(tsf_list[0], 1)
            else:
                x = keras.ops.concatenate([keras.ops.expand_dims(tsf, 1) for tsf in tsf_list], axis=1)

            if self.tagging:
                _ones = keras.ops.ones_like(x)
                while _ones.shape[-1] < x_tags.shape[-1]:
                    _ones = keras.ops.concatenate([_ones, _ones], axis=-1)
                x = keras.ops.concatenate([x, keras.ops.multiply(_ones[:,:,:,0:x_tags.shape[-1]], x_tags)], axis=-1)

            x = blocked_tsf_mixer(x)
            x_list.append(x)

        # select aggregation method
        if self.use_add_for_blk_concatenate:
            _x = None
            for __x in x_list:
                if _x == None:
                    _x = __x
                else:
                    _x = keras.ops.add(_x, __x)
            _x = self.add_norm_activ(_x)
        else:
            _x = keras.ops.concatenate(x_list, axis=1) if len(x_list) > 1 else x_list[0]

        # generate rotation parameters
        rot_params = []
        for rot_param_gen_list in self.rot_param_gen_parallels:
            prev_rot_params = None
            for i, rot_param_gen in enumerate(rot_param_gen_list):

                x = rot_param_gen(_x)

                if prev_rot_params is not None:
                    x = keras.ops.add(x, prev_rot_params)

                prev_rot_params = x

                # append export points
                if i in self.rot_out_num:
                    rot_params.append(x)

        # calc rotated x for each output points
        if self.rot_kind == 0:
            imu_rotted = [_imu_rot(in_x, param) for param in rot_params]
        elif self.rot_kind == 1:
            imu_rotted = [_imu_rot(in_x, param, no_cos_limit=True) for param in rot_params]
        elif self.rot_kind == 2:
            imu_rotted = [_imu_rot_dmc(in_x, param) for param in rot_params]
        elif self.rot_kind == 3:
            imu_rotted = [_imu_rot_dmc(in_x, param, no_axis_limit=True) for param in rot_params]
        elif self.rot_kind == 4:
            imu_rotted = [_imu_rot_dmc(in_x, param, use_as_sin_value=True) for param in rot_params]
        elif self.rot_kind == 5:
            imu_rotted = [_imu_rot_dmc(in_x, param, no_theta_tanh=True) for param in rot_params]
        elif self.rot_kind == 6:
            imu_rotted = [_imu_rot_dmc(in_x, param, no_axis_limit=True, no_theta_tanh=True) for param in rot_params]
        else:
            raise NotImplementedError(f'rot_kind limited from 0 to 4: {rot_kind=}')
        #imu_rotted = [_imu_rot_old(in_x, param) for param in rot_params]
        if self.return_list:
            return imu_rotted
        else:
            return imu_rotted[0] if len(imu_rotted) == 1 else keras.ops.concatenate(imu_rotted, axis=2)

