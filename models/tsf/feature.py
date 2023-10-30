from tensorflow.keras.layers import Layer, Conv1D, Conv2D, LeakyReLU, \
        Flatten, Dense, Dropout, TimeDistributed, LayerNormalization, \
        Multiply, Reshape, Lambda, Activation, MultiHeadAttention, \
        Add, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import math
import tensorflow as tf
import numpy as np

from .utils import AbstructLayer

def _ts_feature_list():
    return ['mean', 
            'min', 
            'max', 
            'rms', 
            'std', 
            'skewness',
            'kurtosis',
            'variance',
            'mean_change', 
            'sum_of_change', 
            'mean_abs_change', 
            'quantiles', 
            'tb_quantiles',
            'abs_energy', 
            'abs_max', 
            'abs_sum_of_changes', 
            'cid_ce', 
            'count_above_zero',
            'count_above_mean',
            'count_above_tb_start',
            'count_above_tb_25p',
            'count_above_tb_median',
            'count_above_tb_75p',
            'count_above_tb_end',
            'number_crossing_zero', 
            'number_crossing_mean', 
            'number_crossing_25p', 
            'number_crossing_median', 
            'number_crossing_75p', 
            'number_crossing_tb_start', 
            'number_crossing_tb_25p', 
            'number_crossing_tb_median', 
            'number_crossing_tb_75p', 
            'number_crossing_tb_end', 
            'fft_amp', 
            'fft_amp_agg_mean', 
            'fft_amp_agg_var', 
            'fft_amp_agg_skew', 
            'fft_amp_agg_kurt', 
            'fft_angle',
            'fft_amp_zero_hz', 
            'fft_amp_ratio', 
            'fft_amp_ratio_agg_mean', 
            'fft_amp_ratio_agg_var', 
            'fft_amp_ratio_agg_skew', 
            'fft_amp_ratio_agg_kurt', 
            'auto_corralation',
            'auto_corralation_raw'
            'auto_corralation_mean',
            'auto_corralation_var',
            'auto_corralation_skew',
            'auto_corralation_kurt',
            'auto_corralation_ratio_of_max'
            ]


def gen_default_ts_feature_params():
    params = {}
    for tsf in _ts_feature_list():
        params[tsf] = True

    params['auto_corralation'] = False
    return params

__default_ext_tsf_params = gen_default_ts_feature_params()

def ext_tsf(x, block_size=None, stride=None, return_list=True, ext_tsf_params=__default_ext_tsf_params):
    if len(x.shape) == 4:
        x = conv2d(1, (1,1))(x)
        x = tf.squeeze(x, 3)

    if block_size is None:
        block_size = x.shape[1]
        stride = block_size

    tsf = [] 
    for six in range(0, x.shape[1]-block_size+1, stride):
        tsf.append(ExtractTsFeatures(ext_tsf_params=ext_tsf_params)(x[:,six:(six+block_size),:])) # b l f tsf

    if return_list:
        return tsf

    else:
        return Concatenate()(tsf)
        


def autocorrelation(x, lags):
    """lags: max(lags) < x.shape[1]"""
    x_sub_means = sub_means(x)
    variance = tf.math.reduce_variance(x, 1)
    return _autocorrelation(x_sub_means, variance, lags)


def _autocorrelation(x_sub_means, variance, lags, var_min_limit=0.1):
    #x_ones = tf.ones_like(x_sub_means)
    #tf.reduce_max(tf.reduce_max(tf.reduce_sum(x_ones, axis=1), axis=0), axis=0) # shape=1
    x_total = tf.constant(x_sub_means.shape[1], dtype=tf.keras.mixed_precision.global_policy().compute_dtype)

    lags = np.asarray(lags)
    lags_max = np.max(lags)

    if len(x_sub_means.shape) == 3:
        x_sub_means = tf.expand_dims(x_sub_means, 3)

    if len(variance.shape) == 2:
        variance = tf.expand_dims(variance, 2)

    e = np.eye(lags_max+1)
    e = e[:,lags]
    if e.shape[0] > 2:
        padded_x_sub_means = tf.pad(x_sub_means, [[0,0],[0, e.shape[0]-1],[0,0],[0,0]], "CONSTANT")
    else:
        padded_x_sub_means = x_sub_means

    kn = tf.constant(e.reshape((e.shape[0], 1, 1, e.shape[1])), dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    v1 = tf.nn.conv2d(padded_x_sub_means, kn, strides=[1,1,1,1], padding='VALID')
    v2 = v1 * x_sub_means
    ###############
    #v3 = tf.reduce_mean(v2, axis=1) # cannot use reduce_mean because there are many pads whose value is zero.
    #v5 = tf.where(variance < var_min_limit, 0.0, v3 / variance)
    # with float, if variance < 1, the value going to bigger, so do not calculate <0.1
    ################
    ################
    #v3 = tf.reduce_mean(v2, axis=1)
    #v5 = tf.where(variance < 0.5, 0.0, v3 / variance)
    ## with float, if variance < 1, the value going to bigger, so do not calculate <0.5
    ################
    #v3 = tf.reduce_mean(v2, axis=1)
    #v5 = tf.where(variance == 0, variance, v3 / variance)
    ################
    v3 = tf.reduce_sum(v2, axis=1)
    tf_lags = tf.constant(lags, dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    v4 = v3 / (x_total - tf_lags)
    #v5 = tf.where(variance < var_min_limit, 0.0, v4 / variance)
    v5 = tf.math.divide_no_nan(v4, variance)
    #v5 = v4 / _add_small_value_for_zeros(variance)
    # with float, if variance < 1, the value going to bigger, so do not calculate <0.1
    ################
    #v3 = tf.reduce_sum(v2, axis=1)
    #tf_lags = tf.constant(lags, dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    #v4 = v3 / (x_total / tf_lags)
    #v5 = tf.math.divide_no_nan(v4, variance)
    ################
    ##v5 = v3 * tf_lags
    ##v5 = v3 * tf_lags / _add_small_value_for_zeros(variance)
    #v5 = tf.math.divide_no_nan(v3 * tf_lags , variance)
    return v5

def _magic_ac(x_sub_means, variance, lags, var_min_limit=0.1):
    x_total = tf.constant(x_sub_means.shape[1], dtype=tf.keras.mixed_precision.global_policy().compute_dtype)

    lags = np.asarray(lags)
    lags_max = np.max(lags)

    if len(x_sub_means.shape) == 3:
        x_sub_means = tf.expand_dims(x_sub_means, 3)

    if len(variance.shape) == 2:
        variance = tf.expand_dims(variance, 2)

    e = np.eye(lags_max+1)
    e = e[:,lags]
    if e.shape[0] > 2:
        padded_x_sub_means = tf.pad(x_sub_means, [[0,0],[0, e.shape[0]-1],[0,0],[0,0]], "CONSTANT")
    else:
        padded_x_sub_means = x_sub_means

    kn = tf.constant(e.reshape((e.shape[0], 1, 1, e.shape[1])), dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    v1 = tf.nn.conv2d(padded_x_sub_means, kn, strides=[1,1,1,1], padding='VALID')
    v2 = v1 * x_sub_means
    v3 = tf.reduce_sum(v2, axis=1)
    tf_lags = tf.constant(lags, dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    v4 = v3 / (x_total / tf_lags)
    v5 = tf.math.divide_no_nan(v4, variance)
    return v5

@tf.function
def diff(x):
    if len(x.shape) == 3:
        x = tf.expand_dims(x, 3)

    kernel = tf.constant([
        [[[0]]],
        [[[1]]],
        [[[-1]]]], dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    x = tf.nn.conv2d(x, kernel, strides=[1,1,1,1], padding='VALID')
    x = tf.squeeze(x, axis=3)
    return x


@tf.function
def sub_means(x, means=None):
    # x: b t f
    # means: b, f

    if means is None:
        means = tf.reduce_mean(x, 1, keepdims=True) 
    else:
        means = tf.expand_dims(means, 1)

    # means: b 1 f
    return x - means


@tf.function
def extract_quantiles(x, points=[0.25, 0.5, 0.75]):
    x = tf.transpose(x, perm=(0, 2, 1))
    x = tf.sort(x, axis=-1)

    eix = x.shape[2]
    return tf.gather(x, [round(p*eix) for p in points], axis=-1)


@tf.function
def extract_time_based_quantiles(x, points=[0, 0.25, 0.5, 0.75, 1]):
    x = tf.transpose(x, perm=(0, 2, 1))
    eix = x.shape[2]

    return tf.gather(x, [round(p*eix) for p in points], axis=-1)


@tf.function
def _sqrt(x): # tf.sqrt makes inf with 0. Why does tf take such implementation???
    # this implementation is better than tf.where(x == 0, 0, sqrt(x)) 
    # from viewpoint of gradient tape for the elements which are zero
    # probably.
    return tf.where(x == 0, 0.0, tf.sqrt(x))
    #return tf.sqrt(_add_small_value_for_zeros(x))

def _add_small_value_for_zeros(x, value=1e-12):
    e = tf.where(x == 0, value, 0.0)
    #e = (tf.ones_like(x) - tf.abs(tf.sign(x))) * value # put 1e-12 on points of zeros
    return x + e # add 1e-12 for zero

@tf.function
def skewness(x):
    mean = tf.math.reduce_mean(x, axis=1)
    std = tf.math.reduce_std(x, axis=1)
    x_sub_mean = x - mean
    n = x.shape[1]
    return _skewness(n, x_sub_mean, std)

@tf.function
def _skewness(n, x_sub_mean, std):
    # x_sub_mean: b, t, axis
    # std: b, 1, axis
    right = tf.math.divide_no_nan(x_sub_mean, std)
    return _skewness2(n, tf.math.reduce_sum(right**3, axis=1))

@tf.function
def _skewness2(n, sum_of_x_sub_mean_div_std_p3):
    alpha = n/(n-1)/(n-2)
    return alpha * sum_of_x_sub_mean_div_std_p3

@tf.function
def kurtosis(x):
    mean = tf.math.reduce_mean(x, axis=1)
    x_sub_mean = x - mean
    n = x.shape[1]
    return _kurtosis(n, x_sub_mean)

@tf.function
def _kurtosis(n, x_sub_mean):
    k4 = tf.math.reduce_sum(x_sub_mean**4, axis=1)
    k22 = tf.math.reduce_sum(x_sub_mean**2, axis=1)**2
    return _kurtosis2(n, k4, k22)

@tf.function
def _kurtosis2(n, k4, k22):
    # x_sub_mean: b, t, axis
    alpha = n*(n+1)*(n-1)/(n-2)/(n-3)
    right = 3*((n-1)**2)/(n-2)/(n-3)
    return alpha*tf.math.divide_no_nan(k4, k22)-right

def extract_ts_features(x, ext_tsf_params=__default_ext_tsf_params, version=0):
    """assuming the shape: batch, time, features"""

    def _is_group_enabled(key_prefix):
        return any([key.startswith(key_prefix) for key in ext_tsf_params])

    def _is_enabled(key):
        if key not in ext_tsf_params:
            return False

        val = ext_tsf_params[key]
        if type(val) == bool:
            return ext_tsf_params[key]
        
        return val is not None

    assert any([_is_enabled(val) for val in ext_tsf_params])

    x_sample_len = x.shape[1]

    need_expand_dims = []
    need_cast_expand_dims = []
    f_list = []
    f_list2 = []

    x_abs = tf.abs(x)
    xx = x**2
    x_diff = diff(x)
    x_diff_x_diff = x_diff**2
    x_diff_abs = tf.abs(x_diff)
    ###################################### 

    means = tf.reduce_mean(x, 1)
    if _is_enabled('mean'):
        need_cast_expand_dims.append(means)
    if _is_enabled('min'):
        f_list.append(tf.reduce_min(x, 1, keepdims=True))
    if _is_enabled('max'):
        f_list.append(tf.reduce_max(x, 1, keepdims=True))


    x_sub_means = sub_means(x, means) # b t f
    x_sub_means_sign = tf.sign(x_sub_means) # b t s

    if _is_enabled('rms'):
        f_list.append(_sqrt(tf.reduce_mean(xx, 1, keepdims=True)))  # root_mean_square
    if _is_enabled('ms'):
        f_list.append(tf.reduce_mean(xx, 1, keepdims=True))  # mean_square
    variance = tf.reduce_mean(x_sub_means**2, 1, keepdims=True)
    if _is_enabled('variance'):
        f_list.append(variance)
    standard_deviation = _sqrt(variance)
    if _is_enabled('std'):
        f_list.append(standard_deviation)

    if _is_enabled('skewness'):
        _skew = _skewness(x.shape[1], x_sub_means, standard_deviation)
        need_expand_dims.append(_skew)

    if _is_enabled('kurtosis'):
        _kurt = _kurtosis(x.shape[1], x_sub_means)
        need_expand_dims.append(_kurt)


    if _is_enabled('mean_change'):
        f_list.append(tf.reduce_mean(x_diff, 1, keepdims=True)) # mean_change
    if _is_enabled('sum_of_change'):
        f_list.append(tf.reduce_sum(x_diff, 1, keepdims=True))
    if _is_enabled('mean_abs_change'):
        f_list.append(tf.reduce_mean(x_diff_abs, 1, keepdims=True)) # mean_abs_change

    quantiles = extract_quantiles(x) # including 25%, median, 75%: b, f, 5
    if _is_enabled('quantiles'):
        f_list2.append(quantiles)

    tb_quantiles = extract_time_based_quantiles(x) # including the value of start, 25%, median, 75%, end of time: b, f, 5
    if _is_enabled('tb_quantiles'):
        f_list2.append(tb_quantiles)

    # f_list.append(tf.reduce_sum(x, 1, keepdims=True)) # sum_values, not needed. it means "mean times x_sample_len"

    if _is_enabled('abs_energy'):
        f_list.append(tf.math.reduce_sum(xx, 1, keepdims=True)) # abs_energy; removed; almost equals with 'rms'
    if _is_enabled('abs_max'):
        f_list.append(tf.reduce_max(x_abs, 1, keepdims=True)) # absolute_maximum
    if _is_enabled('abs_sum_of_changes'):
        f_list.append(tf.reduce_sum(x_diff_abs, 1, keepdims=True)) # absolute_sum_of_changes

    ###################################### 
    if _is_enabled('cid_ce'):
        f_list.append(_sqrt(tf.reduce_sum(x_diff_x_diff, 1, keepdims=True))) # cid_ce

    ###################################### 
    def count_above_zero(x_sub_val_sign):
        x = tf.math.count_nonzero(x_sub_val_sign -1, 1) # target values are marked as zero, others < 0
        return (x * -1) + x_sample_len # count zeros and calc diff from the time length

    x_sub_tb_val_sign = None
    x_sign = tf.sign(x)

    #if _is_enabled('sample_length'):
    #    f_list.append(tf.expand_dims(tf.ones_like(x)[:,0,:]*x_sample_len, 1)) # it will be consided as bias of FC

    if _is_enabled('count_above_zero'):
        need_cast_expand_dims.append(count_above_zero(x_sign))
    if _is_enabled('count_above_mean'):
        need_cast_expand_dims.append(count_above_zero(x_sub_means_sign))

    if _is_group_enabled("count_above_tb_"):
        if x_sub_tb_val_sign is None:
            x_sub_tb_val_sign =  [tf.sign(x - tf.expand_dims(tb_quantiles[:,:,i], 1)) for i in range(5)]

        if _is_enabled('count_above_tb_start'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[0]))
        if _is_enabled('count_above_tb_start_p'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[0])/x.shape[1])
        if _is_enabled('count_above_tb_25p'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[1]))
        if _is_enabled('count_above_tb_25p_p'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[1])/x.shape[1])
        if _is_enabled('count_above_tb_median'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[2]))
        if _is_enabled('count_above_tb_median_p'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[2])/x.shape[1])
        if _is_enabled('count_above_tb_75p'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[3]))
        if _is_enabled('count_above_tb_75p_p'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[3])/x.shape[1])
        if _is_enabled('count_above_tb_end'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[4]))
        if _is_enabled('count_above_tb_end_p'):
            need_cast_expand_dims.append(count_above_zero(x_sub_tb_val_sign[4])/x.shape[1])

    ###################################### 
    # crossings
    if _is_enabled('number_crossing_mean'):
        number_crossing_mean = tf.math.count_nonzero(diff(x_sub_means_sign), 1)
        need_cast_expand_dims.append(number_crossing_mean)
    if _is_enabled('number_crossing_zero'):
        number_crossing_zero = tf.math.count_nonzero(diff(tf.sign(x)), 1)
        need_cast_expand_dims.append(number_crossing_zero)
    if _is_enabled('number_crossing_25p'):
        number_crossing_25p = tf.math.count_nonzero(diff(tf.sign(x - tf.expand_dims(quantiles[:,:,0], 1))), 1)
        need_cast_expand_dims.append(number_crossing_25p)
    if _is_enabled('number_crossing_25p_p'):
        number_crossing_25p = tf.math.count_nonzero(diff(tf.sign(x - tf.expand_dims(quantiles[:,:,0], 1))), 1)
        need_cast_expand_dims.append(number_crossing_25p/x.shape[1]*2)
    if _is_enabled('number_crossing_median'):
        number_crossing_median = tf.math.count_nonzero(diff(tf.sign(x - tf.expand_dims(quantiles[:,:,1], 1))), 1)
        need_cast_expand_dims.append(number_crossing_median)
    if _is_enabled('number_crossing_75p'):
        number_crossing_75p = tf.math.count_nonzero(diff(tf.sign(x - tf.expand_dims(quantiles[:,:,2], 1))), 1)
        need_cast_expand_dims.append(number_crossing_75p)
    if _is_enabled('number_crossing_75p_p'):
        number_crossing_75p = tf.math.count_nonzero(diff(tf.sign(x - tf.expand_dims(quantiles[:,:,2], 1))), 1)
        need_cast_expand_dims.append(number_crossing_75p/x.shape[1]*2)

    if _is_group_enabled("number_crossing_tb_"):
        if x_sub_tb_val_sign is None:
            x_sub_tb_val_sign =  [tf.sign(x - tf.expand_dims(tb_quantiles[:,:,i], 1)) for i in range(5)]

        if _is_enabled('number_crossing_tb_start'):
            need_cast_expand_dims.append(tf.math.count_nonzero(diff(x_sub_tb_val_sign[0]), 1))
        if _is_enabled('number_crossing_tb_25p'):
            need_cast_expand_dims.append(tf.math.count_nonzero(diff(x_sub_tb_val_sign[1]), 1))
        if _is_enabled('number_crossing_tb_median'):
            need_cast_expand_dims.append(tf.math.count_nonzero(diff(x_sub_tb_val_sign[2]), 1))
        if _is_enabled('number_crossing_75p'):
            need_cast_expand_dims.append(tf.math.count_nonzero(diff(x_sub_tb_val_sign[3]), 1))
        if _is_enabled('number_crossing_tb_end'):
            need_cast_expand_dims.append(tf.math.count_nonzero(diff(x_sub_tb_val_sign[4]), 1))
    ###################################### 

    def freq_agg(freq, kind=('mean', 'var', 'std', 'skew', 'kurt')): # freq: b, f, freq 
        bin_ix = tf.ones_like(freq) * np.arange(freq.shape[2])
        freq_sum = tf.math.reduce_sum(freq, axis=2, keepdims=True)
        def _get_m(m):
            return tf.math.divide_no_nan(tf.reduce_sum((bin_ix ** m)*freq, axis=2, keepdims=True), freq_sum)

        m1 = None 
        m2 = None
        m3 = None
        m4 = None
        _var = None
       
        res = []
        if 'mean' in kind:
            m1 = _get_m(1)
            res.append(m1)

        if 'var' in kind:
            if m1 is None:
                m1 = _get_m(1)
            m2 = _get_m(2)
            _var = m2 - m1**2 # E[X^2]-(E[X])^2
            res.append(_var)

        if 'std' in kind:
            if _var is None:
                if m1 is None:
                    m1 = _get_m(1)
                m2 = _get_m(2)
                _var = m2 - m1**2 # E[X^2]-(E[X])^2
            res.append(_sqrt(_var))

        if 'skew' in kind:
            if m1 is None:
                m1 = _get_m(1)
            if _var is None:
                m2 = _get_m(2)
                _var = m2 - m1**2 # E[X^2]-(E[X])^2
            m3 = _get_m(3)
            _v15 = _add_small_value_for_zeros(_var ** 1.5)
            _skew = (m3 - 3*m1*_var - (m1**3)) / _v15
            #_skew = tf.math.divide_no_nan(m3 - 3*m1*_var - (m1**3),  _var**1.5)
            res.append(_skew)

        if 'kurt' in kind:
            if m1 is None:
                m1 = _get_m(1)
            if m2 is None:
                m2 = _get_m(2)
            if m3 is None:
                m3 = _get_m(3)
            if _var is None:
                _var = m2 - m1**2 # E[X^2]-(E[X])^2
            m4 = _get_m(4)
            _v2 = _add_small_value_for_zeros(_var ** 2)
            _kurt = (m4 - 4*m1*m3 + 6*m2*m1**2 - 3*m1 ) / _v2
            #_kurt = tf.math.divide_no_nan(m4 - 4*m1*m3 + 6*m2*m1**2 - 3*m1, _var**2)
            res.append(_kurt)

        return res

    if _is_group_enabled('fft_'):
        # TODO window function selection
        hamming_window = 0.54 + 0.46 * np.cos(2*np.pi*np.linspace(start=0, stop=1, num=x_sample_len))
        _x = tf.multiply(tf.transpose(x, perm=(0, 2, 1)), hamming_window)
        #fft_coef = tf.signal.fft(tf.complex(_x, tf.zeros_like(_x))) # b f freq
        #fft_coef = tf.signal.rfft(_x)[:,:,1:] # b f freq # remove 0hz
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            _x = tf.cast(_x, dtype='float32')
        fft_coef = tf.signal.rfft(_x) # b f freq # remove 0hz
        fft_coef_real = tf.math.real(fft_coef)
        fft_coef_imag = tf.math.imag(fft_coef)
        #f_list2.append(tf.math.abs(fft_coef)) # fft_amp: it includes sqrt in calculation. so, will it make inf
        fft_amp = _sqrt(fft_coef_real**2 + fft_coef_imag**2)

        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            fft_amp = tf.cast(fft_amp, dtype='float16')

        if _is_enabled('fft_amp'):
            f_list2.append(fft_amp) # fft_amp

        if _is_group_enabled('fft_amp_agg_'):
            _kind = []
            for k in ['mean', 'var', 'skew', 'kurt']:
                if _is_enabled(f'fft_amp_agg_{k}'):
                    _kind.append(k)
            f_list2.extend(freq_agg(fft_amp, _kind))

        if _is_enabled('fft_amp_zero_hz'):
            f_list2.append(tf.expand_dims(fft_amp[:,:,0],2)) # fft_amp

        if _is_group_enabled('fft_amp_ratio'):
            _fft_ratio = tf.math.divide_no_nan(fft_amp, tf.reduce_max(fft_amp, axis=-1, keepdims=True))

            if _is_enabled('fft_amp_ratio'):
                f_list2.append(_fft_ratio)

            if _is_group_enabled('fft_amp_ratio_agg_'):
                _kind = []
                for k in ['mean', 'var', 'skew', 'kurt']:
                    if _is_enabled(f'fft_amp_ratio_agg_{k}'):
                        _kind.append(k)
                f_list2.extend(freq_agg(_fft_ratio, _kind))

        if _is_enabled('fft_angle'):
            #fft_angle = tf.math.atan2(fft_coef_imag, _add_small_value_for_zeros(fft_coef_real))
            fft_angle = tf.math.atan2(fft_coef_imag, fft_coef_real)
            fft_angle = tf.where(fft_coef_real == 0, 0.0, fft_angle)
            #f_list2.append(tf.math.angle(fft_coef)) # fft_angle: it also make inf. 

            fft_angle = fft_angle + math.pi # conv -pi < x < pi --> 0 < x < 2pi

            _fft_amp = fft_amp
            _fft_amp = _fft_amp - tf.math.reduce_max(_fft_amp, keepdims=True) # max = 0, others < 0
            _fft_amp = tf.math.sign(tf.math.sign(_fft_amp) + 1) # max = 1, others = 0
            _angle_of_max_amp = fft_angle * _fft_amp # remain max fft
            fft_angle = fft_angle - _angle_of_max_amp # adjust the angle of max fft as zero

            def _plus_mask(x):
                s = tf.sign(x)
                return s*tf.sign(s+1) # sign(s+1): zero and plus = 1, others = 0 
            _mask_gt_pi = _plus_mask(fft_angle - math.pi)
            _mask_lt_m_pi = _plus_mask(fft_angle*-1 - math.pi)
            _mask_no_op = tf.ones_like(fft_angle) - _mask_gt_pi - _mask_lt_m_pi
            fft_angle = fft_angle * _mask_no_op + \
                        ((fft_angle - math.pi) * _mask_gt_pi) + \
                        ((fft_angle + math.pi) * _mask_lt_m_pi)

            f_list2.append(fft_angle) # fft_angle

        if version > 0:
            f_list.extend(agg(rfft_coef))

    if _is_enabled('magic_ac'):
        magic_ac = _magic_ac(x_sub_means, tf.transpose(variance, perm=(0,2,1)),ext_tsf_params['magic_ac'])
        f_list2.append(magic_ac)

    if _is_enabled('auto_corralation'):
        autocrr_lags  = ext_tsf_params['auto_corralation']
        autocrrs = _autocorrelation(x_sub_means, tf.transpose(variance, perm=(0,2,1)), autocrr_lags)
        if not _is_group_enabled('auto_corralation_'):
            f_list2.append(autocrrs)

        else:
            if not _is_enabled('auto_corralation_raw'):
                f_list2.append(autocrrs)

            ac_mean = tf.reduce_mean(autocrrs, axis=2, keepdims=True)
            ac_var = None
            if _is_enabled('auto_corralation_ratio_of_max'):
                f_list2.append(tf.math.divide_no_nan(autocrrs, tf.math.reduce_max(tf.abs(autocrrs), axis=2, keepdims=True)))

            if any([_is_enabled(f'auto_corralation_{_k}') for _k in ('var', 'std', 'skew', 'kurt')]):
                ac_sub_mean = autocrrs - ac_mean

            if _is_enabled('auto_corralation_mean'):
                f_list2.append(ac_mean)

            if any([_is_enabled(f'auto_corralation_{_k}') for _k in ('var', 'std', 'skew')]):
                ac_var = tf.math.reduce_mean(ac_sub_mean ** 2, axis=2, keepdims=True)
                if _is_enabled('auto_corralation_var'):
                    f_list2.append(ac_var)

            if any([_is_enabled(f'auto_corralation_{_k}') for _k in ('std', 'skew')]):
                ac_std = _sqrt(ac_var)
                if _is_enabled('auto_corralation_std'):
                    f_list2.append(ac_std)

            if _is_enabled('auto_corralation_skew'):
                _skew = _skewness(ac_sub_mean.shape[2], 
                        tf.transpose(ac_sub_mean, perm=(0,2,1)), 
                        tf.transpose(ac_std, perm=(0,2,1)))
                need_expand_dims.append(_skew)

            if _is_enabled('auto_corralation_kurt'):
                _kurt = _kurtosis(ac_sub_mean.shape[2],
                        tf.transpose(ac_sub_mean, perm=(0,2,1)))
                need_expand_dims.append(_kurt)

    ###################################### 
    for f in need_cast_expand_dims:
        f = tf.cast(f, tf.keras.mixed_precision.global_policy().compute_dtype)
        f = tf.expand_dims(f, 2)
        f_list2.append(f)

    #[for debug]
    #for f in f_list:
    #   print(f.shape)
    f1 = None
    if len(f_list) > 0:
        f1 = tf.concat(f_list, axis=1)
        f_list2.append(tf.transpose(f1, perm=(0,2,1)))

    if len(f_list2) == 1:
        return f_list2[0]
    else:
        return tf.concat(f_list2, axis=2)
    # not implemented yet

    # change_quantiles
    # agg_autocorrelation
    # agg_linear_trend
    # approximate_entropy
    # ar_coefficient
    # augmented_dickey_fuller
    # autocorrelation
    # benford_correlation
    # binned_entropy
    # c3
    # change_quantiles
    # count_above
    # count_below
    # cwt_coefficients
    # energy_ratio_by_chunks
    # fft_aggregated
    # first_location_of_maximum
    # first_location_of_minimum
    # fourier_entropy
    # friedrich_coefficients
    # has_duplicate
    # has_duplicate_max
    # has_duplicate_min
    # index_mass_quantile
    # kurtosis
    # large_standard_deviation
    # last_location_of_maximum
    # last_location_of_minimum
    # lempel_ziv_complexity
    # length
    # linear_trend
    # linear_trend_timewise
    # longest_strike_above_mean
    # longest_strike_below_mean
    # matrix_profile
    # max_langevin_fixed_point
    # mean_n_absolute_max
    # mean_second_derivative_central
    # minimum
    # number_cwt_peaks
    # number_peaks
    # partial_autocorrelation
    # percentage_of_reoccurring_datapoints_to_all_datapoints
    # percentage_of_reoccurring_values_to_all_values
    # permutation_entropy
    # query_similarity_count
    # range_count
    # ratio_beyond_r_sigma
    # ratio_value_number_to_time_series_length
    # root_mean_square
    # sample_entropy
    # set_property
    # spkt_welch_density
    # sum_of_reoccurring_data_points
    # sum_of_reoccurring_values
    # symmetry_looking
    # time_reversal_asymmetry_statistic
    # value_count
    # variance_larger_than_standard_deviation
    # variation_coefficient
    # 

@tf.keras.saving.register_keras_serializable('tsf')
class ExtractTsFeatures(Layer):

    def __init__(self, ext_tsf_params, **kwargs):
        super(ExtractTsFeatures, self).__init__(**kwargs)
        self.ext_tsf_params = ext_tsf_params

    def get_config(self):
        config = super(ExtractTsFeatures, self).get_config()
        config.update({"ext_tsf_params": self.ext_tsf_params})
        return config

    def call(self, x):
        return extract_ts_features(x, ext_tsf_params=self.ext_tsf_params)


@tf.keras.saving.register_keras_serializable('tsf')
class MultiheadBlockedTsfMixer(AbstructLayer):
    def __init__(self,
                 head_num = 1, 
                 tsf_mixer_depth = 2, 
                 tsf_mixer_base_kn = 32,
                 block_mixer_depth = 2,
                 block_mixer_base_kn = 32,
                 ax_weight_depth = 0,
                 ax_weight_base_kn = 16,
                 tsf_weight_depth = 0,
                 tsf_weight_base_kn = 16,
                 blk_weight_depth = 0,
                 blk_weight_base_kn = 16,
                 regularization_rate = 1e-5,
                 few_norm=False,
                 few_acti=False,
                 no_out_activation = False,
                 **kwargs):
        super(MultiheadBlockedTsfMixer, self).__init__(**kwargs)
        self.head_num = head_num
        self.tsf_mixer_depth = tsf_mixer_depth
        self.tsf_mixer_base_kn = tsf_mixer_base_kn
        self.block_mixer_depth = block_mixer_depth
        self.block_mixer_base_kn = block_mixer_base_kn
        self.ax_weight_depth = ax_weight_depth
        self.ax_weight_base_kn = ax_weight_base_kn
        self.tsf_weight_depth = tsf_weight_depth
        self.tsf_weight_base_kn = tsf_weight_base_kn
        self.blk_weight_depth = blk_weight_depth
        self.blk_weight_base_kn = blk_weight_base_kn
        self.regularization_rate = regularization_rate
        self.few_norm = few_norm
        self.few_acti = few_acti
        self.no_out_activation = no_out_activation
        if 'name' in kwargs:
            del kwargs['name']
        self.kwargs = kwargs

    def build(self, input_shape):
        self.btm_list = []
        for _ in range(self.head_num):
            btm = BlockedTsfMixer(
                    tsf_mixer_depth = self.tsf_mixer_depth,
                    tsf_mixer_base_kn = self.tsf_mixer_base_kn,
                    block_mixer_depth = self.block_mixer_depth,
                    block_mixer_base_kn = self.block_mixer_base_kn,
                    ax_weight_depth = self.ax_weight_depth,
                    ax_weight_base_kn = self.ax_weight_base_kn,
                    tsf_weight_depth = self.tsf_weight_depth,
                    tsf_weight_base_kn = self.tsf_weight_base_kn,
                    blk_weight_depth = self.blk_weight_depth,
                    blk_weight_base_kn = self.blk_weight_base_kn,
                    regularization_rate = self.regularization_rate,
                    few_norm = self.few_norm,
                    few_acti = self.few_acti,
                    no_out_activation = True,
                    name=self._gen_name('blocked_tsf_mixer'),
                    **self.kwargs)
            self.btm_list.append(Sequential([
                btm,
                LayerNormalization(name=self._gen_name('layer_nomalization')),
                ]))
        self.finisher_list = []
        for _i in range(self.head_num):
            s = Sequential()
            if _i > 0:
                s.add(Add(name=self._gen_name('add')))

            if not self.no_out_activation:
                if not self.few_norm:
                    s.add(self.get_normalizer())
                s.add(self.get_activation())

            s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))
            self.finisher_list.append(s)


    def get_config(self):
        config = super(MultiheadBlockedTsfMixer, self).get_config()
        config.update({'head_num': self.head_num})
        config.update({'tsf_mixer_depth': self.tsf_mixer_depth})
        config.update({'tsf_mixer_base_kn': self.tsf_mixer_base_kn})
        config.update({'block_mixer_depth': self.block_mixer_depth})
        config.update({'block_mixer_base_kn': self.block_mixer_base_kn})
        config.update({'ax_weight_depth': self.ax_weight_depth})
        config.update({'ax_weight_base_kn': self.ax_weight_base_kn})
        config.update({'tsf_weight_depth': self.tsf_weight_depth})
        config.update({'tsf_weight_base_kn': self.tsf_weight_base_kn})
        config.update({'blk_weight_depth': self.blk_weight_depth})
        config.update({'blk_weight_base_kn': self.blk_weight_base_kn})
        config.update({'regularization_rate': self.regularization_rate})
        config.update({'few_norm': self.few_norm})
        config.update({'few_acti': self.few_acti})
        config.update({'no_out_activation': self.no_out_activation})

        return config

    def call(self, inputs):
        results = [] 
        _l = [btm(inputs) for btm in self.btm_list]
        for i in range(len(_l)):
            if i == 0:
                results.append(self.finisher_list[i](_l[0]))
            else:
                results.append(self.finisher_list[i](_l[0:(i+1)]))

        return Concatenate()(results)

@tf.keras.saving.register_keras_serializable('tsf')
class BlockedTsfMixer(AbstructLayer):
    def __init__(self, 
                 tsf_mixer_depth = 2, 
                 tsf_mixer_base_kn = 32,
                 block_mixer_depth = 2,
                 block_mixer_base_kn = 32,
                 ax_weight_depth = 0,
                 ax_weight_base_kn = 16,
                 tsf_weight_depth = 0,
                 tsf_weight_base_kn = 16,
                 blk_weight_depth = 0,
                 blk_weight_base_kn = 16,
                 regularization_rate = 1e-5,
                 few_norm=False,
                 few_acti=False,
                 no_out_activation = False,
                 **kwargs):
        super(BlockedTsfMixer, self).__init__(**kwargs)
        self.tsf_mixer_depth = tsf_mixer_depth
        self.tsf_mixer_base_kn = tsf_mixer_base_kn
        self.block_mixer_depth = block_mixer_depth
        self.block_mixer_base_kn = block_mixer_base_kn
        self.ax_weight_depth = ax_weight_depth
        self.ax_weight_base_kn = ax_weight_base_kn
        self.tsf_weight_depth = tsf_weight_depth
        self.tsf_weight_base_kn = tsf_weight_base_kn
        self.blk_weight_depth = blk_weight_depth
        self.blk_weight_base_kn = blk_weight_base_kn
        self.regularization_rate = regularization_rate
        self.few_norm = few_norm
        self.few_acti = few_acti
        self.no_out_activation = no_out_activation

    def build(self, input_shape):
        assert len(input_shape) == 4
        """
        input_shape: must be [batch, block, axes, tsf]
        """

        def gen_mixer():
            s = Sequential(name=self._gen_name('sequential'))
            for i in range(self.tsf_mixer_depth-1, -1, -1):
                s.add(Conv2D(self.tsf_mixer_base_kn*(2**i), (1,1), name=self._gen_name('conv2d')))
                if not self.few_norm or i == self.tsf_mixer_depth-1:
                    s.add(self.get_normalizer())
                if not self.few_acti or i == 0:
                    s.add(self.get_activation())
                s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))
            return s

        # mixing on block- and feature wize 
        self.bf_wise_mixer = gen_mixer()

        self.bf_wise_mixer_ax_w = gen_mixer()
        self.bf_wise_mixer_tsf_w = gen_mixer()


        self.block_wise_mixers = {}
        for key in ['for_out', 'for_ax_weight', 'for_tsf_weight']:
            if key == 'for_ax_weight' and  self.ax_weight_depth <= 0:
                continue
            if key == 'for_tsf_weight' and  self.tsf_weight_depth <= 0:
                continue

            s = Sequential(name=self._gen_name('sequential'))
            # blockwise mixer
            for d in range(self.block_mixer_depth-1, -1, -1):
                if d == self.block_mixer_depth-1:
                    _kn = int(self.block_mixer_base_kn*(2**d))
                    s.add(Conv2D(_kn, (1, input_shape[2]), name=self._gen_name('conv2d')))
                    # batch, block, features, tsf
                    s.add(Reshape((input_shape[1], _kn), name=self._gen_name('reshape')))
                else:
                    s.add(Conv1D(self.block_mixer_base_kn*(2**d), 1, name=self._gen_name('conv1d')))

                if key == 'for_out' and self.no_out_activation and d == 0:
                    pass
                else:
                    if not self.few_norm or d == self.block_mixer_depth-1:
                        s.add(self.get_normalizer())
                    if not self.few_acti or d == 0:
                        s.add(self.get_activation())

                    s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))

            self.block_wise_mixers[key] = s

        def gen_weight(out_len, depth, base_kn):
            s = Sequential(name=self._gen_name('sequential'))
            # blockwise mixer
            for d in range(depth-1, -1, -1):
                s.add(Conv1D(base_kn*(2**d), 1, name=self._gen_name('conv1d')))

                if not self.few_norm or d == depth-1:
                    s.add(self.get_normalizer())
                if not self.few_acti or d == 0:
                    s.add(self.get_activation())
                s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))

            s.add(Conv1D(out_len, 1, kernel_regularizer=l2(self.regularization_rate), name=self._gen_name('conv1d')))
            s.add(Lambda(lambda x: x-tf.nn.relu(x*-1), name=self._gen_name('lambda')))
            s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))
            return s

        if self.ax_weight_depth > 0:
            self.gen_ax_weight = gen_weight(input_shape[2], self.ax_weight_depth, self.ax_weight_base_kn)

        if self.tsf_weight_depth > 0:
            self.gen_tsf_weight = gen_weight(self.tsf_mixer_base_kn, self.tsf_weight_depth, self.tsf_weight_base_kn)

        if self.blk_weight_depth > 0:
            self.gen_blk_weight = gen_weight(1, self.blk_weight_depth, self.blk_weight_base_kn)


    def call(self, inputs):
        _ax_w = None
        if self.ax_weight_depth > 0:
            _w_x_ax  = self.block_wise_mixers['for_ax_weight'](self.bf_wise_mixer_ax_w(inputs))
            _ax_w = tf.expand_dims(self.gen_ax_weight(_w_x_ax), axis=3) 

        _tsf_w = None
        if self.tsf_weight_depth > 0:
            _w_x_tsf = self.block_wise_mixers['for_tsf_weight'](self.bf_wise_mixer_tsf_w(inputs))
            _tsf_w = tf.expand_dims(self.gen_tsf_weight(_w_x_tsf), axis=2) # batch, block, sensor_axes, 1
  
        x = self.bf_wise_mixer(inputs)
        if _ax_w is not None:
            x = x * _ax_w
            x = tf.math.divide_no_nan(x, _ax_w)
        if _tsf_w is not None:
            x = x * _tsf_w
            x = tf.math.divide_no_nan(x, _tsf_w) # binary selection (sig/ sig) = 0 or 1

        x = self.block_wise_mixers['for_out'](x)
        if self.blk_weight_depth > 0:
            x = x * Reshape((x.shape[1], 1))(self.gen_blk_weight(x)) # batch, block, sensor_axes, 1

        return x

    def get_config(self):
        config = super(BlockedTsfMixer, self).get_config()
        config.update({'tsf_mixer_depth': self.tsf_mixer_depth})
        config.update({'tsf_mixer_base_kn': self.tsf_mixer_base_kn})
        config.update({'block_mixer_depth': self.block_mixer_depth})
        config.update({'block_mixer_base_kn': self.block_mixer_base_kn})
        config.update({'ax_weight_depth': self.ax_weight_depth})
        config.update({'ax_weight_base_kn': self.ax_weight_base_kn})
        config.update({'tsf_weight_depth': self.tsf_weight_depth})
        config.update({'tsf_weight_base_kn': self.tsf_weight_base_kn})
        config.update({'blk_weight_depth': self.blk_weight_depth})
        config.update({'blk_weight_base_kn': self.blk_weight_base_kn})
        config.update({'regularization_rate': self.regularization_rate})
        config.update({'few_norm': self.few_norm})
        config.update({'few_acti': self.few_acti})
        config.update({'no_out_activation': self.no_out_activation})
        return config

