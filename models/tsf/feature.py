from keras.layers import Layer, Conv1D, Conv2D, \
        Dropout, LayerNormalization, Reshape, Lambda, Add, Concatenate
from keras.models import Sequential
from keras.regularizers import l2
import keras
import numpy as np

from .utils import AbstructLayer, divide_no_nan
from keras.utils import register_keras_serializable


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
            'autocorralation',
            'autocorralation_raw'
            'autocorralation_mean',
            'autocorralation_var',
            'autocorralation_skew',
            'autocorralation_kurt',
            'autocorralation_ratio_of_max'
            ]


def gen_default_ts_feature_params():
    params = {}
    for tsf in _ts_feature_list():
        params[tsf] = True

    params['autocorralation'] = False
    return params

__default_ext_tsf_params = gen_default_ts_feature_params()

def ext_tsf(x, block_size=None, stride=None, return_list=True, ext_tsf_params=__default_ext_tsf_params):
    if len(x.shape) == 4:
        x = Conv2D(1, 1)(x)
        x = keras.ops.squeeze(x, 3)

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



# not used
#def autocorrelation(x, lags):
#    """lags: max(lags) < x.shape[1]"""
#    x_sub_means = sub_means(x)
#    variance = keras.ops.var(x, 1)
#    return _autocorrelation(x_sub_means, variance, lags)


def _autocorrelation(x_sub_means, variance, lags, var_min_limit=0.1):
    x_total = keras.ops.convert_to_tensor(x_sub_means.shape[1], dtype=keras.mixed_precision.global_policy().compute_dtype)

    lags_max = keras.ops.max(lags)

    if len(x_sub_means.shape) == 3:
        x_sub_means = keras.ops.expand_dims(x_sub_means, 3)

    if len(variance.shape) == 2:
        variance = keras.ops.expand_dims(variance, 2)

    e = keras.ops.eye(keras.ops.add(lags_max,1)) # not work with jax
    e = keras.ops.take(e, lags, axis=1)
    if e.shape[0] > 2:
        padded_x_sub_means = keras.ops.pad(x_sub_means, [[0,0],[0, e.shape[0]-1],[0,0],[0,0]], mode="constant", constant_values=0.0)
    else:
        padded_x_sub_means = x_sub_means

    e = keras.ops.expand_dims(e, 1)
    kn = keras.ops.expand_dims(e, 1)
    v1 = keras.ops.conv(padded_x_sub_means, kn, strides=1, padding='valid')
    v2 = keras.ops.multiply(v1, x_sub_means)
    v3 = keras.ops.sum(v2, axis=1)
    lags = keras.ops.cast(lags, dtype=keras.mixed_precision.global_policy().compute_dtype)
    v4 = divide_no_nan(v3, keras.ops.subtract(x_total, lags))
    v5 = divide_no_nan(v4, variance)

    return v5

def _magic_ac(x_sub_means, variance, lags, var_min_limit=0.1):
    x_total = keras.ops.convert_to_tensor(x_sub_means.shape[1], dtype=keras.mixed_precision.global_policy().compute_dtype)

    lags_max = keras.ops.max(lags)

    if len(x_sub_means.shape) == 3:
        x_sub_means = keras.ops.expand_dims(x_sub_means, 3)

    if len(variance.shape) == 2:
        variance = keras.ops.expand_dims(variance, 2)

    e = keras.ops.eye(keras.ops.add(lags_max,1))
    e = keras.ops.take(e, lags, axis=1)
    if e.shape[0] > 2:
        padded_x_sub_means = keras.ops.pad(x_sub_means, [[0,0],[0, e.shape[0]-1],[0,0],[0,0]], mode="constant", constant_values=0.0)
    else:
        padded_x_sub_means = x_sub_means

    e = keras.ops.expand_dims(e, 1)
    kn = keras.ops.expand_dims(e, 1)
    v1 = keras.ops.conv(padded_x_sub_means, kn, strides=1, padding='valid')
    v2 = keras.ops.multiply(v1, x_sub_means)
    v3 = keras.ops.sum(v2, axis=1)
    lags = keras.ops.cast(lags, dtype=keras.mixed_precision.global_policy().compute_dtype)
    v4 = divide_no_nan(v3, divide_no_nan(x_total, lags)) # Here
    v5 = divide_no_nan(v4, variance)

    return v5

def diff(x):
    if len(x.shape) == 3:
        x = keras.ops.expand_dims(x, 3)

    kernel = keras.ops.convert_to_tensor([
        [[[0]]],
        [[[1]]],
        [[[-1]]]], dtype=keras.mixed_precision.global_policy().compute_dtype)
    x = keras.ops.conv(x, kernel, strides=1, padding='valid')
    x = keras.ops.squeeze(x, axis=3)
    return x


def sub_means(x, means=None):
    # x: b t f
    # means: b, f

    if means is None:
        means = keras.ops.mean(x, 1, keepdims=True)
    else:
        means = keras.ops.expand_dims(means, 1)

    # means: b 1 f
    return keras.ops.subtract(x, means)


def extract_quantiles(x, points=[0.25, 0.5, 0.75]):
    x = keras.ops.transpose(x, (0, 2, 1))
    x = keras.ops.sort(x, axis=-1)

    eix = x.shape[2]-1
    return keras.ops.take(x, [round(p*eix) for p in points], axis=-1)


def extract_time_based_quantiles(x, points=[0, 0.25, 0.5, 0.75, 1]):
    x = keras.ops.transpose(x, (0, 2, 1))
    eix = x.shape[2]-1

    return keras.ops.take(x, [round(p*eix) for p in points], axis=-1)


def _sqrt(x): 
    # this implementation is better than tf.sqrt(_add_small_value_for_zeros(x))
    # from viewpoint of gradient tape for the elements which are zero
    # probably.

    cond = x == 0
    w = keras.ops.where(cond, 1., 0.)
    x = keras.ops.add(x, w)
    x = keras.ops.sqrt(x)
    return keras.ops.subtract(x, w)

#def _add_small_value_for_zeros(x, value=1e-12):
#    e = keras.ops.where(x == 0, value, 0.0)
#    #e = (keras.ops.ones_like(x) - keras.ops.abs(keras.ops.sign(x))) * value # put 1e-12 on points of zeros
#    return keras.ops.add(x, e) # add 1e-12 for zero

# not used
#def skewness(x):
#    mean = keras.ops.mean(x, axis=1)
#    std = keras.ops.std(x, axis=1)
#    x_sub_mean = x - mean
#    n = x.shape[1]
#    return _skewness(n, x_sub_mean, std)

def _skewness(n, x_sub_mean, std):
    #assert n >= 3
    # x_sub_mean: b, t, axis
    # std: b, 1, axis
    right = divide_no_nan(x_sub_mean, std)
    return _skewness2(n, keras.ops.sum(keras.ops.power(right, 3), axis=1))

def _skewness2(n, sum_of_x_sub_mean_div_std_p3):
    #assert n >= 3
    alpha = keras.ops.divide(n, n-1)
    alpha = keras.ops.divide(alpha, n-2)
    return keras.ops.multiply(alpha, sum_of_x_sub_mean_div_std_p3)

# not used
#def kurtosis(x):
#    mean = keras.ops.mean(x, axis=1)
#    x_sub_mean = x - mean
#    n = x.shape[1]
#    return _kurtosis(n, x_sub_mean)

def _kurtosis(n, x_sub_mean):
    #assert n >= 4
    k4 = keras.ops.sum(keras.ops.power(x_sub_mean, 4), axis=1)
    k22 = keras.ops.sum(keras.ops.power(x_sub_mean, 2), axis=1)
    k22 = keras.ops.power(k22, 2)
    return _kurtosis2(n, k4, k22)

def _kurtosis2(n, k4, k22):
    #assert n >= 4
    # x_sub_mean: b, t, axis
    alpha = keras.ops.multiply(n, n+1)
    alpha = keras.ops.multiply(alpha, n-1)
    alpha = keras.ops.divide(alpha, n-2)
    alpha = keras.ops.divide(alpha, n-3)
    #alpha = n*(n+1)*(n-1)/(n-2)/(n-3)
    right = keras.ops.power(n-1, 2)
    right = keras.ops.multiply(right, 3)
    right = keras.ops.divide(right, n-2)
    right = keras.ops.divide(right, n-3)
    #right = 3*((n-1)**2)/(n-2)/(n-3)
    tmp = keras.ops.multiply(alpha, divide_no_nan(k4, k22))
    return keras.ops.subtract(tmp, right)

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

    #assert any([_is_enabled(val) for val in ext_tsf_params])

    x_batch_n, x_sample_len, x_featur_n = x.shape

    f_list = []

    x_abs = keras.ops.abs(x)
    xx = keras.ops.power(x, 2)
    x_diff = diff(x)
    x_diff_x_diff = keras.ops.power(x_diff, 2)
    x_diff_abs = keras.ops.abs(x_diff)
    ######################################

    means = keras.ops.mean(x, 1)
    if _is_enabled('mean'):
        f_list.append(means)
    if _is_enabled('min'):
        f_list.append(keras.ops.min(x, 1, keepdims=True))
    if _is_enabled('max'):
        f_list.append(keras.ops.max(x, 1, keepdims=True))


    x_sub_means = sub_means(x, means) # b t f
    x_sub_means_sign = keras.ops.sign(x_sub_means) # b t s

    if _is_enabled('rms'):
        f_list.append(_sqrt(keras.ops.mean(xx, 1, keepdims=True)))  # root_mean_square
    if _is_enabled('ms'):
        f_list.append(keras.ops.mean(xx, 1, keepdims=True))  # mean_square
    variance = keras.ops.mean(keras.ops.power(x_sub_means,2), 1, keepdims=True)
    if _is_enabled('variance'):
        f_list.append(variance)
    standard_deviation = _sqrt(variance)
    if _is_enabled('std'):
        f_list.append(standard_deviation)

    if _is_enabled('skewness'):
        _skew = _skewness(x.shape[1], x_sub_means, standard_deviation)
        f_list.append(_skew)

    if _is_enabled('kurtosis'):
        _kurt = _kurtosis(x.shape[1], x_sub_means)
        f_list.append(_kurt)


    if _is_enabled('mean_change'):
        f_list.append(keras.ops.mean(x_diff, 1, keepdims=True)) # mean_change
    if _is_enabled('sum_of_change'):
        f_list.append(keras.ops.sum(x_diff, 1, keepdims=True))
    if _is_enabled('mean_abs_change'):
        f_list.append(keras.ops.mean(x_diff_abs, 1, keepdims=True)) # mean_abs_change

    quantiles = None
    if _is_enabled('quantiles') or _is_group_enabled('number_crossing_'):
        quantiles = extract_quantiles(x) # including 25%, median, 75%: b, f, 5
        if _is_enabled('quantiles'):
            f_list.append(quantiles)

    tb_quantiles = None
    if _is_enabled('tb_quantiles') or _is_group_enabled("count_above_tb_") or _is_group_enabled("number_crossing_tb_"):
        tb_quantiles = extract_time_based_quantiles(x) # including the value of start, 25%, median, 75%, end of time: b, f, 5
        if _is_enabled('tb_quantiles'):
            f_list.append(tb_quantiles)

    # f_list.append(keras.ops.sum(x, 1, keepdims=True)) # sum_values, not needed. it means "mean times x_sample_len"

    if _is_enabled('abs_energy'):
        f_list.append(keras.ops.sum(xx, 1, keepdims=True)) # abs_energy; removed; almost equals with 'rms'
    if _is_enabled('abs_max'):
        f_list.append(keras.ops.max(x_abs, 1, keepdims=True)) # absolute_maximum
    if _is_enabled('abs_sum_of_changes'):
        f_list.append(keras.ops.sum(x_diff_abs, 1, keepdims=True)) # absolute_sum_of_changes

    ######################################
    if _is_enabled('cid_ce'):
        f_list.append(_sqrt(keras.ops.sum(x_diff_x_diff, 1, keepdims=True))) # cid_ce

    ######################################
    def count_above_zero(x_sub_val_sign):
        x = keras.ops.count_nonzero(x_sub_val_sign -1, 1) # target values are marked as zero, others < 0
        return keras.ops.add(keras.ops.multiply(x, -1), x_sample_len)
        #return (x * -1) + x_sample_len # count zeros and calc diff from the time length

    x_sub_tb_val_sign = None
    x_sign = keras.ops.sign(x)

    #if _is_enabled('sample_length'):
    #    f_list.append(keras.ops.expand_dims(keras.ops.ones_like(x)[:,0,:]*x_sample_len, 1)) # it will be consided as bias of FC

    if _is_enabled('count_above_zero'):
        f_list.append(count_above_zero(x_sign))
    if _is_enabled('count_above_mean'):
        f_list.append(count_above_zero(x_sub_means_sign))

    if _is_group_enabled("count_above_tb_"):
        if x_sub_tb_val_sign is None:
            x_sub_tb_val_sign =  [keras.ops.sign(keras.ops.subtract(x, keras.ops.expand_dims(tb_quantiles[:,:,i], 1))) for i in range(5)]

        if _is_enabled('count_above_tb_start'):
            f_list.append(count_above_zero(x_sub_tb_val_sign[0]))
        if _is_enabled('count_above_tb_start_p'): # p means percentage
            f_list.append(count_above_zero(keras.ops.divide(x_sub_tb_val_sign[0],x.shape[1])))
        if _is_enabled('count_above_tb_25p'):
            f_list.append(count_above_zero(x_sub_tb_val_sign[1]))
        if _is_enabled('count_above_tb_25p_p'):
            f_list.append(count_above_zero(keras.ops.divide(x_sub_tb_val_sign[1],x.shape[1])))
        if _is_enabled('count_above_tb_median'):
            f_list.append(count_above_zero(x_sub_tb_val_sign[2]))
        if _is_enabled('count_above_tb_median_p'):
            f_list.append(count_above_zero(keras.ops.divide(x_sub_tb_val_sign[2],x.shape[1])))
        if _is_enabled('count_above_tb_75p'):
            f_list.append(count_above_zero(x_sub_tb_val_sign[3]))
        if _is_enabled('count_above_tb_75p_p'):
            f_list.append(count_above_zero(keras.ops.divide(x_sub_tb_val_sign[3],x.shape[1])))
        if _is_enabled('count_above_tb_end'):
            f_list.append(count_above_zero(x_sub_tb_val_sign[4]))
        if _is_enabled('count_above_tb_end_p'):
            f_list.append(count_above_zero(keras.ops.divide(x_sub_tb_val_sign[4],x.shape[1])))

    ######################################
    # crossings
    if _is_enabled('number_crossing_mean'):
        number_crossing_mean = keras.ops.count_nonzero(diff(x_sub_means_sign), 1)
        f_list.append(number_crossing_mean)
    if _is_enabled('number_crossing_zero'):
        number_crossing_zero = keras.ops.count_nonzero(diff(keras.ops.sign(x)), 1)
        f_list.append(number_crossing_zero)
    if _is_enabled('number_crossing_25p'):
        number_crossing_25p = keras.ops.count_nonzero(diff(keras.ops.sign(keras.ops.subtract(x, keras.ops.expand_dims(quantiles[:,:,0], 1)))), 1)
        f_list.append(number_crossing_25p)
    if _is_enabled('number_crossing_25p_p'):
        number_crossing_25p = keras.ops.count_nonzero(diff(keras.ops.sign(keras.ops.subtract(x, keras.ops.expand_dims(quantiles[:,:,0], 1)))), 1)
        f_list.append(keras.ops.divide(number_crossing_25p, x.shape[1]*2))
    if _is_enabled('number_crossing_median'):
        number_crossing_median = keras.ops.count_nonzero(diff(keras.ops.sign(keras.ops.subtract(x, keras.ops.expand_dims(quantiles[:,:,1], 1)))), 1)
        f_list.append(number_crossing_median)
    if _is_enabled('number_crossing_75p'):
        number_crossing_75p = keras.ops.count_nonzero(diff(keras.ops.sign(keras.ops.subtract(x, keras.ops.expand_dims(quantiles[:,:,2], 1)))), 1)
        f_list.append(number_crossing_75p)
    if _is_enabled('number_crossing_75p_p'):
        number_crossing_75p = keras.ops.count_nonzero(diff(keras.ops.sign(keras.ops.subtract(x, keras.ops.expand_dims(quantiles[:,:,2], 1)))), 1)
        f_list.append(keras.ops.divide(number_crossing_75p,x.shape[1]*2))

    if _is_group_enabled("number_crossing_tb_"):
        if x_sub_tb_val_sign is None:
            x_sub_tb_val_sign =  [keras.ops.sign(keras.ops.subtract(x, keras.ops.expand_dims(tb_quantiles[:,:,i], 1))) for i in range(5)]

        if _is_enabled('number_crossing_tb_start'):
            f_list.append(keras.ops.count_nonzero(diff(x_sub_tb_val_sign[0]), 1))
        if _is_enabled('number_crossing_tb_25p'):
            f_list.append(keras.ops.count_nonzero(diff(x_sub_tb_val_sign[1]), 1))
        if _is_enabled('number_crossing_tb_median'):
            f_list.append(keras.ops.count_nonzero(diff(x_sub_tb_val_sign[2]), 1))
        if _is_enabled('number_crossing_tb_75p'):
            f_list.append(keras.ops.count_nonzero(diff(x_sub_tb_val_sign[3]), 1))
        if _is_enabled('number_crossing_tb_end'):
            f_list.append(keras.ops.count_nonzero(diff(x_sub_tb_val_sign[4]), 1))
    ######################################

    def freq_agg(freq, kind=('mean', 'var', 'std', 'skew', 'kurt')): # freq: b, f, freq
        bin_ix = keras.ops.multiply(keras.ops.ones_like(freq), keras.ops.arange(freq.shape[2], dtype=keras.mixed_precision.global_policy().compute_dtype))
        freq_sum = keras.ops.sum(freq, axis=2, keepdims=True)
        def _get_m(m):
            return divide_no_nan(keras.ops.sum(keras.ops.multiply(keras.ops.power(bin_ix, m), freq), axis=2, keepdims=True), freq_sum)

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
            _var = keras.ops.subtract(m2, keras.ops.power(m1, 2)) # E[X^2]-(E[X])^2
            res.append(_var)

        if 'std' in kind:
            if _var is None:
                if m1 is None:
                    m1 = _get_m(1)
                m2 = _get_m(2)
                _var = keras.ops.subtract(m2, keras.ops.power(m1, 2)) # E[X^2]-(E[X])^2
            res.append(_sqrt(_var))

        if 'skew' in kind:
            if m1 is None:
                m1 = _get_m(1)
            if _var is None:
                m2 = _get_m(2)
                _var = keras.ops.subtract(m2, keras.ops.power(m1, 2)) # E[X^2]-(E[X])^2
            m3 = _get_m(3)
            _v15 = keras.ops.power(_var, 1.5)
            #_skew = (m3 - 3*m1*_var - (m1**3)) / _v15
            _skew = keras.ops.multiply(3, keras.ops.multiply(m1, _var))
            _skew = keras.ops.subtract(m3, _skew)
            _skew = keras.ops.subtract(m3, keras.ops.power(m1, 3))
            _skew = divide_no_nan(_skew, _v15)
            res.append(_skew)

        if 'kurt' in kind:
            if m1 is None:
                m1 = _get_m(1)
            if m2 is None:
                m2 = _get_m(2)
            if m3 is None:
                m3 = _get_m(3)
            if _var is None:
                _var = keras.ops.subtract(m2, keras.ops.power(m1, 2)) # E[X^2]-(E[X])^2
            m4 = _get_m(4)
            _v2 = keras.ops.power(_var, 2)
            #_kurt = (m4 - 4*m1*m3 + 6*m2*m1**2 - 3*m1 ) / _v2
            _tmp = keras.ops.multiply(4, keras.ops.multiply(m1, m3))
            _kurt = keras.ops.subtract(m4, _tmp)
            _tmp = keras.ops.multiply(6, keras.ops.multiply(m2, keras.ops.power(m1, 2)))
            _kurt = keras.ops.add(_kurt, _tmp)
            _kurt = keras.ops.subtract(_kurt, keras.ops.multiply(3, m1))
            _kurt = divide_no_nan(_kurt, _v2)
            #_kurt = keras.ops.divide_no_nan(m4 - 4*m1*m3 + 6*m2*m1**2 - 3*m1, _var**2)
            res.append(_kurt)

        return res

    if _is_group_enabled('fft_'):
        # TODO window function selection
        hamming_window = 0.54 + 0.46 * np.cos(2*np.pi*np.linspace(start=0, stop=1, num=x_sample_len))
        hamming_window = keras.ops.convert_to_tensor(hamming_window, dtype=keras.mixed_precision.global_policy().compute_dtype)
        _x = keras.ops.multiply(keras.ops.transpose(x, (0, 2, 1)), hamming_window)
        #fft_coef = keras.ops.fft(tf.complex(_x, tf.zeros_like(_x))) # b f freq
        #fft_coef = keras.ops.rfft(_x)[:,:,1:] # b f freq # remove 0hz
        if keras.mixed_precision.global_policy().compute_dtype == 'float16':
            _x = keras.ops.cast(_x, dtype='float32')
        fft_coef_real, fft_coef_imag = keras.ops.rfft(_x) # b f freq # remove 0hz
        #fft_coef = keras.ops.rfft(_x) # b f freq # remove 0hz
        #fft_coef_real = keras.ops.real(fft_coef)
        #fft_coef_imag = keras.ops.imag(fft_coef)
        #f_list.append(tf.math.abs(fft_coef)) # fft_amp: it includes sqrt in calculation. so, will it make inf
        fft_amp = _sqrt(keras.ops.add(keras.ops.power(fft_coef_real, 2), keras.ops.power(fft_coef_imag, 2)))

        if keras.mixed_precision.global_policy().compute_dtype == 'float16':
            fft_amp = keras.ops.cast(fft_amp, dtype='float16')

        if _is_enabled('fft_amp'):
            f_list.append(fft_amp) # fft_amp

        if _is_group_enabled('fft_amp_agg_'):
            _kind = []
            for k in ['mean', 'var', 'skew', 'kurt']:
                if _is_enabled(f'fft_amp_agg_{k}'):
                    _kind.append(k)
            f_list.extend(freq_agg(fft_amp, _kind))

        if _is_enabled('fft_amp_zero_hz'):
            f_list.append(keras.ops.expand_dims(fft_amp[:,:,0],2)) # fft_amp

        if _is_group_enabled('fft_amp_ratio'):
            _fft_ratio = divide_no_nan(fft_amp, keras.ops.max(fft_amp, axis=-1, keepdims=True))

            if _is_enabled('fft_amp_ratio'):
                f_list.append(_fft_ratio)

            if _is_group_enabled('fft_amp_ratio_agg_'):
                _kind = []
                for k in ['mean', 'var', 'skew', 'kurt']:
                    if _is_enabled(f'fft_amp_ratio_agg_{k}'):
                        _kind.append(k)
                f_list.extend(freq_agg(_fft_ratio, _kind))

        if _is_enabled('fft_angle'):
            #fft_angle = tf.math.atan2(fft_coef_imag, _add_small_value_for_zeros(fft_coef_real))
            fft_angle = keras.ops.arctan2(fft_coef_imag, fft_coef_real)
            fft_angle = keras.ops.where(fft_coef_real == 0, 0.0, fft_angle)
            #f_list.append(tf.math.angle(fft_coef)) # fft_angle: it also make inf.

            fft_angle = keras.ops.add(fft_angle, np.pi) # conv -pi < x < pi --> 0 < x < 2pi

            _fft_amp = fft_amp
            _fft_amp = keras.ops.subtract(_fft_amp, keras.ops.max(_fft_amp, keepdims=True)) # max = 0, others < 0
            _fft_amp = keras.ops.sign(keras.ops.add(keras.ops.sign(_fft_amp), 1)) # max = 1, others = 0
            _angle_of_max_amp = keras.ops.multiply(fft_angle, _fft_amp) # remain max fft
            fft_angle = keras.ops.subtract(fft_angle, _angle_of_max_amp) # adjust the angle of max fft as zero

            def _plus_mask(x):
                s = keras.ops.sign(x)
                return keras.ops.multiply(s, keras.ops.sign(keras.ops.add(s, 1))) # sign(s+1): zero and plus = 1, others = 0
            _mask_gt_pi = _plus_mask(keras.ops.subtract(fft_angle, np.pi))
            _mask_lt_m_pi = _plus_mask(keras.ops.subtract(keras.ops.multiply(fft_angle, -1), np.pi))
            _mask_no_op = keras.ops.subtract(keras.ops.subtract(keras.ops.ones_like(fft_angle), _mask_gt_pi), _mask_lt_m_pi)
            _tmp1 = keras.ops.multiply(fft_angle, _mask_no_op)
            _tmp2 = keras.ops.multiply(keras.ops.subtract(fft_angle, np.pi), _mask_gt_pi)
            _tmp3 = keras.ops.multiply(keras.ops.add(fft_angle, np.pi), _mask_lt_m_pi)
            fft_angle = keras.ops.add(keras.ops.add(_tmp1, _tmp2), _tmp3)

            f_list.append(fft_angle) # fft_angle


    if _is_enabled('magic_ac'):
        _mac_lags = keras.ops.convert_to_tensor(ext_tsf_params['magic_ac'])
        magic_ac = _magic_ac(x_sub_means, keras.ops.transpose(variance, (0,2,1)), _mac_lags )
        f_list.append(magic_ac)

    if _is_enabled('autocorralation'):
        autocrr_lags  = keras.ops.convert_to_tensor(ext_tsf_params['autocorralation'])
        autocrrs = _autocorrelation(x_sub_means, keras.ops.transpose(variance, (0,2,1)), autocrr_lags)
        if not _is_group_enabled('autocorralation_'):
            f_list.append(autocrrs)

        else:
            if not _is_enabled('autocorralation_raw'):
                f_list.append(autocrrs)

            ac_mean = keras.ops.mean(autocrrs, axis=2, keepdims=True)
            ac_var = None
            if _is_enabled('autocorralation_ratio_of_max'):
                f_list.append(divide_no_nan(autocrrs, keras.ops.max(keras.ops.abs(autocrrs), axis=2, keepdims=True)))

            if any([_is_enabled(f'autocorralation_{_k}') for _k in ('var', 'std', 'skew', 'kurt')]):
                ac_sub_mean = keras.ops.subtract(autocrrs, ac_mean)

            if _is_enabled('autocorralation_mean'):
                f_list.append(ac_mean)

            if any([_is_enabled(f'autocorralation_{_k}') for _k in ('var', 'std', 'skew')]):
                ac_var = keras.ops.mean(keras.ops.power(ac_sub_mean, 2), axis=2, keepdims=True)
                if _is_enabled('autocorralation_var'):
                    f_list.append(ac_var)

            if any([_is_enabled(f'autocorralation_{_k}') for _k in ('std', 'skew')]):
                ac_std = _sqrt(ac_var)
                if _is_enabled('autocorralation_std'):
                    f_list.append(ac_std)

            if _is_enabled('autocorralation_skew'):
                _skew = _skewness(ac_sub_mean.shape[2],
                        keras.ops.transpose(ac_sub_mean, (0,2,1)),
                        keras.ops.transpose(ac_std, (0,2,1)))
                f_list.append(_skew)

            if _is_enabled('autocorralation_kurt'):
                _kurt = _kurtosis(ac_sub_mean.shape[2],
                        keras.ops.transpose(ac_sub_mean, (0,2,1)))
                f_list.append(_kurt)

    ######################################
    f_list2 = []
    for f in f_list:
        if f.dtype != keras.mixed_precision.global_policy().compute_dtype:
            f = keras.ops.cast(f, keras.mixed_precision.global_policy().compute_dtype)
        #assert len(f.shape) in (2,3)
        if len(f.shape) == 2:
            f = keras.ops.expand_dims(f, 2)

        order = [0, 1, 2]
        for i in range(3):
            if f.shape[i] == x_batch_n:
                order[0] = i
            elif f.shape[i] == x_featur_n:
                order[1] = i
            else:
                order[2] = i
        #assert set(order) == set((0,1,2))
        if order != [0, 1, 2]:
            f = keras.ops.transpose(f, order)

        f_list2.append(f)
        
    if len(f_list2) == 1:
        return f_list2[0]
    else:
        return keras.ops.concatenate(f_list2, axis=2)
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

@register_keras_serializable('tsf')
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


@register_keras_serializable('tsf')
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
                 regularization_rate = 1e-6,
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
        for i in range(self.head_num):
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
            s = Sequential([
                btm,
                LayerNormalization(name=self._gen_name('layer_nomalization')),
                ])
            self.btm_list.append(s)

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
                results.append(self.finisher_list[i](_l[i:(i+2)]))

        return Concatenate()(results)

@register_keras_serializable('tsf')
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

        # mixing on block- and feature wize
        self.bf_wise_mixer = self.gen_mixer(self.tsf_mixer_depth, self.tsf_mixer_base_kn)
        self.bf_wise_mixer_ax_w = self.gen_mixer(self.tsf_mixer_depth, self.tsf_mixer_base_kn)
        self.bf_wise_mixer_tsf_w = self.gen_mixer(self.tsf_mixer_depth, self.tsf_mixer_base_kn)


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
                    # batch, block, axes, TSFs
                    s.add(Conv2D(_kn, (1, input_shape[2]), name=self._gen_name('conv2d')))
                    # batch, block, 1, mixed_features
                    s.add(Reshape((input_shape[1], _kn), name=self._gen_name('reshape')))
                    # batch, block, mixed_features

                    # the above is equals 
                    # x = Reshape((input_shape[1], input_shape[2]*input_shape[3]))(x)
                    # x = Conv1D(_kn, 1)(x)
                    #
                    # input_shape[3] may be None sometime, so we use the above procedure.
                else:
                    s.add(Conv1D(self.block_mixer_base_kn*(2**d), 1, name=self._gen_name('conv1d')))
                    # Because of kernel_size is 1, it mixes features on each block
                    # batch, block, mixed_features

                if key == 'for_out' and self.no_out_activation and d == 0:
                    pass
                else:
                    if not self.few_norm or d == self.block_mixer_depth-1:
                        s.add(self.get_normalizer())
                    if not self.few_acti or d == 0:
                        s.add(self.get_activation())

                    s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))

            #setattr(self, key, s)
            self.block_wise_mixers[key] = s

        if self.ax_weight_depth > 0:
            self.gen_ax_weight = self.gen_weight(input_shape[2], self.ax_weight_depth, self.ax_weight_base_kn)

        if self.tsf_weight_depth > 0:
            self.gen_tsf_weight = self.gen_weight(self.tsf_mixer_base_kn, self.tsf_weight_depth, self.tsf_weight_base_kn)

        if self.blk_weight_depth > 0:
            self.gen_blk_weight = self.gen_weight(1, self.blk_weight_depth, self.blk_weight_base_kn)

    def gen_mixer(self, depth, base_kn):
        s = Sequential(name=self._gen_name('sequential'))
        for i in range(depth-1, -1, -1):
            s.add(Conv2D(base_kn*(2**i), (1,1), name=self._gen_name('conv2d')))
            if not self.few_norm or i == depth-1:
                s.add(self.get_normalizer())
            if not self.few_acti or i == 0:
                s.add(self.get_activation())
            s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))
        return s


    def gen_weight(self, out_len, depth, base_kn):
        s = Sequential(name=self._gen_name('sequential'))
        # blockwise mixer
        for d in range(depth-1, -1, -1):
            s.add(Conv1D(base_kn*(2**d), 1, name=self._gen_name('conv1d')))

            if not self.few_norm or d == depth-1:
                s.add(self.get_normalizer())
            if not self.few_acti or d == 0:
                s.add(self.get_activation())
            s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))

        #s.add(Conv1D(out_len, 1, kernel_regularizer=L2(self.regularization_rate), name=self._gen_name('conv1d')))
        s.add(Conv1D(out_len, 1, kernel_regularizer=l2(self.regularization_rate), name=self._gen_name('conv1d')))
        #s.add(Conv1D(out_len, 1, name=self._gen_name('conv1d')))
        #s.add(Lambda(lambda x: keras.ops.subtract(x, keras.ops.relu(x)), name=self._gen_name('lambda')))
        s.add(Dropout(self.dropout_rate, name=self._gen_name('dropout')))
        return s


    def call(self, inputs):
        _ax_w = None
        if self.ax_weight_depth > 0:
            _w_x_ax  = self.block_wise_mixers['for_ax_weight'](self.bf_wise_mixer_ax_w(inputs))
            _ax_w = keras.ops.expand_dims(self.gen_ax_weight(_w_x_ax), axis=3)

        _tsf_w = None
        if self.tsf_weight_depth > 0:
            _w_x_tsf = self.block_wise_mixers['for_tsf_weight'](self.bf_wise_mixer_tsf_w(inputs))
            _tsf_w = keras.ops.expand_dims(self.gen_tsf_weight(_w_x_tsf), axis=2) # batch, block, sensor_axes, 1

        x = self.bf_wise_mixer(inputs)
        if _ax_w is not None:
            x = keras.ops.multiply(x, _ax_w)
            _w = keras.ops.where(_ax_w == 0, 1., _ax_w)
            x = keras.ops.divide(x, _w) # x*_ax_w/1 if _ax_w == 0, x*_ax_w/_ax_w = x if _ax_w != 0
        if _tsf_w is not None:
            x = keras.ops.multiply(x, _tsf_w)
            _w = keras.ops.where(_tsf_w == 0, 1., _tsf_w)
            x = keras.ops.divide(x, _w)

        x = self.block_wise_mixers['for_out'](x)
        if self.blk_weight_depth > 0:
            x = keras.ops.multiply(x, Reshape((x.shape[1], 1))(self.gen_blk_weight(x))) # batch, block, sensor_axes, 1

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

