import numpy as np
import keras
from keras.layers import Layer, LayerNormalization, LeakyReLU, Reshape
from keras.utils import register_keras_serializable

def extract_imu_tensor_func_oppotunity_separation(in_x, version=1):
    imu_feature_num = 3
    with_sid = False
    if in_x.shape[-1] == 3*imu_feature_num+1:
        with_sid = True
    elif in_x.shape[-1] != 3*imu_feature_num:
        raise ValueError(f'in_x.shape[-1] is not {3*imu_feature_num} or {3*imu_feature_num+1}')

    x_imu = [in_x[:,:,0:3*imu_feature_num]]
    l = []
    for j, flag in enumerate((Tag.ACC, Tag.GYRO, Tag.MAG)):
        l.extend([[0, flag, axis] for axis in Tag.XYZ])

    tags = [np.array(l)]

    if with_sid:
        return None, None, [x_imu], [tags], in_x[:,0:1,3*imu_feature_num:(3*imu_feature_num+1)] # sensor ids for each batches
    else:
        return None, None, [x_imu], [tags]


def extract_imu_tensor_func_oppotunity(in_x, version=1):
    tags = []

    sensor_location = 1

    # 5 imu with 3x3 features
    imu_num = 5
    imu_feature_num = 3
    x_imu = []
    for i in range(imu_num):
        x_imu.append(in_x[:,:,(0+3*imu_feature_num*i):(3*imu_feature_num*(i+1))])

        l = []
        for j, flag in enumerate((Tag.ACC, Tag.GYRO, Tag.MAG)):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))
        sensor_location += 1

    base_ix = 3*imu_feature_num*imu_num
    # 2 imu with 5x3 features plus 1 (compus)
    x_except_imu = []
    imu_num = 2
    imu_feature_num = 5
    x_tags_except_imu = []
    for i in range(imu_num):
        offset = base_ix + (3*imu_feature_num+1)*i
        x_imu.append(in_x[:,:,(0+offset):(3*imu_feature_num+offset)])
        x_except_imu.append(keras.ops.expand_dims(in_x[:,:,3*imu_feature_num+offset], 2)) # compus
        x_tags_except_imu.append(np.array([[sensor_location, Tag.COMPUS, Tag.NONE]]))

        l = []
        for j, flag in enumerate((Tag.ANGLE, Tag.ACC, Tag.ACC, Tag.ANG_VEL, Tag.ANG_VEL)):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))
        sensor_location += 1

    return x_except_imu, x_tags_except_imu, [x_imu[0:5], x_imu[5:7]], [tags[0:5], tags[5:7]]


def extract_imu_tensor_func_oppotunity_task_c(in_x, version=1):
    tags = []

    print(in_x.shape)
    sensor_location = 1

    # 5 imu with 3x3 features
    imu_num = 5
    imu_feature_num = 3
    x_imu = []
    for i in range(imu_num):
        x_imu.append(in_x[:,:,(0+3*imu_feature_num*i):(3*imu_feature_num*(i+1))])

        l = []
        for j, flag in enumerate((Tag.ACC, Tag.GYRO, Tag.MAG)):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))
        sensor_location += 1

    return None, None, [x_imu], [tags]


def extract_imu_tensor_func_pamap2_separation(in_x, version=1):

    imu_feature_num = 4 # a16, a8, gyro, mag
    with_sid = False
    if in_x.shape[-1] == 3*imu_feature_num+1:
        with_sid = True
    elif in_x.shape[-1] != 3*imu_feature_num:
        raise ValueError(f'in_x.shape[-1] is not {3*imu_feature_num} or {3*imu_feature_num+1}')

    x_imu = [in_x[:,:,0:3*imu_feature_num]]

    l = []
    for j, flag in enumerate((Tag.ACC, Tag.ACC, Tag.GYRO, Tag.MAG)):
        l.extend([[0, flag, axis] for axis in Tag.XYZ])

    tags = [np.array(l)]

    if with_sid:
        # in_x: batch, seq, sensor_axes + separation id
        # A batch has the same separation id in each row
        return None, None, [x_imu], [tags], in_x[:,0:1,3*imu_feature_num:(3*imu_feature_num+1)] # sensor ids for each batches
    else:
        return None, None, [x_imu], [tags]


def extract_imu_tensor_func_pamap2(in_x, version=1):
    sensor_location = 1

    imu_num = 3
    imu_feature_num = 4 # a16, a8, gyro, mag
    x_imu = []
    for i in range(imu_num):
        x_imu.append(in_x[:,:,(0+3*imu_feature_num*i):(3*imu_feature_num+3*imu_feature_num*i)])

    tags = []
    for i in range(imu_num):
        l = []
        for j, flag in enumerate((Tag.ACC, Tag.ACC, Tag.GYRO, Tag.MAG)):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))
        sensor_location += 1

    return None, None, [x_imu], [tags]


def extract_imu_tensor_func_ucihar(in_x, version=1):
    x_imu = [in_x]
    sensor_location = 1

    tags = []
    for j, flag in enumerate((Tag.ACC, Tag.GYRO, Tag.ACC)):
        tags.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

    tags = [np.array(tags)]

    return None, None, [x_imu], [tags]


def extract_imu_tensor_func_daphnet(in_x, version=1):
    sensor_location = 1
    imu_num = 3
    imu_feature_num = 1
    x_imu = []
    tags = []
    for i in range(imu_num):
        offset = 3*imu_feature_num*i
        x_imu.append(in_x[:,:,(0+offset):(3*imu_feature_num+offset)])

        tags.append(np.array([[sensor_location, Tag.ACC, axis] for axis in Tag.XYZ]))
        sensor_location += 1

    return None, None, [x_imu], [tags]


def extract_imu_tensor_func_wisdm(in_x, version=1):
    x_imu = []
    tags = []
    x_imu.append(in_x[:,:,0:3])
    tags.append(np.array([[1, Tag.ACC, axis] for axis in Tag.XYZ]))

    return None, None, [x_imu], [tags]


def extract_imu_tensor_func_real_world_separation(in_x, version=1):
    sensor_location = 1

    imu_feature_num = 3 # acc, gyro, mag

    with_sid = False
    if in_x.shape[-1] == 3*imu_feature_num+1:
        with_sid = True
    elif in_x.shape[-1] != 3*imu_feature_num:
        raise ValueError(f'in_x.shape[-1] is not {3*imu_feature_num} or {3*imu_feature_num+1}')

    x_imu = [in_x[:,:,0:3*imu_feature_num]]
    l = []
    for j, flag in enumerate((Tag.ACC, Tag.GYRO, Tag.MAG)):
        l.extend([[0, flag, axis] for axis in Tag.XYZ])

    tags = [np.array(l)]

    if with_sid:
        return None, None, [x_imu], [tags], in_x[:,0:1,3*imu_feature_num:(3*imu_feature_num+1)] # sensor ids for each batches
    elif version >= 2:
        return None, None, [x_imu], [tags]


def extract_imu_tensor_func_real_world(in_x, version=1):
    sensor_location = 1

    imu_num = 7
    imu_feature_num = 3 # acc, gyro, mag
    x_imu = []
    for i in range(imu_num):
        x_imu.append(in_x[:,:,(0+3*imu_feature_num*i):(3*imu_feature_num+3*imu_feature_num*i)])

    tags = []
    for i in range(imu_num):
        l = []
        for j, flag in enumerate((Tag.ACC, Tag.GYRO, Tag.MAG)):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))
        sensor_location += 1

    return None, None, [x_imu], [tags]


def extract_imu_tensor_func_m_health(in_x, version=1):
    """
        Normal: OK
        Combination: OK
        Combination-with_sid: NO
        Separation: NO
        Separation-with_sid: NO
    """
    sensor_location = 1
    x_imu = []
    tags = []
    x_except_imu = None
    x_tags_except_imu = None

    imu_num = 2
    available_sensors = [True, True]
    if in_x.shape[-1] == 3: # 0 only
        available_sensors = [True, False]
    elif in_x.shape[-1] == 2: # 1 (ecgs) only
        available_sensors = [False, True]
    elif in_x.shape[-1] % 9 == 5: # 0 + 1 + [2, 3]
        imu_num = (in_x.shape[-1]-5)//9
    elif in_x.shape[-1] % 9 == 2:  # 1 + [2, 3]
        imu_num = (in_x.shape[-1]-2)//9
        available_sensors = [False, True]
    elif in_x.shape[-1] % 9 == 3:  # 0 + [2, 3]
        imu_num = (in_x.shape[-1]-3)//9
        available_sensors = [True, False]
    elif in_x.shape[-1] % 9 == 0:  # [2, 3]
        imu_num = (in_x.shape[-1]-3)//9
        available_sensors = [False, False]
    else:
        raise ValueError(f"Invalid input shape: {in_x.shape=}")

    six = 0
    if available_sensors[0]:
        x_imu.append(in_x[:,:,0:3])
        tags.append(np.array([[sensor_location, Tag.ACC, axis] for axis in Tag.XYZ]))
        six += 3
    sensor_location += 1

    ecg_six = six
    if available_sensors[1]:
        six = 2 # skip two ecg


    imu_feature_num = 3 # acc, mag, gyro
    for i in range(imu_num):
        x_imu.append(in_x[:,:,(six+3*imu_feature_num*i):(six+3*imu_feature_num*(1+i))])

        l = []
        for j, flag in enumerate((Tag.ACC, Tag.MAG, Tag.GYRO)):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))
        sensor_location += 1

    if available_sensors[1]:
        x_except_imu = [in_x[:,:,ecg_six:(ecg_six+2)]] # ECGs
        x_tags_except_imu = [np.array([[sensor_location+i, Tag.ECG, Tag.NONE] for i in range(0, 2)])]

    return x_except_imu, x_tags_except_imu, [x_imu[0:1], x_imu[1:3]], [tags[0:1], tags[1:3]]


def extract_imu_tensor_func_uschad(in_x, version=1):
    sensor_location = 1
    x_imu = []
    tags = []

    x_imu.append(in_x[:,:,0:6])
    l = []
    for flag in (Tag.ACC, Tag.GYRO):
        l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

    tags.append(np.array(l))

    return None, None, [x_imu], [tags]


def extract_imu_tensor_func_mighar(in_x, version=1):

    sensor_num = in_x.shape[-1]//9
    sid_list = list(range(sensor_num))

    in_x_sid = None
    if in_x.shape[-1]%9 == 1: # separation with id
        in_x_sid = in_x[:,0:1,-2:-1]
        in_x = in_x[:,:,0:-1]
    elif in_x.shape[1]%2 == 1: # combination with id
        # TODO remove the following GPU to CPU data transfer
        sid_list = list[np.array(in_x[0,-1,range(0, in_x.shape[-1], 9)])]
        in_x = in_x[:,0:-1,:]
    elif in_x.shape[-1]%9 != 0 or in_x.shape[1]%2 != 0: # error
        raise ValueError(f'Invalid input shape: {in_x.shape=}')

    x_imu = []
    tags = []

    for i, sid in zip(range(sensor_num), sid_list):
        six = i*9
        eix = six+9

        x_imu.append(in_x[:,:,six:eix])

        l = []
        for flag in (Tag.ACC, Tag.GYRO, Tag.MAG):
            l.extend([[sid, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))

    if in_x_sid:
        return None, None, [x_imu], [tags], in_x_sid
    else:
        return None, None, [x_imu], [tags]

class Tag:
    UNKNOWN = 0
    ACC = 1
    GYRO = 2
    MAG = 3
    ANGLE = 4
    ANG_VEL = 5
    COMPUS = 6
    ECG = 7
    #######
    NONE = 0
    XYZ = [1, 2, 3]
    L2 = 4


@register_keras_serializable('tsf')
class AbstructLayer(Layer):

    def __init__(self, activation_func, normalizer, dropout_rate, version = 1, **kwargs):
        super(AbstructLayer, self).__init__(**kwargs)
        self.normalizer = normalizer
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.version = version
        self.layer_count = {}


    def get_activation(self):
        if self.activation_func is None:
            return LeakyReLU(name=self._gen_name('leaky_relu'))
        else:
            return Lambda(self.activation_func, name=self._gen_name('lambda'))


    def get_normalizer(self):
        if self.normalizer is None:
            return LayerNormalization(name=self._gen_name('layer_nomalization'))
        else:
            return self.normalizer()


    def _gen_name(self, name):
        if name not in self.layer_count:
            self.layer_count[name] = 0
        else:
            self.layer_count[name] += 1

        return f'{self.name}_{name}_{self.layer_count[name]}'


    def get_config(self):
        config = super(AbstructLayer, self).get_config()
        config.update({'normalizer': self.normalizer})
        config.update({'activation_func': self.activation_func})
        config.update({'dropout_rate': self.dropout_rate})
        config.update({'version': self.version})
        return config


def divide_no_nan(x, y):
    # x*y/1 -> 0 if elem y ==0, x*y/y/y = x/y if y != 0; 
    # keep GTape for both x and y on every elements.
    yy = keras.ops.multiply(y, y)
    x = keras.ops.multiply(x, y) # keeping GTape for elements related to y == 0
    w = keras.ops.where(y == 0, 1., yy)
    x = keras.ops.divide(x, w) 
    return x
