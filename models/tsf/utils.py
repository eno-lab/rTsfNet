import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, LeakyReLU

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
        x_except_imu.append(tf.expand_dims(in_x[:,:,3*imu_feature_num+offset], 2)) # compus
        x_tags_except_imu.append(np.array([[sensor_location, Tag.COMPUS, Tag.NONE]]))

        l = []
        for j, flag in enumerate((Tag.ANGLE, Tag.ACC, Tag.ACC, Tag.ANG_VEL, Tag.ANG_VEL)):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

        tags.append(np.array(l))
        sensor_location += 1

    if version == 1:
        return x_except_imu, x_tags_except_imu, x_imu, tags
    elif version >= 2:
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

    if version == 1:
        return None, None, x_imu, tags
    elif version >= 2:
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
                
    if version == 1:
        return None, None, x_imu, tags
    elif version >= 2:
        return None, None, [x_imu], [tags]


def extract_imu_tensor_func_ucihar(in_x, version=1):
    x_imu = [in_x]
    sensor_location = 1

    tags = []
    for j, flag in enumerate((Tag.ACC, Tag.GYRO, Tag.ACC)):
        tags.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])

    tags = [np.array(tags)]

    if version == 1:
        return None, None, x_imu, tags
    elif version >= 2:
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

    if version == 1:
        return None, None, x_imu, tags
    elif version >= 2:
        return None, None, [x_imu], [tags]


def extract_imu_tensor_func_wisdm(in_x, version=1):
    x_imu = []
    tags = []
    x_imu.append(in_x[:,:,0:3])
    tags.append(np.array([[1, Tag.ACC, axis] for axis in Tag.XYZ]))

    if version == 1:
        return None, None, x_imu, tags
    elif version >= 2:
        return None, None, [x_imu], [tags]


class Tag:
    UNKNOWN = 0
    ACC = 1
    GYRO = 2
    MAG = 3
    ANGLE = 4
    ANG_VEL = 5
    COMPUS = 6
    #######
    NONE = 0
    XYZ = [1, 2, 3]
    L2 = 4



@tf.keras.saving.register_keras_serializable('tsf')
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
        return f'{self.name}/{name}_{self.layer_count[name]}'


    def get_config(self):
        config = super(AbstructLayer, self).get_config()
        config.update({'normalizer': self.normalizer})
        config.update({'activation_func': self.activation_func})
        config.update({'dropout_rate': self.dropout_rate})
        config.update({'version': self.version})
        return config
