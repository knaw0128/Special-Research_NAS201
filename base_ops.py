# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base operations used by the modules in this search space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import tensorflow as tf
from keras import backend as K


# Currently, only channels_last is well supported.
VALID_DATA_FORMATS = frozenset(['channels_last', 'channels_first'])
MIN_FILTERS = 8
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, conv_size, conv_filters, is_training, data_format):
        super(ConvBnRelu, self).__init__()
        self.conv_size = conv_size
        self.conv_filters = conv_filters
        self.is_training = is_training
        self.data_format = data_format
        """Convolution followed by batch norm and ReLU."""
        if data_format == 'channels_last':
            axis = 3
        elif data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('invalid data_format')

        self.is_training = is_training
        self.conv = tf.keras.layers.Conv2D(
            filters=conv_filters,
            kernel_size=conv_size,
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            padding='same',
            data_format=data_format)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=BN_MOMENTUM,
            epsilon=BN_EPSILON)

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x, training=self.is_training)
        x = self.relu(x)
        return x

    def get_config(self):
        config = {
            "conv_size": self.conv_size,
            "conv_filters": self.conv_filters,
            "is_training": self.is_training,
            "data_format": self.data_format
        }
        return config


class ConvBn(tf.keras.layers.Layer):
    def __init__(self, conv_size, conv_filters, is_training, data_format):
        super(ConvBn, self).__init__()
        self.conv_size = conv_size
        self.conv_filters = conv_filters
        self.is_training = is_training
        self.data_format = data_format
        """Convolution followed by batch norm ."""
        if data_format == 'channels_last':
            axis = 3
        elif data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('invalid data_format')

        self.is_training = is_training
        self.conv = tf.keras.layers.Conv2D(
            filters=conv_filters,
            kernel_size=conv_size,
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            padding='same',
            data_format=data_format)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=BN_MOMENTUM,
            epsilon=BN_EPSILON)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x, training=self.is_training)
        return x

    def get_config(self):
        config = {
            "conv_size": self.conv_size,
            "conv_filters": self.conv_filters,
            "is_training": self.is_training,
            "data_format": self.data_format
        }
        return config

'''
def conv_bn_relu(inputs, conv_size, conv_filters, is_training, data_format):
    """Convolution followed by batch norm and ReLU."""
    if data_format == 'channels_last':
        axis = 3
    elif data_format == 'channels_first':
        axis = 1
    else:
        raise ValueError('invalid data_format')

    net = tf.keras.layers.Conv2D(
        filters=conv_filters,
        kernel_size=conv_size,
        strides=(1, 1),
        use_bias=False,
        kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
        padding='same',
        data_format=data_format)(inputs)

    net = tf.keras.layers.BatchNormalization(
        axis=axis,
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON)(net, training=is_training)

    net = tf.nn.relu(net)

    return net
'''


class BaseOp(object):
    """Abstract base operation class."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, is_training, data_format='channels_last'):
        self.is_training = is_training
        if data_format.lower() not in VALID_DATA_FORMATS:
            raise ValueError('invalid data_format')
        self.data_format = data_format.lower()

    @abc.abstractmethod
    def build(self, channels):
        """Builds the operation with input tensors and returns an output tensor.
        Args:
          channels: int number of output channels of operation. The operation may
            choose to ignore this parameter.
        Returns:
          a 4-D Tensor with the same data format.
        """
        pass


class Zeroize(BaseOp):

    def build(self, channels):
        return tf.keras.layers.Lambda(lambda x: tf.zeros(tf.shape(x)), name='Zeroize_'+str(K.get_uid('Zeroize')))


class Identity(BaseOp):
    """Identity operation (ignores channels)."""

    def build(self, channels):
        del channels  # Unused
        return tf.keras.layers.Lambda(lambda x: x, name='Identity_'+str(K.get_uid('Identity')))


class Conv1x1BnRelu(BaseOp):
    """1x1 convolution with batch norm and ReLU activation."""

    def build(self, channels):
        return ConvBnRelu(1, channels, self.is_training, self.data_format)


class Conv3x3BnRelu(BaseOp):
    """3x3 convolution with batch norm and ReLU activation."""

    def build(self, channels):
        return ConvBnRelu(3, channels, self.is_training, self.data_format)


class AvgPool3x3(BaseOp):
    """3x3 max pool with no subsampling."""

    def build(self, channels):
        del channels  # Unused
        net = tf.keras.layers.AvgPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format)

        return net


class AvgPool2x2(BaseOp):
    """3x3 max pool with no subsampling."""

    def build(self, channels):
        del channels  # Unused
        net = tf.keras.layers.AvgPool2D(
            pool_size=(2, 2),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format)

        return net


class Conv1x1(BaseOp):

    def build(self, channels):
        net = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            padding='same',
            data_format=self.data_format
        )
        return net


class Conv3x3Bn(BaseOp):

    def build(self, channels):
        return ConvBn(3,Conv3x3Bn,self.is_training,self.data_format)


'''
class BottleneckConv3x3(BaseOp):
    """[1x1(/4)]+3x3+[1x1(*4)] conv. Uses BN + ReLU post-activation."""

    # TODO(chrisying): verify this block can reproduce results of ResNet-50.

    def build(self, inputs, channels):
        with tf.compat.v1.variable_scope('BottleneckConv3x3'):
            net = conv_bn_relu(
                inputs, 1, channels // 4, self.is_training, self.data_format)
            net = conv_bn_relu(
                net, 3, channels // 4, self.is_training, self.data_format)
            net = conv_bn_relu(
                net, 1, channels, self.is_training, self.data_format)

        return net


class BottleneckConv5x5(BaseOp):
    """[1x1(/4)]+5x5+[1x1(*4)] conv. Uses BN + ReLU post-activation."""

    def build(self, inputs, channels):
        with tf.compat.v1.variable_scope('BottleneckConv5x5'):
            net = conv_bn_relu(
                inputs, 1, channels // 4, self.is_training, self.data_format)
            net = conv_bn_relu(
                net, 5, channels // 4, self.is_training, self.data_format)
            net = conv_bn_relu(
                net, 1, channels, self.is_training, self.data_format)

        return net


class MaxPool3x3Conv1x1(BaseOp):
    """3x3 max pool with no subsampling followed by 1x1 for rescaling."""

    def build(self, inputs, channels):
        with tf.compat.v1.variable_scope('MaxPool3x3-Conv1x1'):
            net = tf.keras.layers.MaxPool2D(
                pool_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format=self.data_format)(inputs)

            net = conv_bn_relu(net, 1, channels, self.is_training, self.data_format)

        return net
'''

# Commas should not be used in op names
OP_MAP = {
    'none': Zeroize,
    'skip_connect': Identity,
    'nor_conv_1x1': Conv1x1BnRelu,
    'nor_conv_3x3': Conv3x3BnRelu,
    'avg_pool_3x3': AvgPool3x3,
    'avg_pool_2x2': AvgPool2x2,
    'conv_1x1': Conv1x1,
    'convbn_3x3': Conv3x3Bn,

    #'bottleneck3x3': BottleneckConv3x3,
    #'bottleneck5x5': BottleneckConv5x5,
    #'maxpool3x3-conv1x1': MaxPool3x3Conv1x1,
}
