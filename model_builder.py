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

import tensorflow as tf
import base_ops
import numpy as np
from model_spec import ModelSpec


def compute_vertex_channels(input_channels, output_channels, matrix):
    """
    Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Args:
        input_channels: input channel count.
        output_channels: output channel count.
        matrix: adjacency matrix for the module (pruned by model_spec).
    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # tf.logging.info('vertex_channels: %s', str(vertex_channels))

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


def projection(channels, is_training, data_format):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return base_ops.ConvBnRelu(1, channels, is_training, data_format)


def truncate(inputs_shape, inputs, channels, data_format):
    """Slice the inputs to channels if necessary."""
    if data_format == 'channels_last':
        input_channels = inputs_shape[3]
    else:
        assert data_format == 'channels_first'
        input_channels = inputs_shape[1]

    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs  # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        if data_format == 'channels_last':
            return tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, channels])
        else:
            return tf.slice(inputs, [0, 0, 0, 0], [-1, channels, -1, -1])


#class CellModel(tf.keras.layers.Layer):
class CellModel(tf.keras.Model):
    def __init__(self, spec: ModelSpec, inputs_shape, channels, is_training):
        super(CellModel, self).__init__()

        self.inputs_shape = inputs_shape
        self.spec = spec
        self.is_training = is_training
        self.channels = channels
        self.num_vertices = np.shape(spec.matrix)[0]
        if spec.data_format == 'channels_last':
            self.channel_axis = 3
        elif spec.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            raise ValueError('invalid data_format')

        input_channels = inputs_shape[self.channel_axis]
        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = compute_vertex_channels(
            input_channels, channels, spec.matrix)

        # --------------------------------------------------------
        self.ops = {}
        self.proj_list = {}

        final_concat_in = []
        # Construct tensors shape from input forward
        self.tensors = [inputs_shape]

        for t in range(1, self.num_vertices - 1):

            # Create add connection from projected input
            if self.spec.matrix[0, t]:
                self.proj_list[t] = projection(
                    self.vertex_channels[t],
                    self.is_training,
                    self.spec.data_format)

                # Perform op at vertex t
            op = base_ops.OP_MAP[self.spec.ops[t]](
                is_training=self.is_training,
                data_format=self.spec.data_format)
            self.ops[t] = op.build(self.vertex_channels[t])

            t_shape = list(inputs_shape)
            t_shape[self.channel_axis] = self.vertex_channels[t]

            self.tensors.append(tuple(t_shape))
            if self.spec.matrix[t, self.num_vertices - 1]:
                final_concat_in.append(self.tensors[t])

        # Construct final output tensor by concating all fan-in and adding input.
        if not final_concat_in:
            # No interior vertices, input directly connected to output
            assert spec.matrix[0, self.num_vertices - 1]

            self.outputs1 = projection(
                self.channels,
                self.is_training,
                self.spec.data_format)

        else:
            if self.spec.matrix[0, self.num_vertices - 1]:
                self.outputs1 = projection(
                    self.channels,
                    self.is_training,
                    self.spec.data_format)

    def call(self, inputs):
        # Construct tensors from input forward
        tensors = [tf.identity(inputs, name='input')]
        final_concat_in = []

        for t in range(1, self.num_vertices - 1):

            # Create interior connections, truncating if necessary
            add_in = [truncate(self.tensors[src], tensors[src], self.vertex_channels[t], self.spec.data_format)
                      for src in range(1, t) if self.spec.matrix[src, t]]

            # Create add connection from projected input
            if self.spec.matrix[0, t]:
                add_in.append(self.proj_list[t](tensors[0]))

            if len(add_in) == 1:
                vertex_input = add_in[0]
            else:
                vertex_input = tf.add_n(add_in)

            # Perform op at vertex t
            vertex_value = self.ops[t](vertex_input)

            tensors.append(vertex_value)
            if self.spec.matrix[t, self.num_vertices - 1]:
                final_concat_in.append(tensors[t])

        # Construct final output tensor by concating all fan-in and adding input.
        if not final_concat_in:
            # No interior vertices, input directly connected to output
            assert spec.matrix[0, self.num_vertices - 1]
            outputs = self.outputs1(tensors[0])
        else:
            if len(final_concat_in) == 1:
                outputs = final_concat_in[0]
            else:
                outputs = tf.concat(final_concat_in, self.channel_axis)

            if self.spec.matrix[0, self.num_vertices - 1]:
                outputs += self.outputs1(tensors[0])

        return outputs

    def build_graph(self):
        shape = tuple(list(self.inputs_shape)[1:])
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = {
            "spec": self.spec,
            "inputs_shape": self.inputs_shape,
            "channels": self.channels,
            "is_training": self.is_training
        }
        return config

'''
class Arch_Model(tf.keras.Model):
    def __init__(self, spec: ModelSpec, inputs_shape, is_training, num_stacks=3, num_cells=3):
        super(Arch_Model, self).__init__()
        self.spec = spec
        self.is_training = is_training
        self.num_stacks = num_stacks
        self.num_cells = num_cells
        self.inputs_shape = inputs_shape

        self.stem = base_ops.Conv_BN_ReLU(3, 128, is_training, self.spec.data_format)
        self.down_sample_layers = {}
        self.cell_layers = {}

        now_channel = 128
        for i in range(num_stacks):
            if i > 0:
                self.down_sample_layers[i] = tf.keras.layers.MaxPool2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding='same',
                    data_format=self.spec.data_format)
                now_channel *= 2

            layers_list = []
            for j in range(num_cells):
                layers_list.append(Cell_Model(self.spec,
                                              inputs_shape=inputs_shape,
                                              channels=now_channel,
                                              is_training=is_training))
            self.cell_layers[i] = layers_list

        self.glob_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format=self.spec.data_format)

    def call(self, inputs):
        x = self.stem(inputs)
        for i in range(self.num_stacks):
            if i > 0:
                x = self.down_sample_layers[i](x)

            for cell in self.cell_layers[i]:
                x = cell(x)

        x = self.glob_avg_pool(x)
        return x

    def build_graph(self):
        shape = tuple(list(self.inputs_shape)[1:])
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
'''

class residual_block(tf.keras.Model):
    def __init__(self, spec, init_channel, data_format='channels_last'):
        super(residual_block, self).__init__()
        self.pool = tf.keras.layers.AvgPool2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same',
                data_format=spec.data_format)
        self.conv = tf.keras.layers.Conv2D(
                filters=init_channel,
                kernel_size=1,
                strides=(1, 1),
                use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                padding='same',
                data_format=spec.data_format)

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.conv(x)
        return x


def build_arch_model_original(spec: ModelSpec, inputs_shape, init_channel=16, num_stacks=3, num_cells=5, is_training=None):
    model = tf.keras.Sequential()
    # stem
    model.add(base_ops.ConvBn(3, 16, is_training, spec.data_format))
    shape = list(inputs_shape)
    shape[3] = 16

    for i in range(num_stacks):
        if i > 0:
            model.add(residual_block(spec, init_channel))

            init_channel *= 2
            shape[1] = shape[1] // 2
            shape[2] = shape[2] // 2

        for j in range(num_cells):
            if j > 0:
                shape[3] = init_channel
            model.add(CellModel(spec,
                                inputs_shape=tuple(shape),
                                channels=init_channel,
                                is_training=is_training))

    #model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=spec.data_format))
    return model


def build_arch_model(spec: ModelSpec, inputs_shape, init_channel=16, num_stacks=3, num_cells=5, is_training=None):
    model = tf.keras.Sequential()
    # stem
    model.add(base_ops.ConvBn(3, 16, is_training, spec.data_format))
    shape = list(inputs_shape)
    shape[3] = 16

    for i in range(num_stacks):
        if i > 0:
            model.add(residual_block(spec, init_channel))

            init_channel *= 2
            shape[1] = shape[1] // 2
            shape[2] = shape[2] // 2

        for j in range(num_cells):
            if j > 0:
                shape[3] = init_channel
            model.add(CellModel(spec,
                                inputs_shape=tuple(shape),
                                channels=init_channel,
                                is_training=is_training).build_graph())

    #model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=spec.data_format))
    return model


if __name__ == '__main__':

    matrix = np.array([[0, 1, 1, 1, 0, 1, 0],  # input layer
                       [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
                       [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
                       [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
                       [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
                       [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
                       [0, 0, 0, 0, 0, 0, 0]])

    ops = ['INPUT', 'conv3x3-bn-relu', 'maxpool3x3', 'conv1x1-bn-relu', 'maxpool3x3', 'identity',
           'OUTPUT']

    spec = ModelSpec(matrix, ops)

    model = build_arch_model_original(spec, (None, 28, 28, 1), init_channel=128, is_training=True, num_stacks=3, num_cells=3)
    model.build([None, 28, 28, 1])
    print(model.summary())
    tf.keras.utils.plot_model(
        model, to_file='arch_model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
        layer_range=None, show_layer_activations=False)

    for layer_no in range(len(model.layers)):
        print(model.layers[layer_no].name)
        if 'cell' in model.layers[layer_no].name:
            cell = model.layers[layer_no]
            for i in range(len(cell.layers)):
                print(cell.layers[i].name)

    del model

    model2 = build_arch_model(spec, (None, 28, 28, 1), init_channel=128, is_training=True, num_stacks=3, num_cells=3)
    for layer_no in range(len(model2.layers)):
        print(model2.layers[layer_no].name)
        if 'model' in model2.layers[layer_no].name:
            cell = model2.layers[layer_no]
            for i in range(len(cell.layers)):
                print(cell.layers[i].name)
    '''
    del model
    model2 = build_arch_model(spec, (None, 28, 28, 1), init_channel=128, is_training=True, num_stacks=3, num_cells=3)
    model2.build([None, 28, 28, 1])
    print(model2.summary())
    '''
