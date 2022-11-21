from spektral.data import Dataset, Graph
import numpy as np
import tensorflow as tf
from spektral.layers import ECCConv, GlobalSumPool, GINConv



class ECC_Net(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        # self.masking = GraphMasking()
        self.conv1 = ECCConv(32, activation="relu")
        self.drop = tf.keras.layers.Dropout(0.3)
        self.global_pool = GlobalSumPool()
        self.dense = tf.keras.layers.Dense(1) # same as dataset.n_labels
        self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x, a, e = inputs
        x = self.conv1([x, a, e])
        x = self.batchnorm(x)
        x = self.drop(x)
        output = self.global_pool(x)
        output = self.dense(output)

        return output

class GIN_Net(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        # self.masking = GraphMasking()
        self.conv1 = GINConv(32, activation="relu")
        self.drop = tf.keras.layers.Dropout(0.3)
        self.global_pool = GlobalSumPool()
        self.dense = tf.keras.layers.Dense(1) # same as dataset.n_labels
        self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x, a, e = inputs

        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(a, zero)
        indices = tf.where(where)
        values = tf.gather_nd(a, indices)
        a_sparse = tf.SparseTensor(indices, values, a.shape)

        x = self.conv1([x, a_sparse])
        x = self.batchnorm(x)
        x = self.drop(x)
        output = self.global_pool(x)
        output = self.dense(output)

        return output




