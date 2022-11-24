from spektral.data import Dataset, Graph
import numpy as np
import tensorflow as tf
from spektral.layers import ECCConv, GlobalSumPool, GINConvBatch



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
        self.conv1 = GINConvBatch(32, activation="relu")
        self.drop = tf.keras.layers.Dropout(0.3)
        self.global_pool = GlobalSumPool()
        self.dense = tf.keras.layers.Dense(1) # same as dataset.n_labels
        self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x, a, e = inputs

        x = self.conv1([x, a])
        x = self.batchnorm(x)
        x = self.drop(x)
        output = self.global_pool(x)
        output = self.dense(output)

        return output




