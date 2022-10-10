import logging
from spektral.data import Dataset, Graph
import pickle
import numpy as np
import csv
import tensorflow as tf
import os
import wget
from os import path
from nas_bench_201_dataset import NasBench101Dataset

from spektral.data import BatchLoader
from spektral.layers import ECCConv, GlobalSumPool

################################################################################
# Config
################################################################################
learning_rate = 1e-3    # Learning rate
epochs = 40             # Number of training epochs
batch_size = 8         # Batch size
################################################################################
# Load data
################################################################################
file = open('./model_label.pkl', 'rb')
record = pickle.load(file)
file.close()
dataset = NasBench101Dataset(record_dic=record, shuffle_seed=0, start=0,
                                end=100, inputs_shape=(None, 32, 32, 3), num_classes=10)

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]


################################################################################
# Build model
################################################################################
class Net(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        # self.masking = GraphMasking()
        self.conv1 = ECCConv(32, activation="relu")
        self.drop = tf.keras.layers.Dropout(0.3)
        self.global_pool = GlobalSumPool()
        self.dense = tf.keras.layers.Dense(n_out)
        self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x, a, e = inputs
        x = self.conv1([x, a, e])
        x = self.batchnorm(x)
        x = self.drop(x)
        output = self.global_pool(x)
        output = self.dense(output)

        return output


model = Net()
optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer, loss="mse")
checkpoint_filepath = './checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

################################################################################
# Fit model
################################################################################
loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, mask=True)
model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, 
            epochs=epochs, callbacks=[model_checkpoint_callback])

################################################################################
# Evaluate model
################################################################################
print("Testing model")
loader_te = BatchLoader(dataset_te, batch_size=batch_size, mask=True)
loss = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}".format(loss))






