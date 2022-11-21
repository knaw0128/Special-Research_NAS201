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
from argparse import ArgumentParser, Namespace
from spektral.data import BatchLoader
from spektral.layers import ECCConv, GlobalSumPool
from model_NAS import ECC_Net 

################################################################################
# Config
################################################################################
# learning_rate = 1e-3    # Learning rate
# epochs = 40             # Number of training epochs
# batch_size = 8         # Batch size
def args_parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Directory to ckpt",
        default="./ckpt",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    return args

args = args_parse()
################################################################################
# Load data
################################################################################
file = open('./model_label.pkl', 'rb')
record = pickle.load(file)
file.close()
dataset = NasBench101Dataset(record_dic=record, shuffle_seed=0, start=0,
                                end=15624, inputs_shape=(None, 32, 32, 3), num_classes=10)

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

model = ECC_Net()
optimizer = tf.keras.optimizers.Adam(args.lr)
model.compile(optimizer=optimizer, loss="mse")
checkpoint_filepath = './checkpoint/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

################################################################################
# Fit model
################################################################################
loader_tr = BatchLoader(dataset_tr, batch_size=args.batch_size, mask=True)
model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, 
            epochs=args.num_epoch, callbacks=[model_checkpoint_callback])

################################################################################
# Evaluate model
################################################################################
print("Testing model")
loader_te = BatchLoader(dataset_te, batch_size=args.batch_size, mask=True)
loss = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}".format(loss))






