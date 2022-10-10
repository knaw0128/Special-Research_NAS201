import logging
import random
from spektral.data import Dataset, Graph
import pickle
import numpy as np
import csv
from model_spec import ModelSpec
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import model_util
from classifier import Classifier
import os
import wget
from keras import backend as K
from os import path
import re
import hashlib
import model_builder
from nas_bench_201_dataset import get_model_by_id_and_layer_original
os.environ['TORCH_HOME'] = "D:\Assignment_of_Computrer_Science\CCF_Project\David_NAS_project"

file = open('./model_label.pkl', 'rb')
record = pickle.load(file)
file.close()
random.seed(0)
random.shuffle(record)

id=0
matrix, ops = record[id][0], record[id][1]
inputs_shape=(None, 32, 32, 3)
total_layers = 18

spec = ModelSpec(np.array(matrix), ops)
num_nodes = matrix.shape[0]
# get model
K.clear_session()
model = tf.keras.Sequential()
model.add(model_builder.CellModel(spec,
                    inputs_shape=tuple(inputs_shape),
                    channels=16,
                    is_training=None))

# get profile_filename for speed up get_flops

# restart layer counter
K.clear_session()















