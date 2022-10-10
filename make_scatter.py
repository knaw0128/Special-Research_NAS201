import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model_spec import ModelSpec
from nas_bench_201_dataset import get_params
from model_builder import CellModel



os.environ['TORCH_HOME'] = "D:\Assignment_of_Computrer_Science\CCF_Project\David_NAS_project"
dict_path=os.path.join(os.environ['TORCH_HOME'],'NASBENCH_201_dict')
file_path=os.path.join(dict_path,'model_label.pkl')
with open(file_path,'rb') as file:
    all_data = pickle.load(file)

idx = np.arange(0, 5001, dtype=int)
train_acc_data=[]
for i in all_data[0:5001] :
    train_acc_data.append(i[2]['train_accuracy'])
# plt.scatter(idx, train_acc_data, alpha=0.3)
# plt.xlabel('Cell index')
# plt.ylabel('training accuracy')
# plt.show()

inputs_shape = (None, 32, 32, 3)
param_data = []
for i in idx:
    spec = ModelSpec(np.array(all_data[i][0]), all_data[i][1])
    shape = list(inputs_shape)
    shape[3]=16
    model = tf.keras.Sequential()
    model.add(CellModel(spec,
                inputs_shape=tuple(shape),
                channels=16,
                is_training=None))
    model.build([*inputs_shape])
    param_data.append(get_params(model.layers[0]))

plt.scatter(param_data, train_acc_data, alpha=0.3)
plt.xlabel('Parameter Size')
plt.ylabel('training accuracy')
plt.show()
