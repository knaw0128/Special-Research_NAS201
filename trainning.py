import pickle
import numpy as np
import csv
import tensorflow as tf
import os
import math
from os import path
from nas_bench_201_dataset import NasBench101Dataset
from argparse import ArgumentParser
from spektral.data import BatchLoader
from model_NAS import ECC_Net, GIN_Net 
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    parser.add_argument(
        "--dataset",
        type=str,
        default='cifar10-valid',
        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--do_train", action="store_true")
    args = parser.parse_args()
    return args

def K_rank(data): # input is an list of pair, containing prediction and true value
    n = len(data)
    ans = []

    for k in  range(1000):
        disorder = 0.0
        inputs = data.copy()
        np.random.shuffle(inputs)
        inputs = inputs[:min(100, len(data))]
        for i in range(n):
            for j in range(i+1, n):
                if  (inputs[i][0] > inputs[j][0] and inputs[i][1] < inputs[j][1]) or \
                    (inputs[i][0] < inputs[j][0] and inputs[i][1] > inputs[j][1]) :
                    disorder += 1
        ans.append(1 - 2 * disorder / (n*(n-1)/2))

    return [np.mean(ans), np.std(ans), np.max(ans), np.min(ans)]


def MAP(all): # input is an list of pair, containing prediction and true value
    ret = []
    for k in range(1000):
        inputs = all.copy()
        n = len(inputs)
        for i in range(n):
            inputs[i].append(i)
        
        np.random.shuffle(inputs)
        inputs = inputs[:min(100, n)]

        true_rank = [data[2] for data in inputs.sort(key=lambda x: x[1])]
        pred_rank = [data[2] for data in inputs.sort(key=lambda x: x[0])]

        ans = 0.0
        for i in range(1,n+1):
            now = 0.0
            for j in pred_rank:
                if j in true_rank[:i]:
                    now += 1
            ans += now / i
        ret.append( ans / n )
    return [np.mean(ret), np.std(ret), np.max(ret), np.min(ret)]

def NDCG(all): # input is an list of pair, containing prediction and true value

    ret = []
    for k in range(1000):
        inputs = all.copy()
        n = len(inputs)
        for i in range(n):
            inputs[i].append(i)
        
        np.random.shuffle(inputs)
        inputs = inputs[:min(100, n)]

        true_rank = [data[2] for data in inputs.sort(key=lambda x: x[1])]
        pred_rank = [data[2] for data in inputs.sort(key=lambda x: x[0])]

        
        idcg = 0.0
        for i in range(len(true_rank)):
            idcg += true_rank[i]/math.log2(2+i)
        
        dcg = 0.0
        for i in range(len(pred_rank)):
            dcg += pred_rank[i]/math.log2(2+i)
            
        ret.append(dcg / idcg)
    
    return [np.mean(ret), np.std(ret), np.max(ret), np.min(ret)]
    
    


args = args_parse()
################################################################################
# Load data
################################################################################
model_label_path = os.path.join('./Get_Feature/NASBENCH_201_dict', args.dataset+"_model_label.pkl")
file = open(model_label_path, 'rb')
record = pickle.load(file)
file.close()
dataset = NasBench101Dataset(record_dic=record, shuffle_seed=0, start=0,
                                end=156, inputs_shape=(None, 32, 32, 3), num_classes=10, dataset_name=args.dataset)

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
np.random.seed(114514)
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

print("Loading")
model = GIN_Net()
print("Finish")
optimizer = tf.keras.optimizers.Adam(args.lr)
model.compile(optimizer=optimizer, loss="mse")
checkpoint_filepath = './checkpoint/'+args.dataset+"-"+str(args.num_epoch)+"epo"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

################################################################################
# Fit model
################################################################################
if args.do_train:
    loader_tr = BatchLoader(dataset_tr, batch_size=args.batch_size, mask=True)
    model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, 
                epochs=args.num_epoch, callbacks=[model_checkpoint_callback])

# ################################################################################
# # Evaluate model
# ################################################################################
model.load_weights(checkpoint_filepath).expect_partial()
print("Testing model")
loader_te = BatchLoader(dataset_te, batch_size=1, mask=True)
loss = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}".format(loss))

loader_me = BatchLoader(dataset_te, batch_size=1, epochs=1, mask=True, shuffle=False)

metric_input = []
progress = tqdm(total=len(dataset_te))
for i,data in enumerate(loader_me.load()):
    
    # print(data[0])
    # print(all_label[i])
    # print(all_pred[i])

    # print(model.predict(data[0])[0][0])
    # print(data[1][0][0])
    metric_input.append([model.predict(data[0], verbose=0), data[1]])
    progress.update(1)
    # assert(0)
print(K_rank(metric_input))
print(MAP(metric_input))
print(NDCG(metric_input))




