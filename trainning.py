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
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

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
    parser.add_argument(
        "--loss",
        type=str,
        default='mse',
        choices=[   
                    'contrastive_loss', 'triplet_hard_loss', 'triplet_semihard_loss', 
                    'npairs_loss', 'npairs_multilabel_loss', 'mse'
                ],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=99999)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--do_train", action="store_true")
    args = parser.parse_args()
    return args

# Loss(x, batch[]) =  abs ( (output(x) - output(batch[])) /(label(x) - label(batch[])) -1 )
# batch[] is a random sample a data from the batch
def K_rank(data): # input is an list of pair, containing prediction and true value
    n = len(data)
    ans = []

    for k in  range(1000):
        disorder = 0.0
        inputs = data.copy()
        np.random.shuffle(inputs)
        inputs = inputs[:min(100, len(data))]
        pred_list = [data[0] for data in inputs]
        true_list = [data[1] for data in inputs]
        kt, p = kendalltau(pred_list, true_list)
        ans.append(kt)

    return [np.mean(ans), np.std(ans), np.max(ans), np.min(ans)]

def MAP(data, K): # input is an list of pair, containing prediction and true value
    ret = []
    for k in range(1000):
        inputs = data.copy()
        n = len(data)
        for i in range(n):
            inputs[i].append(i)
        
        np.random.shuffle(inputs)
        inputs = inputs[:min(100, n)]

        inputs.sort(key=lambda x: x[1])
        true_rank = [data[2] for data in inputs]
        inputs.sort(key=lambda x: x[0])
        pred_rank = [data[2] for data in inputs]

        ans = 0.0
        now = 0.0
        for i, j in enumerate(pred_rank):
            if j in true_rank[:min(n, K)]:
                now += 1
                ans += now / (i+1)
        ret.append( ans / now )
    return [np.mean(ret), np.std(ret), np.max(ret), np.min(ret)]

def NDCG(all): # input is an list of pair, containing prediction and true value

    ret = []
    def dcg_score(y_true, y_score):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order)

        gains = y_true
        # highest rank is 1 so +2 instead of +1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)
    for k in range(1000):
        inputs = all.copy()
        np.random.shuffle(inputs)
        inputs = inputs[:min(100, len(inputs))]
        y_true = np.array([data[0] for data in inputs])
        y_score = np.array([data[1] for data in inputs])

        best = dcg_score(y_true, y_true)
        actual = dcg_score(y_true, y_score)
        ret.append(actual / best)
    
    return [np.mean(ret), np.std(ret), np.max(ret), np.min(ret)]
    
def MAE(inputs): # input is an list of pair, containing prediction and true value

    ret = []
    for i in inputs:
        ret.append(abs(i[0]-i[1]))
    
    return np.mean(ret)

def MSE(inputs): # input is an list of pair, containing prediction and true value

    ret = []
    for i in inputs:
        ret.append((i[0]-i[1])**2)
    
    return np.mean(ret)



args = args_parse()
LossMenu =  {    
                'mse' : 'mse',
                'contrastive_loss' : tfa.losses.ContrastiveLoss(), 
                'triplet_hard_loss' : tfa.losses.TripletHardLoss(), 
                'triplet_semihard_loss' : tfa.losses.TripletSemiHardLoss(), 
                'npairs_loss' : tfa.losses.NpairsLoss(), 
                'npairs_multilabel_loss' : tfa.losses.NpairsMultilabelLoss(),
            }
################################################################################
# Load data
################################################################################
model_label_path = os.path.join('./Get_Feature/NASBENCH_201_dict', args.dataset+"_model_label.pkl")
file = open(model_label_path, 'rb')
record = pickle.load(file)
file.close()
dataset = NasBench101Dataset(record_dic=record, shuffle_seed=0, start=0,
                                end=15624, inputs_shape=(None, 32, 32, 3), num_classes=10, dataset_name=args.dataset)

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
np.random.seed(114514)
idxs = np.random.permutation(len(dataset))
split = [ 1000*i for i in range(1,14) ]
# split = int( 0.9 * len(dataset))
# idx_all = np.split(idxs, split)
idx_all = np.split(idxs, split)
idx_trs = idx_all[:13]
idx_te = idx_all[13]

################################################################################
# Fit model
################################################################################

K_ranks = []
NDCGs   = []
MAPs    = []

now_idx = np.array([],dtype="int64")
for dataset_len in range(13):
    now_idx = np.concatenate((now_idx, idx_trs[dataset_len]),dtype=idx_trs[dataset_len].dtype)
    dataset_tr = dataset[now_idx]
    loader_tr = BatchLoader(dataset_tr, batch_size=args.batch_size, mask=True)
    
    model = GIN_Net()
    optimizer = tf.keras.optimizers.Adam(args.lr)
    model.compile(optimizer=optimizer, loss = LossMenu[args.loss])
    checkpoint_filepath = './checkpoint/'+args.dataset+"_datasize_"+str(1000*(dataset_len+1))

    if args.loss != 'mse':
        checkpoint_filepath += "_loss_" + args.loss

    if args.do_train:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=25,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=False,
        )
        model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, 
                    epochs=args.num_epoch, callbacks=[model_checkpoint_callback, early_stop_callback])

    ######################################################################
    # Evaluating Model & Making Prediction
    ######################################################################
    dataset_te = dataset[idx_te]
    loader_te = BatchLoader(dataset_te, batch_size=args.batch_size, mask=True)
    model.load_weights(checkpoint_filepath).expect_partial()
    print("Testing model")
    loss = model.evaluate(loader_te.load(), steps = loader_te.steps_per_epoch)
    loader_me = BatchLoader(dataset_te, batch_size = 1, epochs=1, mask=True, shuffle=False)
    metric_input = []
    progress = tqdm(total = len(dataset_te))
    for data in loader_me.load():
        metric_input.append([model.predict(data[0], verbose=0)[0][0], data[1][0][0]])
        progress.update(1)

    print("K rank : ")
    K_ranks.append(K_rank(metric_input))
    print(K_ranks[-1])
    print("MAP@10 : ")
    MAPs.append(MAP(metric_input, 10))
    print(MAPs[-1])
    print("NDCG : ")
    NDCGs.append(NDCG(metric_input))
    print(NDCGs[-1])

train_size = [ i*1000 for i in range(1, 14) ]
# plot K_rank
plt.figure(1)
plt.plot(train_size, [data[0] for data in K_ranks])
plt.plot(train_size, [data[1] for data in K_ranks])
plt.plot(train_size, [data[2] for data in K_ranks])
plt.plot(train_size, [data[3] for data in K_ranks])
plt.savefig(f"K_rank_{args.dataset}_{args.loss}.png")

plt.figure(2)
plt.plot(train_size, [data[0] for data in MAPs])
plt.plot(train_size, [data[1] for data in MAPs])
plt.plot(train_size, [data[2] for data in MAPs])
plt.plot(train_size, [data[3] for data in MAPs])
plt.savefig(f"MAP_{args.dataset}_{args.loss}.png")

plt.figure(3)
plt.plot(train_size, [data[0] for data in NDCGs])
plt.plot(train_size, [data[1] for data in NDCGs])
plt.plot(train_size, [data[2] for data in NDCGs])
plt.plot(train_size, [data[3] for data in NDCGs])
plt.savefig(f"NDCG_{args.dataset}_{args.loss}.png")







