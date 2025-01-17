import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from run_anom import TSG2L
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
# import warnings
# warnings.filterwarnings("ignore")
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import column_or_1d
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.mcd import MCD
from pyod.models.lscp import LSCP



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='SWaT', help='The dataset name')
    parser.add_argument('--dataset_sub', default='machine-2-2', help='The dataset name')
    parser.add_argument('--size_cent', type=int, default=30, help='Batch size for the first stage')
    parser.add_argument('--size_1', type=int, default=30, help='Batch size for the second stage')
    parser.add_argument('--epoch', type=int, default=6,
                        help='Number of epochs for single-scale training in the first stage')
    parser.add_argument('--epoch_1', type=int, default=1, help='Number of epochs for the second stage')
    parser.add_argument('--gru_dime', type=int, default=150, help='GRU hidden layer dimension')
    parser.add_argument('--a_3', type=int, default=300, help='Representation dimension')
    parser.add_argument('--c', type=int, default=12,
                        help='Aggregation window length, should not be too large due to remainder')
    parser.add_argument('--pred_len', type=int, default=34, help='Prediction length')
    parser.add_argument('--pre_len', type=int, default=100, help='Downstream task data preprocessing length')
    parser.add_argument('--batch_size', type=int, default=100,)
    parser.add_argument('--p_recon', type=float, default=1,
                        help='Noise probability for the first stage, default was 1 in older versions')
    parser.add_argument('--p', type=float, default=0.5, help='Mask probability for the second stage')
    parser.add_argument('--multi', type=int, default=3, help='Number of sampling times in the first stage')
    parser.add_argument('--count', type=int, default=3, help='Must be less than epoch')
    parser.add_argument('--port', type=int, default=20, help='Number of data splits when calculating Fourier scales')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=8,
                        help='The maximum allowed number of threads used by this process')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')



    # name = 'SMD'
    # name_1 = 'machine-1-2'
    # name = 'SMAP'
    # name_1 = 'E-4'
    # name = 'MSL'
    # name_1 = 'M-2'
    name = args.dataset
    name_1=args.dataset_sub
    batchsize =args.batch_size # 640
    pre_len = args.pre_len

    if (name == 'SMD' or name == 'SMAP' or name=='MSL'):
        train_data = pd.read_csv(f'datasets/anomaly/{name}/{name_1}_train.csv', sep=',').to_numpy()
        test_data = pd.read_csv(f'datasets/anomaly/{name}/{name_1}_test.csv', sep=',').to_numpy()
        test_label = pd.read_csv(f'datasets/anomaly/{name}/{name_1}_labels.csv', sep=',').to_numpy()
        test_label = np.any(test_label == 1, axis=1).astype(int)
        s=0
    elif (name == 'Anomaly_Detection_Falling_People'):
        train_data = pd.read_csv(f'datasets/anomaly/{name}/train.csv', sep=',').to_numpy()
        test_data = pd.read_csv(f'datasets/anomaly/{name}/test.csv', sep=',').to_numpy()
        train_data = train_data[:, :-1]
        test_label = test_data[:, -1]
        test_data = test_data[:, :-1]
    else:
        train_data = pd.read_csv(f'datasets/anomaly/{name}/train.csv', sep=',').to_numpy()
        test_data = pd.read_csv(f'datasets/anomaly/{name}/test.csv', sep=',').to_numpy()
        test_label = pd.read_csv(f'datasets/anomaly/{name}/labels.csv', sep=',').to_numpy()
        test_label = np.any(test_label == 1, axis=1).astype(int)
        s=0
    data = np.concatenate((train_data, test_data), axis=0)
    #############################################################



    model =TSG2L(
        input_dims=train_data.shape[1],
        device=device,
        size_cent=args.size_cent,
        size_1=args.size_1,
        epoch=args.epoch,
        epoch_1=args.epoch_1,
        gru_dime=args.gru_dime,
        a_3=args.a_3,
        c=args.c,
        pred_len=args.pred_len,
        p_recon=args.p_recon,
        p=args.p,
        multi=args.multi,
        count=args.count,
        port=args.port

    )
    t = time.time()
    c, max_size_dict = model.fit(
        train_data
    )
    t = time.time() - t

    if(max_size_dict>pre_len) :pre_len=max_size_dict
    # if(name=='SMD'):pre_len=max_size_dict
    # print(lidu)



    out, eval_res = tasks.eval_anomaly_detection(model,pre_len, batchsize,train_data, test_data, test_label, 7,c)
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print(name)
    if (name == 'SMD' or name == 'SMAP' or name == 'MSL'):
        print(name_1)
    model.output()
    print(" 下游算法补全长度len ",pre_len)
    print(batchsize)
    # print('Evaluation result:', eval_res)
    for key, value in eval_res.items():
        print(f"{key}:  {value}")
    # print("此处为部分代码加上eval()")
