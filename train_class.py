import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from run_class import TSG2L
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
# import warnings
# warnings.filterwarnings("ignore")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='SharePriceIncrease', help='The dataset name')
    parser.add_argument('--size_cent', type=int, default=18, help='Batch size for the first stage')
    parser.add_argument('--size_1', type=int, default=18, help='Batch size for the second stage')
    parser.add_argument('--epoch', type=int, default=5,
                        help='Number of epochs for single-scale training in the first stage')
    parser.add_argument('--epoch_1', type=int, default=8, help='Number of epochs for the second stage')
    parser.add_argument('--gru_dime', type=int, default=200, help='GRU hidden layer dimension')
    parser.add_argument('--a_3', type=int, default=800, help='Representation dimension')
    parser.add_argument('--c', type=int, default=4,
                        help='Aggregation window length, should not be too large due to remainder')
    parser.add_argument('--pred_len', type=int, default=10, help='Prediction length')
    parser.add_argument('--p_recon', type=float, default=1,
                        help='Noise probability for the first stage, default was 1 in older versions')
    parser.add_argument('--p', type=float, default=0.1, help='Mask probability for the second stage')
    parser.add_argument('--multi', type=int, default=50, help='Number of sampling times in the first stage')
    parser.add_argument('--count', type=int, default=3, help='Must be less than epoch')
    parser.add_argument('--port', type=int, default=2, help='Number of data splits when calculating Fourier scales')
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
    task_type = 'classification'
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)


    print('done')


    data = np.concatenate((train_data, test_data), axis=0)
    data = data.reshape(data.shape[0], data.shape[1])

    model = TSG2L(
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
    n_covariate_cols = model.fit(
        train_data
    )
    print(args.dataset)
    model.output()
    # print(lidu)

    t = time.time() - t
    # print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")


    if task_type == 'classification':
            result = tasks.eval_classification(model, data, train_data, train_labels, test_data, test_labels,
                                               eval_protocol='svm')



    print(result)
