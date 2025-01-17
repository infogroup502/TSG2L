import torch
import numpy as np
import argparse
import time
import datetime
from run_forecast import TSG2L
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import warnings
# warnings.filterwarnings("ignore")


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='airquality', help='The dataset name')
    parser.add_argument('--size_cent', type=int, default=576, help='Batch size for the first stage')
    parser.add_argument('--size_1', type=int, default=576, help='Batch size for the second stage')
    parser.add_argument('--epoch', type=int, default=5,
                        help='Number of epochs for single-scale training in the first stage')
    parser.add_argument('--epoch_1', type=int, default=10, help='Number of epochs for the second stage')
    parser.add_argument('--gru_dime', type=int, default=40, help='GRU hidden layer dimension')
    parser.add_argument('--a_3', type=int, default=100, help='Representation dimension')
    parser.add_argument('--c', type=int, default=14,
                        help='Aggregation window length, should not be too large due to remainder')
    parser.add_argument('--pred_len', type=int, default=200, help='Prediction length')
    parser.add_argument('--p_recon', type=float, default=1,
                        help='Noise probability for the first stage, default was 1 in older versions')
    parser.add_argument('--p', type=float, default=0.6, help='Mask probability for the second stage')
    parser.add_argument('--multi', type=int, default=20, help='Number of sampling times in the first stage')
    parser.add_argument('--count', type=int, default=3, help='Scale number,Must be less than epoch')
    parser.add_argument('--port', type=int, default=6, help='Number of data splits when calculating Fourier scales')
    parser.add_argument('--run_name',default='forecast_csv',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default='forecast_csv',
                        help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
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
    if args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset)
        train_data = data[:, train_slice]

    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset, univar=True)
        train_data = data[:, train_slice]
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    print('done')


    t = time.time()

    data_train = data[:, train_slice]
    data_test = data[:, test_slice]
    model = TSG2L(
        input_dims=data_train.shape[2],
        device=device,
        n_covars=n_covariate_cols,
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
    model.output()
    ep = model.fit(
        data_train,data_test
    )

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print(args.dataset)
    model.output()

    out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens,
                                                   n_covariate_cols)
