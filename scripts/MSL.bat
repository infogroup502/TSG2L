python  train_anom.py     --dataset MSL  --dataset_sub  D-15  --size_cent  24 --size_1 24  --epoch  4   --epoch_1  1  --gru_dime   180  --a_3  300  --c  14  --pred_len  62   --batch_size  128  --p_recon 1  --p  0.1  --multi  2 --count  3  --port  100  --gpu 0   --seed 42  --max-threads  8



python  train_anom.py     --dataset MSL  --dataset_sub  F-4  --size_cent  24 --size_1 24  --epoch  4   --epoch_1  1  --gru_dime   210  --a_3  300  --c  14  --pred_len  25   --batch_size  128  --p_recon 0.3  --p  0.1  --multi  5 --count  3  --port  110  --gpu 0   --seed 42  --max-threads  8



python  train_anom.py     --dataset MSL  --dataset_sub  M-2  --size_cent  24 --size_1 24  --epoch  4   --epoch_1  1  --gru_dime   240  --a_3  300  --c  14  --pred_len  25   --batch_size  128  --p_recon 0.1  --p  0.1  --multi  5 --count  3  --port  80  --gpu 0   --seed 42  --max-threads  8