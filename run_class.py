import torch
from torch.utils.data import TensorDataset, DataLoader
from models import joint,joint_test
import numpy as np


# import keras
torch.backends.cudnn.enabled = False
torch.set_printoptions(precision=8)
import matplotlib.pyplot as plt



#三个个尺度
class TSG2L:
    def __init__(
            self,
            input_dims,
            device, size_cent, size_1, epoch, epoch_1, gru_dime, a_3, c, pred_len, p_recon, p,
            multi, count, port,
    ):

        super().__init__()
        self.device = 'cuda'
        self.lr =0.001

        # s=24*24
        self.input_dims = input_dims
        self.device = device

        self.size_cent = size_cent
        self.size_1 = size_1
        self.epoch = epoch
        self.epoch_1 = epoch_1
        self.gru_dime = gru_dime
        self.a_3 = a_3
        self.c = c
        self.pred_len = pred_len
        self.p_recon = p_recon
        self.p = p
        self.multi = multi
        self.count = count
        self.port = port


        self.n_covars =0
        self.max_len=1000
        self.dime=input_dims
        self.n_convars =0
        self.list_model = []
        #sin模块处定义

        self.epo =self.epoch+self.epoch_1
        self.model=joint(dimension=input_dims,dimension_nihe=input_dims-self.n_convars,a_3=self.a_3,pred_len=self.pred_len,p=self.p,
                         c=self.c,n_covars=self.n_convars,gru_dime=self.gru_dime,multi=self.multi,count=self.count).to(self.device)
        self.model_1= joint_test(dimension=input_dims, dimension_nihe=input_dims - self.n_convars, a_3=self.a_3,
                           pred_len=self.pred_len,  c=self.c, gru_dime=self.gru_dime,count=self.count).to(self.device)

        self.n_epochs = 0
        self.n_iters = 0
        self.input_dims=input_dims

    def fit(self, data):
        data = data.reshape(data.shape[0], data.shape[1])

        ############top  k处理
        data_train = torch.from_numpy(data).to(torch.float).squeeze(0)
        remain = data_train.shape[0] % self.port
        if (remain != 0):
            data_train = data_train[0:-remain]
        data_train = data_train.reshape(self.port, -1, data_train.shape[1])

        xf = torch.fft.rfft(data_train[:, :, self.n_convars:], dim=0)
        frequency_list = abs(xf)
        period = frequency_list.detach().cpu().numpy()
        period =np.mean(np.mean(period, axis=-1),axis=0)
        period = data_train.shape[1] // period
        period = np.unique(np.sort(period, axis=-1)[1:-1])
        temp = np.sort(period, axis=-1)

        self.size_dict=[]
        for i in range(period.shape[0]):
            if (int(period[i]) > self.c):
                self.size_dict.append(int(period[i]))

        ##################第一阶段训练##############################
        #########################################################
        epoch = 0
        pred_len = self.pred_len

        data_y = np.stack([data[i:1 + data.shape[0] + i - pred_len, ] for i in range(pred_len)], axis=1)
        data_y = data_y[pred_len - 1:]

        data_z = np.stack([data[pred_len - i - 1:data.shape[0] - i, ] for i in range(pred_len)], axis=1)
        data_z = data_z[:-(pred_len - 1)]
        data_x = data[pred_len - 1:-(pred_len - 1)]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        selected_dict = []
        self.unit = len(self.size_dict) // self.count
        for i in range(1, self.count + 1):
            selected_dict.append(i * self.unit)
        self.size_dict = selected_dict
        for e in range(self.count):
            for i in range(self.epoch):
                loss = 0
                itr = 0
                index = 0

                # self.size=self.size_cent
                self.size = self.size_dict[e]

                if (data_x.shape[0] % self.size != 0):
                    remain = data_x.shape[0] % self.size
                    input_x = data_x[0:-remain]
                else:
                    input_x = data_x
                input_x = torch.from_numpy(input_x.reshape(self.size, -1, data_x.shape[1])).to(torch.float).permute(1,
                                                                                                                    0,
                                                                                                                    2)
                indices = torch.randperm(input_x.size(1))
                input_x = input_x[:, indices]

                for k in range(input_x.shape[0]):
                    x = input_x[k]
                    window_length = self.size
                    #############################
                    x = x.reshape(-1, 1)
                    n_samples = self.multi
                    sampled_tensor = torch.zeros((window_length, n_samples, 1))
                    for j in range(n_samples):
                        if (i == 2):
                            s = 0
                        start_index = torch.randint(0, x.shape[0] - window_length + 1, (1,)).item()
                        sample = x[start_index:start_index + window_length]
                        #######################################
                        # sampled_tensor[:, j, 0] = sample.ravel()
                        ########################################
                        noise = torch.randn(sample.shape)

                        #########新加入代码
                        mask = torch.bernoulli(torch.full_like(noise, self.p_recon))
                        noise = noise * mask

                        sample_with_noise = sample + noise
                        sampled_tensor[:, j, 0] = sample_with_noise.ravel()

                    x = sampled_tensor
                    #################################
                    # x = x.unsqueeze(-1)

                    optimizer.zero_grad()
                    x_true = x.squeeze(-1).cuda()
                    x = x.numpy()

                    output_1, repr = self.model(x, e)

                    loss_1 = 0
                    mse_1 = torch.nn.MSELoss()
                    loss_1 += mse_1(output_1, x_true)
                    loss_1 = loss_1.requires_grad_()
                    loss_1.backward()

                    loss += loss_1
                    itr += 1

                    optimizer.step()

                    torch.cuda.empty_cache()
                print(' trian_1:', i, ' / ', self.epo, ' loss ', loss.item() / itr)
        ##################第二阶段训练##############################
        #########################################################
        self.rep=[]
        for i in range(self.count):
            loss = 0
            itr = 0
            index = 0


            self.size = self.size_dict[i]
            # self.size=self.size_cent

            flag=False
            if (data_x.shape[0] % self.size != 0):
                pad = self.size-data_x.shape[0] % self.size
                x_pad=torch.from_numpy(data_x[-1]).unsqueeze(0).repeat(pad,1)
                input_x = torch.cat([torch.from_numpy(data_x),x_pad],dim=0)
                flag=True
            else:
                input_x = torch.from_numpy(data_x)
            input_x = input_x.reshape( self.size, -1,data_x.shape[1]).to(torch.float).permute(1, 0, 2)


            for k in range(input_x.shape[0]):
                x=input_x[k]
                window_length = self.size
                x=x.reshape(-1,1)
                x = x.numpy()
                output_1, repr = self.model(x, i)

                repr = repr.reshape(-1, self.dime, repr.shape[1])

                torch.cuda.empty_cache()
                if(k==0):
                    featu=repr.unsqueeze(0).detach().cpu()
                else:
                    featu=torch.cat([featu,repr.unsqueeze(0).detach().cpu()],dim=0)
            featu = featu.permute(1, 0, 2, 3)
            featu = featu.reshape(-1, featu.shape[2], featu.shape[3])
            if(flag==True):
                self.rep.append(featu[0:-pad])
            else:
                self.rep.append(featu)
        #########################第二阶段开始训练
        data_y=torch.from_numpy(data_y).to(torch.float)
        data_z=torch.from_numpy(data_z).to(torch.float)
        optimizer = torch.optim.Adam(self.model_1.parameters(), lr=self.lr)
        self.model_1.train()
        for i in range(self.epoch_1):
            loss = 0
            itr = 0
            index = 0
            ############第一个尺度
            #################
            for index in  range(0,self.rep[0].shape[0],self.size_1):

                optimizer.zero_grad()

                x_true=[]
                for j in range(len(self.rep)):
                    x_true.append(self.rep[j][index:index+self.size_1])
                y = data_y[index:index+self.size_1].cuda()  # 多变量
                z = data_z[index:index+self.size_1].cuda()

                flag = False
                if (index + self.size_1 > self.rep[0].shape[0]):
                    pad = index + self.size_1 - self.rep[0].shape[0]
                    for j in range(len(self.rep)):
                        x_pad = x_true[j][-1].unsqueeze(0).repeat(pad, 1,1)
                        x_true[j] = torch.cat([ x_true[j], x_pad], dim=0)
                    flag = True

                output_1,output_2, _ = self.model_1(x_true, True)

                if(flag==True):
                    output_1=output_1[0:-pad]
                    output_2 = output_2[0:-pad]
                #####################################
                drop_mask = torch.rand(output_1.size(1), device=output_1.device) > self.p
                if not drop_mask.any():
                    drop_mask[0] = True
                output_1 = output_1[:, drop_mask, :]
                y = y[:, drop_mask, :]
                #####################################
                drop_mask = torch.rand(output_2.size(1), device=output_2.device) > self.p
                if not drop_mask.any():
                    drop_mask[0] = True
                output_2 = output_2[:, drop_mask, :]
                z = z[:, drop_mask, :]
                ######################################
                loss_1 = 0
                mse_1 = torch.nn.MSELoss()
                loss_1 += mse_1(output_1,y)
                loss_1 += mse_1(output_2,z)
                loss_1 = loss_1.requires_grad_()
                loss_1.backward()

                loss += loss_1
                itr += 1

                optimizer.step()


                torch.cuda.empty_cache()
            print(' trian_2 :', i+self.epoch, ' / ', self.epo, ' loss ', loss.item() / itr)
        return 0

    def encode(self, data):
        data = data.reshape(data.shape[0],data.shape[1])
        self.model.eval()
        self.model_1.eval()

        self.rep = []
        for i in range(self.count):
            loss = 0
            itr = 0
            index = 0

            self.size = self.size_dict[i ]
            # self.size = self.size_cent

            flag=False
            if (data.shape[0] % self.size != 0):
                pad = self.size - data.shape[0] % self.size
                x_pad = torch.from_numpy(data[-1]).unsqueeze(0).repeat(pad, 1)
                input_x = torch.cat([torch.from_numpy(data), x_pad], dim=0)
                flag=True
            else:
                input_x = torch.from_numpy(data)
            input_x = input_x.reshape(self.size, -1, data.shape[1]).to(torch.float).permute(1, 0, 2)


            for k in range(input_x.shape[0]):
                x = input_x[k]

                x=x.reshape(-1,1)
                x = x.numpy()
                output_1, repr = self.model(x, i)
                repr = repr.reshape(-1, self.dime, repr.shape[1])
                torch.cuda.empty_cache()
                if (k == 0):
                    featu = repr.unsqueeze(0).detach().cpu()
                else:
                    featu = torch.cat([featu, repr.unsqueeze(0).detach().cpu()], dim=0)
            featu=featu.permute(1,0,2,3)
            featu=featu.reshape(-1,featu.shape[2],featu.shape[3])
            if(flag==True):
                self.rep.append(featu[0:-pad])
            else:
                self.rep.append(featu)
        max_len = self.size_1
        for index in range(0, self.rep[0].shape[0], max_len):

            x_input = []
            for j in range(len(self.rep)):
                x_input.append(self.rep[j][index:index + self.size_1])

            flag = False
            if (index + self.size_1 > self.rep[0].shape[0]):
                pad = index + self.size_1 - self.rep[0].shape[0]
                for j in range(len(self.rep)):
                    x_pad = x_input[j][-1].unsqueeze(0).repeat(pad, 1, 1)
                    x_input[j] = torch.cat([x_input[j], x_pad], dim=0)
                flag = True
            _,_, rep = self.model_1(x_input, False)
            if(flag==True):
                rep=rep[0:-pad]
            if (index == 0):
                featu_1 = rep.cpu().detach().numpy()
            else:
                featu_1= np.concatenate([featu_1,rep.cpu().detach().numpy()], axis=0)


        return featu_1

    def select_numbers(self,size_dict, size, num_count):
        closest_index = None
        min_diff = float('inf')

        # 找到最接近size的数字的索引
        for i, num in enumerate(size_dict):
            diff = abs(size - num)
            if diff < min_diff:
                min_diff = diff
                closest_index = i

        # 以最接近的数字为中心，向左右等范围扩张，选择num个整数
        selected_numbers = []
        left = max(0, closest_index - num_count// 2)
        right = min(len(size_dict), closest_index + num_count // 2 + 1)
        selected_numbers.extend(size_dict[left:right])

        # 如果选取的数量不足num个，则向右扩张
        while len(selected_numbers) < num_count and right < len(size_dict):
            selected_numbers.append(size_dict[right])
            right += 1

        # 如果选取的数量仍然不足num个，则向左扩张
        while len(selected_numbers) < num_count and left > 0:
            left -= 1
            selected_numbers.insert(0, size_dict[left])

        return selected_numbers

    def output(self):
        print(' size_cent ',self.size_cent,' size_1 ',self.size_1,' epoch ',self.epoch,' epoch_1 ',self.epoch_1,' gru_dime ',self.gru_dime)
        print(' a_3: ',self.a_3,' c ',self.c,' pred_len ',self.pred_len,' p ',self.p,' multi ',self.multi,' count',self.count,' port',self.port)
        print(' self.p_recon ', self.p_recon)
