import torch
from torch import nn
import numpy as np
from torch.nn import Parameter, ParameterList
# import tensorly as tl
# tl.set_backend('pytorch')

import matplotlib.pyplot as plt

import math
torch.pi=math.pi
torch.e=math.e
torch.inf=math.inf
torch.nan=math.nan
import torch.fft as fft

from models.weight_matrix import w_cheng,w_jia

torch.backends.cudnn.enabled = False

import torch.nn.functional as F



def generate_binomial_mask(B, T, p):
    return torch.from_numpy(np.random.binomial(1, p, size=(B,T ))).to(torch.bool)

#这个备份之后立马就做离散傅里叶变换
class joint(nn.Module):
    def __init__(self, dimension,dimension_nihe, a_3,pred_len,p,c,n_covars,gru_dime,multi,count):
        super(joint, self).__init__()
        self.device = 'cuda'
        self.dime = dimension  # 总维度
        self.dime_nihe=dimension_nihe
        self.pred_len=pred_len
        self.p=p
        self.c=c
        # self.n_covars=self.dime-self.dime_nihe
        self.n_covars = n_covars
        self.gru_dime=gru_dime
        self.multi=multi

        self.a_3 = a_3
        self.count=count
        # 制造关于原始数据的映射

        self.gru_enc = nn.ModuleList(nn.GRU(1, self.a_3)for i in range(self.count))
        self.gru_den =nn.ModuleList( nn.GRU(self.a_3, self.gru_dime)for i in range(self.count))
        self.den_jiang =nn.ModuleList( nn.Sequential(
            nn.Linear(self.gru_dime, self.gru_dime),
            # nn.BatchNorm1d(self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.gru_dime, 1),
        )for i in range(self.count))

    def forward(self, x,label):  # x: B x T x input_dims
        # if(label>=self.count): label=self.count-1
        x_in = torch.from_numpy(x.astype(np.float32)).cuda()

        x_in,_=self.gru_enc[label](x_in)

        y, _ = self.gru_den[label](torch.flip(x_in, [0]))
        y = torch.flip(y, [0])

        y = self.den_jiang[label](y)
        y=y.squeeze(-1)


        return y,x_in

    def parallel(self, x,label):  # x: B x T x input_dims
        x_in = torch.from_numpy(x.astype(np.float32)).cuda()

        batch_size=x.shape[0]
        x_in=x_in.permute(1,0,2)
        x_in=x_in.reshape(x_in.shape[0],-1,1)
        x_in,_=self.gru_enc[label](x_in)
        # x_in=x_in.reshape(x_in.shape[0],batch_size,-1,self.a_3)



        return 0,x_in

