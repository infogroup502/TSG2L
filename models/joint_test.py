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

#这个备份之后立马就做离散傅里叶变换
class joint_test(nn.Module):
    def __init__(self, dimension,dimension_nihe, a_3,c,pred_len,gru_dime,count):
        super(joint_test, self).__init__()
        self.device = 'cuda'
        self.dime = dimension  # 总维度
        self.dime_nihe=dimension_nihe
        self.a_3 = a_3
        self.c=c
        self.pred_len=pred_len
        self.gru_dime=gru_dime
        self.count=count

        self.gru_cell = nn.GRUCell(1, self.gru_dime)
        self.cell_jiang = nn.Sequential(
            nn.Linear(self.gru_dime, self.gru_dime),
            # nn.BatchNorm1d(self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.gru_dime, 1),
        )
        self.jiang = nn.ModuleList(nn.Sequential(
            nn.Linear(self.a_3, self.a_3),
            nn.BatchNorm1d(self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.a_3, 1),
        ) for i in range(self.dime))
        self.gru_cell_1= nn.GRUCell(1, self.gru_dime)
        self.cell_jiang_1 = nn.Sequential(
            nn.Linear(self.gru_dime, self.gru_dime),
            # nn.BatchNorm1d(self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.gru_dime, 1),
        )
        self.jiang_1= nn.ModuleList(nn.Sequential(
            nn.Linear(self.a_3, self.a_3),
            nn.BatchNorm1d(self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.a_3, 1),
        ) for i in range(self.dime))
        #上下文部分
        self.linear_w = torch.nn.Parameter(torch.randn(self.c,  self.a_3).to(self.device).requires_grad_())
        self.linear_b = torch.nn.Parameter(torch.randn(self.c,  self.a_3).to(self.device).requires_grad_())
        self.merge=nn.Sequential(
            nn.Linear(self.count*self.dime*self.a_3, self.a_3),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(self.a_3, self.a_3),
        )
        self.merge_w=torch.nn.Parameter(torch.randn(1,self.dime*self.count,  self.a_3).to(self.device).requires_grad_())
        self.merge_gru=nn.GRU(self.a_3,self.a_3)
        self.shuzhihua = torch.nn.Parameter(
            torch.randn(self.count*self.dime*self.a_3, self.a_3).to(self.device).requires_grad_())
        nn.init.kaiming_uniform_(self.shuzhihua, a=math.sqrt(5))

    def forward(self,x_mul,flag):
        for i in range(len(x_mul)):
            if(i==0):
                x=x_mul[i]
            else:
                x=torch.cat([x,x_mul[i]],dim=1)
        # _,x=self.merge_gru(x.cuda())
        # x= torch.matmul(x.reshape(x.shape[0],-1).cuda(),self.shuzhihua)
        x=torch.sum(x.cuda()*self.merge_w,dim=1)
        # x=self.merge(x.reshape(x.shape[0],-1).cuda())
        # x=x_mul[0].cuda()
        for i in range(self.c - 1, x.shape[0]):
            temp_1 = x[i - self.c + 1:i + 1]
            # 乘上权重矩阵并且做线性变换
            temp_1 = self.linear_w * temp_1 + self.linear_b
            temp_1 = torch.sum(temp_1, dim=0).reshape(1,  -1)
            if (i == self.c - 1):
                rep = temp_1
            else:
                rep = torch.cat([rep, temp_1], dim=0)

        for i in range(self.c - 1):
            # print(i)
            temp_1 = x[0:i + 1]
            temp = x[i].repeat(self.c - (i + 1),  1)
            temp_1 = torch.cat([temp_1, temp], dim=0)
            # 对历史时间戳使用线性变换进行聚合
            temp_1 = self.linear_w * temp_1 + self.linear_b
            temp_1 = torch.sum(temp_1, dim=0).reshape(1,-1)
            if (i == 0):
                pre = temp_1
            else:
                pre = torch.cat([pre, temp_1], dim=0)
        x =torch.cat([pre, rep], dim=0)
        x = F.normalize(x, dim=-1)
        rep = x
        for i in range(self.dime):
            x_in = self.jiang[i](x)
            if (i == 0):
                x_fina = x_in
            else:
                x_fina = torch.cat([x_fina, x_in], dim=1)
        pred = self.predict(x_fina, self.gru_cell, self.cell_jiang)
        pred_1 = self.predict(x_fina, self.gru_cell_1, self.cell_jiang_1)
        return pred, pred_1, rep

    def predict(self, x_fina, gru_cell, cell_jiang):
        input = x_fina.reshape(-1, 1)
        out = torch.zeros(input.shape[0], self.gru_dime).cuda()
        # input = x_fina
        output = []
        for i in range(self.pred_len):
            out = gru_cell(input, out)
            input = cell_jiang(out)
            output.append(input.reshape(1, input.shape[0], -1))
        pred = torch.stack(output)
        pred = pred.reshape(pred.shape[0], pred.shape[2], -1)
        pred = pred.permute(1, 0, 2)
        pred = pred.reshape(x_fina.shape[0], x_fina.shape[1], -1)
        pred = pred.permute(0, 2, 1)
        # pred = pred.reshape(pred.shape[0], -1)

        return pred


    def parallel(self,x_mul,flag):
        for i in range(len(x_mul)):
            if (i == 0):
                x = x_mul[i]
            else:
                x = torch.cat([x, x_mul[i]], dim=2)
        # _,x=self.merge_gru(x.cuda())
        # x= torch.matmul(x.reshape(x.shape[0],-1).cuda(),self.shuzhihua)
        x = torch.sum(x.cuda() * self.merge_w.unsqueeze(0), dim=2)
        # x=self.merge(x.reshape(x.shape[0],-1).cuda())
        # x=x_mul[0].cuda()
        for i in range(self.c - 1, x.shape[1]):
            temp_1 = x[:,i - self.c + 1:i + 1]
            # 乘上权重矩阵并且做线性变换
            temp_1 = self.linear_w.unsqueeze(0) * temp_1 + self.linear_b.unsqueeze(0)
            temp_1 = torch.sum(temp_1, dim=1).unsqueeze(1)
            if (i == self.c - 1):
                rep = temp_1
            else:
                rep = torch.cat([rep, temp_1], dim=1)

        for i in range(self.c - 1):
            # print(i)
            temp_1 = x[:,0:i + 1]
            temp = x[:,i].unsqueeze(1).repeat(1,self.c - (i + 1), 1)
            temp_1 = torch.cat([temp_1, temp], dim=1)
            # 对历史时间戳使用线性变换进行聚合
            temp_1 = self.linear_w.unsqueeze(0) * temp_1 + self.linear_b.unsqueeze(0)
            temp_1 = torch.sum(temp_1, dim=1).unsqueeze(1)
            if (i == 0):
                pre = temp_1
            else:
                pre = torch.cat([pre, temp_1], dim=1)
        x = torch.cat([pre, rep], dim=1)
        x = F.normalize(x, dim=-1)
        rep = x

        return 0, 0, rep

