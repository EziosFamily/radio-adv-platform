import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_loader.base_model import BaseModel

class Based_GRU_18(BaseModel):
    def __init__(self, output_dim):
        super(Based_GRU_18, self).__init__()
        # input(batch, 1, 2, 1024)
        # after shape(batch, 1024, 2)
        self.gru1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.GRU(input_size= 2, hidden_size= 100, num_layers=3, batch_first= True)
        )
        self.relu1 = nn.ReLU()
        # after gru1(batch, 1024, 100)
        self.gru2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.GRU(input_size= 100, hidden_size= 100, num_layers=3, batch_first= True)
        )
        self.relu2 = nn.ReLU()
        self.gru3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.GRU(input_size= 100, hidden_size= 300, num_layers=3, batch_first= True)
        )
        self.relu3 = nn.ReLU()
        # afer gru2(batch, 1024, 100)
        # 取最后一个单元的输出(batch, 1, 100)
        # after shape(batch, 100)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 300, out_features= 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 64, out_features= output_dim)
        )

    def forward(self, x):
        x = x.view(x.shape[0], 2, 1024)
        x = x.transpose(1,2)
        x,_ = self.gru1(x)
        x = self.relu1(x)
        x,_ = self.gru2(x)
        x = self.relu2(x)
        # print('conv1', x.shape)
        x,_ = self.gru3(x)
        x = self.relu3(x)[:,-1,:]
        # print('gru2', x.shape)
        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def loadBased_GRU_18(filepath):
    """[加载LeNet网络模型]

    Args:
        filepath ([str]): [LeNet的预训练模型所在的位置]

    Returns:
        [type]: [返回一个预训练的LeNet]
    """
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model = Based_GRU_18(output_dim = 24)
    model.load_state_dict(checkpoint['state_dict'])  # 加载网络权重参数
    return model