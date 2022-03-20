import torch
from Config import SEED,class_number,load_embedding,embedding_dim

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import DataSet
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self):
        super(FastText, self).__init__()
        Vocab = len(DataSet.getTEXT().vocab)  ## 已知词的数量
        Dim = embedding_dim  ##每个词向量长度
        Cla = class_number  ##类别数
        hidden_size = 128

        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        if load_embedding == "w2v":
            weight_matrix = DataSet.getTEXT().vocab.vectors
            self.embed.weight.data.copy_(weight_matrix)
        elif load_embedding == "glove":
            weight_matrix = DataSet.getTEXT().vocab.vectors
            self.embed.weight.data.copy_(weight_matrix)
        self.fc = nn.Sequential(              #序列函数
            nn.Linear(Dim, hidden_size),  #这里的意思是先经过一个线性转换层
            nn.BatchNorm1d(hidden_size),      #再进入一个BatchNorm1d
            nn.ReLU(inplace=True),            #再经过Relu激活函数
            nn.Linear(hidden_size ,Cla)#最后再经过一个线性变换
        )

    def forward(self, x):
        # [batch len, text size]
        x = self.embed(x)
        x = torch.mean(x,dim=1)
        # [batch size, Dim]
        logit = self.fc(x)
        # [batch size, Cla]
        return logit