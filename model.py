import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP,global_mean_pool, global_max_pool, global_add_pool, SAGPooling


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels)
        self.prelu = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv(x, edge_index)
        x1 = self.prelu(x1)
        return x1


class SAGE(torch.nn.Module):
    def __init__(self, data_name, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, conv_layer, norm_type="none"):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        elif self.norm_type == "layer":
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(conv_layer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_channels))
        self.convs.append(conv_layer(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        # import ipdb; ipdb.set_trace()
        for l, conv in enumerate(self.convs[:-1]): #conv 是当前层的卷积层
            x = conv(x, adj_t)
            if self.norm_type != "none":
                    x = self.norms[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x




class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:#如果只有一层，那么就直接添加一个从输入维度到输出维度的线性层
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1: #检查当前层是否是最后一层
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):  # ratio表示池化操作中保留节点的比例
        super(Pool, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)  # SAGPooling是一种图自适应池化操作，用于对输入数据进行图池化操作，根据节点的重要性选择保留的节点
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)

    def forward(self, x, edge, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)  # global_mean_pool函数对特征进行全局平均池化
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)  # 再使用global_mean_pool函数对x1进行全局平均池化


# 计算输入数据之间得分或相似度的模块，通过逐元素相乘和矩阵乘法操作，将两个输入数据进行组合并进行非线性变换，最终输出得分或相似度的结果
class Scorer(nn.Module):
    def __init__(self, hidden_size):
        super(Scorer, self).__init__()  # self.weight是一个隐藏层大小乘以隐藏层大小的参数矩阵
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

    def forward(self, input1, input2):
        output = torch.sigmoid(torch.sum(input1 * torch.matmul(input2, self.weight), dim=-1))
        return output



class LinkPredictor(torch.nn.Module): #用于预测两个节点之间是否存在链接。这个模型的主要特点是它可以使用不同的预测器，如'mlp'（多层感知机）或'inner'（内积），来进行预测
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList() #首先添加一个从输入通道到隐藏层通道的线性层，然后添加若干个隐藏层，最后添加一个从隐藏层通道到输出通道的线性层。
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))#每个线性层都是一个Linear对象，它表示一个全连接层，包含一个可学习的权重矩阵和一个可学习的偏置向量
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)

        return torch.sigmoid(x)
