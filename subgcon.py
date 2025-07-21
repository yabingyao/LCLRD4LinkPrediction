import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression

EPS = 1e-15


class SugbCon(torch.nn.Module):

    def __init__(self, hidden_size, gcn, pool, scorer):
        super(SugbCon, self).__init__()
        self.gcn = gcn
        self.hidden_size = hidden_size
        self.pool = pool
        self.scorer = scorer
        self.marginloss = nn.MarginRankingLoss(0.5)  # 使用 margin 为 0.5
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.scorer)
        reset(self.gcn)
        reset(self.pool)

    def forward(self, x, edge_index, batch=None, index=None):
        r""" Return node and subgraph representations of each node before and after being shuffled """
        # 使用self.encoder对输入特征x和边索引edge_index进行编码，得到隐藏表示hidden
        h = self.gcn(x, edge_index)
        # 如果index为空，说明不需要返回特定节点的表示，直接将整个隐藏表示hidden作为结果返回
        if index is None:
            return h
        # 如果index不为空，说明需要返回特定节点的表示。代码通过索引操作获取中心节点的隐藏表示z
        z = h[index]
        summary = self.pool(h, edge_index, batch)
        return z, summary

    def loss(self, hidden1, summary1):
        r"""Computes the margin objective."""
        # 使用torch.randperm函数生成一个随机排列的索引shuf_index，用于对summary1的样本顺序进行打乱
        shuf_index = torch.randperm(summary1.size(0))
        # shuf_index对hidden1和summary1进行重新排序
        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]
        # logits是通过逐元素相乘和求和的方式计算得到的
        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
        logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
        logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))

        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)  # ones表示margin(边距)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)

        return TotalLoss

    def test(self, train_z, train_y, val_z, val_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream task.v
        通过逻辑回归下游任务评估潜在空间质量"""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        val_acc = clf.score(val_z.detach().cpu().numpy(), val_y.detach().cpu().numpy())
        test_acc = clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())
        return val_acc, test_acc
