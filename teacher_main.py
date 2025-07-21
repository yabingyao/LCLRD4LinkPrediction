import argparse, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric
import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger, ProductionLogger
from utils import Subgraph, preprocess,get_dataset, do_edge_split
from torch_geometric.nn import SAGEConv
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists
from subgcon import SugbCon
from model import GCN,SAGE,MLP,Scorer, Pool,LinkPredictor
from generate_production_split import do_production_edge_split
from sageconv_updated import SAGEConv_updated
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def CLtrain(CLmodel,subgraph,data,optimizer,sample_number,datasets,epochs):
    # Model training
    CLmodel.train()
    optimizer.zero_grad()
    sample_idx = random.sample(range(data.x.size(0)), sample_number)
    batch, index = subgraph.search(sample_idx)
    z, summary = CLmodel(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())

    clloss = CLmodel.loss(z, summary)
    clloss.backward()
    optimizer.step()
    return clloss.item()


def get_all_node_emb(CLmodel,data,hidden_size,sample_number,subgraph):
     # Obtain central node embs from subgraphs ，
     # 通过np.arange函数创建了一个从0到num_node的整数数组，并根据mask筛选出中心节点的索引。这些索引存储在node_list中
     #mask = mask.cpu()  # 将张量移动到 CPU

     #num_node = data.x.size(0)
     #node_list = np.arange(0, num_node, 1)[mask]
     # list_size = node_list.size
     # # 创建了一个大小为(list_size, args.hidden_size)的零张量z，其中list_size是中心节点的数量，args.hidden_size是嵌入表示的维度
     # z = torch.Tensor(list_size,hidden_size).cuda()
     # # group_nb，表示将中心节点分成多少组进行处理。这是通过将中心节点数量除以批次大小(args.batch_size)并向上取整得到的
     # group_nb = math.ceil(list_size / sample_number)
     # for i in range(group_nb):  # 确定当前组的起始索引(minn)和结束索引(maxx)，然后从node_list中获取相应的节点子集
     #     maxx = min(list_size, (i + 1) * sample_number)
     #     minn = i * sample_number
     #     batch, index = subgraph.search(node_list[minn:maxx])
         # 子集被传递给model进行处理，并返回节点的嵌入表示。
     num_node = data.x.size(0)
     node_list = np.arange(0, num_node, 1)
     batch, index = subgraph.search(node_list)
     node, _ = CLmodel(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
         #z[minn:maxx] = node  # 将每个组的嵌入表示存储在张量z的相应位置上，并在所有组处理完成后返回z作为结果
     return node


#split_edge-训练集、测试集、验证集
#data它表示图数据集的所有信息（节点特征、邻接矩阵、标签）
#dataset 对象来获取图数据集的名称、大小等信息
def BCEtrain(CLmodel, predictor, data, split_edge, optimizer, batch_size, datasets, transductive,
             hidden_channels,sample_number,subgraph):
    if transductive == "transductive":
        row, col = data.adj_t
        pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    else:
        row, col = data.edge_index  # edge_index 表示图的邻接矩阵
        pos_train_edge = data.edge_index.t()
    edge_index = torch.stack([col, row], dim=0)

    predictor.train()

    bce_loss = nn.BCELoss()
    total_bceloss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        h = get_all_node_emb(CLmodel,data,hidden_channels,sample_number,subgraph)

        edge = pos_train_edge[perm].t()

        if datasets != "collab":
            neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0),
                                         num_neg_samples=perm.size(0), method='dense')
        elif datasets == "collab":  # 用于生成负样本边
            neg_edge = torch.randint(0, data.x.size()[0], edge.size(), dtype=torch.long,
                                     device=h.device)

        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        bceloss = bce_loss(out, train_label)

        bceloss.backward()

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        #torch.nn.utils.clip_grad_norm_(CLmodel.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()


        num_examples = edge.size(1)  # 计算批次中的样本数
        total_bceloss += bceloss.item() * num_examples  # 将当前批次的损失添加到总损失中
        total_examples += num_examples

    return total_bceloss / total_examples


def test_transductive(CLmodel, predictor, data, split_edge, evaluator, batch_size, datasets, hidden_channels,sample_number,subgraph,args):
    CLmodel.eval()  # 将模型和预测器设置为评估模式，这意味着模型和预测器在评估过程中不会更新其参数。
    predictor.eval()

    h = get_all_node_emb(CLmodel, data, hidden_channels, sample_number, subgraph)

    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()  # 表示提取一批数据中的所有边，并转置
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]],h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    if datasets != "collab":
        for K in [10, 20, 30, 50]:
            evaluator.K = K
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']
            # 包含验证和测试数据集 K 排名下的命中次数的元组
            results[f'Hits@{K}'] = (valid_hits, test_hits)
    elif datasets == "collab":
        for K in [10, 50, 100]:
            evaluator.K = K

            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (valid_hits, test_hits)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)


    results['AUC'] = (roc_auc_score(valid_result.detach().cpu().numpy(), valid_pred.detach().cpu().numpy()),
                      roc_auc_score(test_result.detach().cpu().numpy(), test_pred.detach().cpu().numpy()))

    return results, h



def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--datasets', type=str, default='cora')
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner','mlp'])
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--transductive', type=str, default='transductive', choices=['transductive', 'production'])
    parser.add_argument('--minibatch', action='store_true')
    parser.add_argument('--sample_number', type=int, help='sample number', default=500)
    parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=20)
    parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=10)

    args = parser.parse_args()
    print(args)

    os.makedirs("./results", exist_ok=True)
    Logger_file = "./results/" + args.datasets + "_supervised_" + args.transductive + ".txt"
    file = open(Logger_file, "a")  # "a" 参数表示以追加模式打开文件
    file.write(str(args))
    file.write(args.encoder + " as the encoder\n")
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.transductive == "transductive":
        if args.datasets != "collab":
            dataset = get_dataset(args.dataset_dir, args.datasets)
            data = dataset[0]
            num_node = data.x.size(0)

            if exists("./data/" + args.datasets + ".pkl"):
                split_edge = torch.load("./data/" + args.datasets + ".pkl")
            else:
                split_edge = do_edge_split(dataset)
                torch.save(split_edge, "./data/" + args.datasets + ".pkl")

            edge_index = split_edge['train']['edge'].t()
            data.adj_t = edge_index
            input_size = data.x.size()[1]  #节点特征维度
            args.metric = 'Hits@20'

        elif args.datasets == "collab":
            dataset = PygLinkPropPredDataset(name=('ogbl-' + args.datasets))
            data = dataset[0]
            num_node = data.x.size(0)
            edge_index = data.edge_index
            data = T.ToSparseTensor()(data)

            split_edge = dataset.get_edge_split()
            input_size = data.num_features
            args.metric = 'Hits@50'

            data.adj_t = edge_index


        if args.use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
            if args.datasets != "collab" and args.datasets != "ppa":
                data.full_adj_t = full_edge_index
            elif args.datasets == "collab" or args.datasets == "ppa":
                data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
                data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t

        data = data.to(device)

 # Setting up the subgraph extractor
    ppr_path = './subgraph/' + args.datasets
    print("----------------------------------")
    print(args.datasets)
    subgraph = Subgraph(data.x, edge_index, ppr_path, args.subgraph_size, args.n_order)
    subgraph.build()

 # Setting up the model and optimizer,将这些参数传递给SugbCon的构造函数，创建了一个model对象。
    CLmodel = SugbCon(
        hidden_size=args.hidden_channels, gcn=GCN(data.num_features, args.hidden_channels),
        pool=Pool(args.hidden_channels),
        scorer=Scorer(args.hidden_channels)).to(device)
    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              2, args.dropout).to(device)
    evaluator = Evaluator(name='ogbl-ddi')

    if args.transductive == "transductive":
        if args.datasets != "collab" and args.datasets != "ppa":
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@20': Logger(args.runs, args),
                'Hits@30': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }
        elif args.datasets == "collab" or args.datasets == "ppa":
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'Hits@100': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }

    val_max = 0.0
    for run in range(args.runs):
        torch_geometric.seed.seed_everything(run)

        CLmodel.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(CLmodel.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0

        for epoch in range(15):
             clloss = CLtrain(CLmodel, subgraph, data, optimizer, args.sample_number, args.datasets, args.epochs)


        for epoch in range(1, 1 + args.epochs):
            if args.transductive == "transductive":
                bceloss = BCEtrain(CLmodel,predictor, data, split_edge, optimizer, args.batch_size,  args.datasets,
                                   args.transductive,args.hidden_channels,args.sample_number,subgraph)

                loss = clloss + bceloss

                results,h = test_transductive(CLmodel, predictor, data, split_edge,
                                               evaluator, args.batch_size, args.datasets,args.hidden_channels, args.sample_number,subgraph,args)


            if results[args.metric][0] > val_max:
                val_max = results[args.metric][0]

                os.makedirs("./saved-features", exist_ok=True)
                os.makedirs("./saved-models", exist_ok=True)

                torch.save({'features': h},
                           "./saved-features/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl")
                torch.save({'gnn': CLmodel.state_dict(), 'predictor': predictor.state_dict()},
                           "./saved-models/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl")

            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0  # cnt_wait 变量用于跟踪指标没有改进的次数
            else:
                cnt_wait += 1

            for key, result in results.items():  #用于将训练结果记录到日志文件
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:  #检查当前 epoch 是否是 args.log_steps 参数的倍数。如果是，则代码继续打印训练结果。
                if args.transductive == "transductive":
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

                print('---')

            if cnt_wait >= args.patience:
                break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    file = open(Logger_file, "a")
    file.write(f'All runs:\n')  # 将所有运行的训练结果写入文件
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

        if args.transductive == "transductive":
            file.write(f'{key}:\n')
            best_results = []

            for r in loggers[key].results:
                r = 100 * torch.tensor(r)
                valid = r[:, 0].max().item()
                test1 = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test1))

            best_result = torch.tensor(best_results)

            r = best_result[:, 1]
            file.write(f'Test: {r.mean():.4f} ± {r.std():.4f}\n')


    file.close()

if __name__ == "__main__":
    main()











