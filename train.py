import argparse
import os.path as osp
import sys
import time
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv

from layer import RGTLayer
from model import Encoder, Model, drop_feature
from eval import evaluation
import numpy as np


def train(pre_train:RGTLayer, model: Model, x, edge_index, edge_type):
    x = pre_train(x, edge_index, edge_type)
    # add high_lv neighbors
    coo1 = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.x.shape[0])
    coo1 = coo1.toarray()
    tmp = coo1.copy()
    for i in range(2, nei_lv + 1):
        coo1 += tmp ** i
    coo1 = sp.coo_matrix(coo1)
    indices = np.vstack((coo1.row, coo1.col))
    edge_index = torch.LongTensor(indices).to(x.device)

    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]

    device = edge_index_1.device

    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    pre_z1 = model(x_1, edge_index_1, 1, None, None)
    pre_z2 = model(x_2, edge_index_2, 2, None, None)

    z1 = model(x_1, edge_index_1, 1, pre_z1, pre_z2)
    z2 = model(x_2, edge_index_2, 2, pre_z1, pre_z2)

    loss = model.loss(z1, z2, edge_index_1, edge_index_2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, name, device, data, learning_rate2, weight_decay2, final=False):
    model.eval()
    z = model(x, edge_index, 1, None, None)
    return evaluation(z, y, name, device, data, learning_rate2, weight_decay2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-2)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--dfr1', type=float, default=0.1)
    parser.add_argument('--dfr2', type=float, default=0.1)
    parser.add_argument('--der1', type=float, default=0.1)
    parser.add_argument('--der2', type=float, default=0.1)
    parser.add_argument('--lv', type=int, default=1)
    parser.add_argument('--cutway', type=int, default=2)
    parser.add_argument('--cutrate', type=float, default=1.0)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--wd2', type=float, default=1e-4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=768)
    parser.add_argument('--num_proj_hidden', type=int, default=512)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='Cresci-2015')
    parser.add_argument("--linear_channels", type=int, default=128)
    parser.add_argument("--num_edge_type", type=int, default=2)
    parser.add_argument("--in_channel", type=int, default=128)
    parser.add_argument("--out_channel", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--trans_head", type=int, default=2)
    parser.add_argument("--semantic_head", type=int, default=5)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    learning_rate = args.lr
    learning_rate2 = args.lr
    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2
    tau = args.tau
    mode = args.mode
    nei_lv = args.lv
    cutway = args.cutway
    cutrate = args.cutrate
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    activation = F.relu
    base_model = GCNConv
    num_layers = args.num_layers
    num_epochs = args.num_epochs
    weight_decay = args.wd
    weight_decay2 = args.wd2
    path = args.path
    dataset = args.dataset

    num_edge_type = args.num_edge_type
    in_channel = args.in_channel
    out_channel = args.out_channel
    trans_head = args.trans_head
    semantic_head = args.semantic_head
    dropout = args.dropout


    def get_data():
        user_x = torch.load(path + "/features.pt", map_location='cpu')
        edge_index = torch.load(path + "/edge_index.pt", map_location='cpu')
        edge_type = torch.load(path + "/edge_type.pt", map_location='cpu')
        labels = torch.load(path + "/labels.pt", map_location='cpu')

        data = Data(x=user_x, edge_index=edge_index, edge_attr=edge_type, y=labels)

        train_mask_idx = torch.load(path + "/train_idx.pt", map_location='cpu')
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_mask_idx] = True
        data.train_mask = train_mask
        valid_mask_idx = torch.load(path + "/valid_idx.pt", map_location='cpu')
        valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        valid_mask[valid_mask_idx] = True
        data.val_mask = valid_mask
        test_mask_idx = torch.load(path + "/test_idx.pt", map_location='cpu')
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[test_mask_idx] = True
        data.test_mask = test_mask
        return data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_data().to(device)

    # pretrain
    pretrain = RGTLayer(num_edge_type=num_edge_type,in_channels=768, out_channels=num_hidden,
                        trans_heads=trans_head,semantic_head=semantic_head,dropout=dropout).to(device)

    # encoder
    encoder = Encoder(num_hidden, num_hidden, activation, mode,
                      base_model=base_model, k=num_layers, cutway=cutway, cutrate=cutrate, tau=tau).to(device)
    # model
    model = Model(encoder, num_hidden, num_proj_hidden, mode, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # add high_lv neighbors
    # coo1 = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.x.shape[0])
    # coo1 = coo1.toarray()
    # tmp = coo1.copy()
    # for i in range(2, nei_lv + 1):
    #     coo1 += tmp ** i
    # coo1 = sp.coo_matrix(coo1)
    # indices = np.vstack((coo1.row, coo1.col))
    # edge_index = torch.LongTensor(indices).to(device)

    # save_path
    cur_path = osp.abspath(__file__)
    cur_dir = osp.dirname(cur_path)
    model_save_path = osp.join(cur_dir, 'pkl', args.dataset + '.pkl')

    start = t()
    prev = start
    if not args.test:
        for epoch in range(1, num_epochs + 1):
            loss = train(pretrain, model, data.x, data.edge_index, data.edge_attr)
            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
            if epoch % 10 == 0:
                print("=== Test ===")
                eval_acc = test(model, data.x, data.edge_index, data.y, args.dataset, device, data, learning_rate2,
                                weight_decay2, final=True)


    if not args.test:
        print('save_path:',model_save_path)
        torch.save(model.state_dict(), model_save_path)
        print('save_success')

    else:
        if osp.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
        else:
            print('model not exit')
            sys.exit(0)

    print("=== Eval ===")
    accs = []
    for i in range(10):
        acc = test(model, data.x, data.edge_index, data.y, args.dataset, device, data, learning_rate2, weight_decay2,
                   final=True)
        accs.append(acc)

    accs = torch.tensor(accs)
    fin_acc=torch.mean(accs)
    fin_std=torch.std(accs)
    print('fin_accuracy',fin_acc,'fin_std',fin_std)
