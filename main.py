import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader,DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, \
    roc_auc_score, mean_squared_error
from os import listdir
import globals
from RGT import RGTDetector
import numpy as np
import random
import os
import json


def build_args():
    parser = argparse.ArgumentParser(
        description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
    parser.add_argument("--path", type=str, default="E:/论文/datasets/processed_data/Cresci-15/", help="dataset path Cresci-15 TwiBot-20")
    parser.add_argument("--dataset", type=str,default='Cresci-15')
    parser.add_argument("--linear_channels", type=int, default=128, help="linear channel")
    parser.add_argument("--user_channel", type=int, default=64, help="user channel")
    parser.add_argument("--user_numeric_num", type=int, default=5, help="numerical feature channel")
    parser.add_argument("--user_cat_num", type=int, default=1, help="TwiBot=3 ,Cresci-15=1 categorical features channel")
    parser.add_argument("--user_des_channel", type=int, default=768, help="description channel")
    parser.add_argument("--user_tweet_channel", type=int, default=768, help="tweet channel")
    parser.add_argument("--out_channels", type=int, default=128, help="output channel")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--edge_type", type=int, default=2, help="number of edge type")
    parser.add_argument("--trans_head", type=int, default=2, help="transformer head channel")
    parser.add_argument("--semantic_head", type=int, default=2, help="description channel")
    parser.add_argument("--batch_size", type=int, default=128, help="description channel")
    parser.add_argument("--test_batch_size", type=int, default=128, help="description channel")
    parser.add_argument("--epochs", type=int, default=200, help="number of epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
    parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
    parser.add_argument("--random_seed", type=int, default=None, help="random")
    parser.add_argument("--accelerator", type=str, default='cuda')
    parser.add_argument("--device_name", type=str, default='cuda:0')
    pretrain = parser.add_mutually_exclusive_group()
    pretrain.add_argument("--pretrain", action="store_true")
    pretrain.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--pretrain_load", action="store_true")
    return parser.parse_args()


def load_data(path):
    print("loading user features...")
    user_cat_features = torch.load(path + "cat_properties_tensor.pt", map_location='cpu')
    user_prop_features = torch.load(path + "num_properties_tensor.pt", map_location='cpu')
    user_tweet_features = torch.load(path + "tweets_tensor.pt", map_location='cpu')
    user_des_features = torch.load(path + "des_tensor.pt", map_location='cpu')
    print("user_des_features",user_des_features)
    # user_x = torch.cat((user_cat_features, user_prop_features, user_tweet_features, user_des_features), dim=1)
    # TwiBot20输入数据需要切片处理
    user_x = torch.cat((user_cat_features, user_prop_features, user_tweet_features, user_des_features), dim=1)[:11826]
    label = torch.load(path + "label.pt", map_location='cpu')
    edge_index = torch.load(path + "edge_index.pt", map_location='cpu')
    edge_type = torch.load(path + "edge_type.pt", map_location='cpu').unsqueeze(-1)
    data = Data(x=user_x, edge_index=edge_index, edge_attr=edge_type, y=label)
    data.train_idx = torch.load(path + "train_idx.pt", map_location='cpu')
    data.valid_idx = torch.load(path + "val_idx.pt", map_location='cpu')
    data.test_idx = torch.load(path + "test_idx.pt", map_location='cpu')
    data.n_id = torch.arange(data.num_nodes)
    return data


if __name__ == "__main__":
    args = build_args()

    acc_list = [];
    precision_list = [];
    recall_list = [];
    f1_list = []

    if args.random_seed != None:
        pl.seed_everything(args.random_seed)

    # load data
    dataset = load_data(args.path)
    print(dataset)

    # set callbacks
    fine_checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{epoch:02d}-{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    # train_loader = NeighborLoader(dataset, num_neighbors=[32],input_nodes=dataset.train_idx, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # train_loader = DataLoader(dataset,  batch_size=args.batch_size,shuffle=True, num_workers=2)
    # valid_loader = NeighborLoader(dataset, num_neighbors=[32], input_nodes=dataset.valid_idx, batch_size=args.batch_size, persistent_workers=True,shuffle=True, num_workers=2)
    #
    # test_loader = NeighborLoader(dataset, num_neighbors=[32], input_nodes=dataset.test_idx, batch_size=args.test_batch_size, persistent_workers=True, shuffle=True, num_workers=2)

    if args.pretrain:
        print("Pretraining...")
        train_loader = NeighborLoader(dataset, num_neighbors=[32], input_nodes=dataset.train_idx,
                                      batch_size=args.batch_size, shuffle=True, num_workers=2)
        # train_loader = DataLoader(dataset,  batch_size=args.batch_size,shuffle=True, num_workers=2)
        valid_loader = NeighborLoader(dataset, num_neighbors=[32], input_nodes=dataset.valid_idx,
                                      batch_size=args.batch_size, persistent_workers=True,  num_workers=2)

        test_loader = NeighborLoader(dataset, num_neighbors=[32], input_nodes=dataset.test_idx,
                                     batch_size=args.test_batch_size, persistent_workers=True,
                                     num_workers=2)
        trainer = pl.Trainer(accelerator=args.accelerator, max_epochs=args.epochs, precision='16-mixed',
                             log_every_n_steps=1, num_nodes=1)

        model = RGTDetector(args, pretrain=True)
        trainer.fit(model, train_loader)
        torch.save(model.state_dict(),"{}_pretrain_model".format(args.dataset))
    # valid_dataset = BotDataset(name="valid",path=args.path)
    # test_dataset = BotDataset(name="test",path=args.path)
    # valid_loader = DataLoader(valid_dataset, batch_size=1)
    # test_loader = DataLoader(test_dataset, batch_size=1)
    print("Finetuning...")
    print("loading data...")
    train_loader = NeighborLoader(dataset, num_neighbors=[32],input_nodes=dataset.train_idx,
                                  batch_size=args.batch_size, shuffle=True, num_workers=2)

    valid_loader = NeighborLoader(dataset, num_neighbors=[32], input_nodes=dataset.valid_idx,
                                  batch_size=args.batch_size, persistent_workers=True,shuffle=True, num_workers=2)

    test_loader = NeighborLoader(dataset, num_neighbors=[32], input_nodes=dataset.test_idx,
                                 batch_size=args.test_batch_size, persistent_workers=True, shuffle=True, num_workers=2)

    model = RGTDetector(args)
    if args.pretrain or args.pretrain_load:
        model.load_state_dict(torch.load("{}_pretrain_model".format(args.dataset)))
    trainer = pl.Trainer(accelerator=args.accelerator, max_epochs=args.epochs, precision='16-mixed', log_every_n_steps=1, num_nodes=1, callbacks=[fine_checkpoint_callback])
    trainer.fit(model, train_loader, valid_loader)
    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
    best_model = RGTDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)
    # json.dump(vars(args), open('lightning_logs/version_{}/arguments.json'.format(trainer.logger.version), 'w'))