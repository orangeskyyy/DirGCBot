from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from layer import RGTLayer
from torch_geometric.nn import GCNConv,GATConv
import pytorch_lightning as pl
from torch import nn
import torch
import globals
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph
from augment import CustomAugmentor
from contrasts import DualBranchContrast_ex

class RGTDetector(pl.LightningModule):
    def __init__(self, args,pretrain=False):
        super(RGTDetector, self).__init__()
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.device_name = args.device_name
        self.alpha = args.alpha
        self.beta = args.beta
        self.pe = args.pe
        self.pa = args.pa
        self.pf = args.pf
        self.model_name = args.model_name
        self.pretrain = pretrain
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.linear_channels = args.linear_channels
        self.user_numeric_num = args.user_numeric_num
        self.user_cat_num = args.user_cat_num
        self.user_des_channel = args.user_des_channel
        self.user_tweet_channel = args.user_tweet_channel
        self.dataset = args.dataset
        self.user_in_linear_numeric = nn.Linear(args.user_numeric_num, int(args.linear_channels / 3), bias=True)
        self.user_in_linear_bool = nn.Linear(args.user_cat_num, int(args.linear_channels / 3), bias=True)
        # self.user_in_linear_des = nn.Linear(args.user_des_channel, int(args.linear_channels / 3), bias=True)
        self.user_in_linear_tweet = nn.Linear(args.user_tweet_channel, int(args.linear_channels / 3), bias=True)

        self.mgtab_user_in_linear_numeric = nn.Linear(args.user_numeric_num, int(args.linear_channels / 4), bias=True)
        self.mgtab_user_in_linear_bool = nn.Linear(args.user_cat_num, int(args.linear_channels / 4), bias=True)
        self.mgtab_user_in_linear_tweet = nn.Linear(args.user_tweet_channel, int(args.linear_channels / 2), bias=True)

        self.user_linear = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGT_layer1 = RGTLayer(num_edge_type=args.edge_type, in_channels=args.linear_channels,
                                   out_channels=args.out_channels,
                                   trans_heads=args.trans_head, semantic_head=args.semantic_head,
                                   dropout=args.dropout, beta=args.beta)
        # linear_channels = 128  out_channels = 128
        self.RGT_layer2 = RGTLayer(num_edge_type=args.edge_type, in_channels=args.linear_channels,
                                   out_channels=args.out_channels,
                                   trans_heads=args.trans_head, semantic_head=args.semantic_head,
                                   dropout=args.dropout, beta=args.beta)
        self.GCN1 = GCNConv(args.linear_channels, args.out_channels)
        self.GCN2 = GCNConv(args.linear_channels, args.out_channels)

        self.GAT1 = GATConv(args.linear_channels, int(args.linear_channels / 4), heads=4)
        self.GAT2 = GATConv(args.linear_channels, args.linear_channels)
        # user_channel=64
        self.out = torch.nn.Linear(args.out_channels, args.user_channel)
        self.classifier = torch.nn.Linear(args.user_channel, 2)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()
        self.contrastive = DualBranchContrast_ex()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        if self.dataset in ["Cresci-15","TwiBot-20"]:
            cat_features = train_batch.x[:, :self.user_cat_num]
            prop_features = train_batch.x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
            tweet_features = train_batch.x[:,
                             self.user_cat_num + self.user_numeric_num: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel]
            des_features = train_batch.x[:,
                           self.user_cat_num + self.user_numeric_num + self.user_tweet_channel: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel + self.user_des_channel]
            user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(tweet_features)))
            # user_features_des = self.drop(self.ReLU(self.user_in_linear_des(des_features)))
            user_features = torch.cat(
                (user_features_numeric, user_features_bool, user_features_tweet),
                dim=1)
        else:
            cat_features = train_batch.x[:, :self.user_cat_num]
            prop_features = train_batch.x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
            tweet_features = train_batch.x[:,
                             self.user_cat_num + self.user_numeric_num: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel]
            user_features_numeric = self.drop(self.ReLU(self.mgtab_user_in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.mgtab_user_in_linear_bool(cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.mgtab_user_in_linear_tweet(tweet_features)))
            user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet),
                                  dim=1)

        user_features = self.drop(self.ReLU(self.user_linear(user_features)))
        label = train_batch.y

        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_attr.view(-1)
        # user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        # user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))
        if self.pretrain:
            augmentor = CustomAugmentor(pf=self.pf, pe=self.pe, pa=self.pa, device=self.device_name)
            graph = Graph(x=user_features, edge_index=edge_index, edge_weights=edge_type)
            aug1 = augmentor.augment(graph)
            x1, edge_index1, edge_weight1 = aug1.x, aug1.edge_index, aug1.edge_weights
            graph = Graph(x=user_features, edge_index=edge_index, edge_weights=edge_type)
            aug2 = augmentor.augment(graph)
            x2, edge_index2, edge_weight2 = aug2.x, aug2.edge_index, aug2.edge_weights

            # aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.5)])
            # aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.5)])
            # x1, edge_index1, edge_weight1 = aug1(user_features, edge_index,edge_type)
            # x2, edge_index2, edge_weight2 = aug2(user_features, edge_index, edge_type)
            h1 = self.encoder(x1,edge_index1,edge_weight1)
            h2 = self.encoder(x2,edge_index2,edge_weight2)
            # user_features = self.drop(self.ReLU(self.out(user_features)))
            # pred = self.classifier(user_features)
            # ce_loss = self.CELoss(pred, label)
            # return self.alpha*ce_loss + (1-self.alpha)*self.contrastive(h1, h2)
            return self.contrastive(h1, h2)
        else:
            if self.model_name == 'DRGT':
                user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
                user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))
            elif self.model_name == 'GCN':
                user_features = self.ReLU(self.GCN1(user_features, edge_index))
                user_features = self.ReLU(self.GCN2(user_features, edge_index))
            else:
                user_features = self.ReLU(self.GAT1(user_features, edge_index))
                user_features = self.ReLU(self.GAT2(user_features, edge_index))
            user_features = self.drop(self.ReLU(self.out(user_features)))

            pred = self.classifier(user_features)
            loss = self.CELoss(pred, label)
            return loss

    def encoder(self,x,edge_index,edge_type):
        x = self.drop(self.ReLU(self.user_linear(x)))
        if self.model_name == 'DRGT':
            x = self.ReLU(self.RGT_layer1(x, edge_index, edge_type))
            x = self.ReLU(self.RGT_layer2(x, edge_index, edge_type))
        elif self.model_name == 'GCN':
            x = self.ReLU(self.GCN1(x, edge_index))
            x = self.ReLU(self.GCN2(x, edge_index))
        else:
            x = self.ReLU(self.GAT1(x, edge_index))
            x = self.ReLU(self.GAT2(x, edge_index))
        x = self.drop(self.ReLU(self.out(x)))
        return x

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            if self.dataset in ["Cresci-15", "TwiBot-20"]:
                cat_features = val_batch.x[:, :self.user_cat_num]
                prop_features = val_batch.x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
                tweet_features = val_batch.x[:,
                                 self.user_cat_num + self.user_numeric_num: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel]
                des_features = val_batch.x[:,
                               self.user_cat_num + self.user_numeric_num + self.user_tweet_channel: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel + self.user_des_channel]
                user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(prop_features)))
                user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(cat_features)))
                user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(tweet_features)))
                # user_features_des = self.drop(self.ReLU(self.user_in_linear_des(des_features)))
                user_features = torch.cat(
                    (user_features_numeric, user_features_bool, user_features_tweet),
                    dim=1)
            else:
                cat_features = val_batch.x[:, :self.user_cat_num]
                prop_features = val_batch.x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
                tweet_features = val_batch.x[:,
                                 self.user_cat_num + self.user_numeric_num: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel]
                user_features_numeric = self.drop(self.ReLU(self.mgtab_user_in_linear_numeric(prop_features)))
                user_features_bool = self.drop(self.ReLU(self.mgtab_user_in_linear_bool(cat_features)))
                user_features_tweet = self.drop(self.ReLU(self.mgtab_user_in_linear_tweet(tweet_features)))
                user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet),
                                          dim=1)
            user_features = self.drop(self.ReLU(self.user_linear(user_features)))

            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_attr.view(-1)

            if self.model_name == 'DRGT':
                user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
                user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))
            elif self.model_name == 'GCN':
                user_features = self.ReLU(self.GCN1(user_features, edge_index))
                user_features = self.ReLU(self.GCN2(user_features, edge_index))
            else:
                user_features = self.ReLU(self.GAT1(user_features, edge_index))
                user_features = self.ReLU(self.GAT2(user_features, edge_index))

            user_features = self.drop(self.ReLU(self.out(user_features)))
            pred = self.classifier(user_features)
            pred_binary = torch.argmax(pred, dim=1)

            label = val_batch.y
            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())

            self.log("val_acc", acc)
            self.log("val_f1", f1)

            print("val_acc: {} val_f1: {}".format(acc, f1))
            # mask = torch.tensor(val_batch.n_id.clone().detach(),dtype=int,device=self.device_name)
            # label = label[mask]
            # # print(pred.size())
            # # todo rename
            globals.fine_pred_val += list(pred_binary.squeeze().cpu())

            globals.fine_label_val += list(label.squeeze().cpu())

    def on_validation_epoch_end(self):
        # precision = precision_score(globals.fine_label_val, globals.fine_pred_val,average="macro",zero_division=0)
        # recall = recall_score(globals.fine_label_val, globals.fine_pred_val,average="macro",zero_division=0)
        # f1 = f1_score(globals.fine_label_val, globals.fine_pred_val, average="macro", zero_division=0)
        f1 = f1_score(globals.fine_label_val, globals.fine_pred_val)
        acc = accuracy_score(globals.fine_label_val, globals.fine_pred_val)
        globals.fine_label_val = []
        globals.fine_pred_val = []

        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        # self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_id):
        self.eval()
        with torch.no_grad():
            if self.dataset in ["Cresci-15", "TwiBot-20"]:
                cat_features = test_batch.x[:, :self.user_cat_num]
                prop_features = test_batch.x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
                tweet_features = test_batch.x[:,
                                 self.user_cat_num + self.user_numeric_num: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel]
                des_features = test_batch.x[:,
                               self.user_cat_num + self.user_numeric_num + self.user_tweet_channel: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel + self.user_des_channel]
                user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(prop_features)))
                user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(cat_features)))
                user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(tweet_features)))
                # user_features_des = self.drop(self.ReLU(self.user_in_linear_des(des_features)))
                user_features = torch.cat(
                    (user_features_numeric, user_features_bool, user_features_tweet),
                    dim=1)
            else:
                cat_features = test_batch.x[:, :self.user_cat_num]
                prop_features = test_batch.x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
                tweet_features = test_batch.x[:,
                                 self.user_cat_num + self.user_numeric_num: self.user_cat_num + self.user_numeric_num + self.user_tweet_channel]
                user_features_numeric = self.drop(self.ReLU(self.mgtab_user_in_linear_numeric(prop_features)))
                user_features_bool = self.drop(self.ReLU(self.mgtab_user_in_linear_bool(cat_features)))
                user_features_tweet = self.drop(self.ReLU(self.mgtab_user_in_linear_tweet(tweet_features)))
                user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet),
                                          dim=1)
            label = test_batch.y

            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_attr.view(-1)
            user_features = self.drop(self.ReLU(self.user_linear(user_features)))

            if self.model_name == 'DRGT':
                user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
                user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))
            elif self.model_name == 'GCN':
                user_features = self.ReLU(self.GCN1(user_features, edge_index))
                user_features = self.ReLU(self.GCN2(user_features, edge_index))
            else:
                user_features = self.ReLU(self.GAT1(user_features, edge_index))
                user_features = self.ReLU(self.GAT2(user_features, edge_index))

            user_features = self.drop(self.ReLU(self.out(user_features)))
            pred = self.classifier(user_features)
            # print(pred.size())
            pred_binary = torch.argmax(pred, dim=1)

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            precision = precision_score(label.cpu(), pred_binary.cpu())
            recall = recall_score(label.cpu(), pred_binary.cpu())
            # auc = roc_auc_score(label.cpu(), pred[:, 1].cpu())

            self.log("acc", acc)
            self.log("f1", f1)
            self.log("precision", precision)
            self.log("recall", recall)
            # self.log("auc", auc)

            print("acc: {} \t f1: {} \t precision: {} \t recall: {}".format(acc, f1, precision, recall))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }
