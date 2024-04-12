from model import LogReg
import torch.nn as nn
import torch as th


def evaluation(embeddings, y, name,device,data,learning_rate2,weight_decay2):
    print("=== Evaluation ===")
    X = embeddings.detach().cpu().numpy()
    Y =  y.detach().cpu().numpy()

    graph=data
    train_mask = graph.train_mask.cpu()
    val_mask = graph.val_mask.cpu()
    test_mask = graph.test_mask.cpu()

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    train_embs = X[train_idx]
    val_embs = X[val_idx]
    test_embs = X[test_idx]

    train_labels = Y[train_idx]
    val_labels = Y[val_idx]
    test_labels = Y[test_idx]

    train_embs= th.tensor(train_embs).to(device)
    test_embs = th.tensor(test_embs).to(device)
    val_embs = th.tensor(val_embs).to(device)

    train_labels = th.tensor(train_labels).to(device)
    test_labels = th.tensor(test_labels).to(device)
    val_labels = th.tensor(val_labels).to(device)

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], 2)
    opt = th.optim.Adam(logreg.parameters(), lr=learning_rate2, weight_decay=weight_decay2)

    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        # train_labels = (3708,2)
        # train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
        train_acc = accuracy(logits,train_labels)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            # val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            val_acc = accuracy(val_logits,val_labels)
            # test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
            test_acc = accuracy(test_logits,test_labels)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

    print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    return  eval_acc

def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = th.eq(preds, labels[:, 1]).sum().item()
    accuracy = correct / len(labels)
    return accuracy
