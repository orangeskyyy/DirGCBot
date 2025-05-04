import torch
from tqdm import  tqdm
from torch_geometric.data import Data

# features = torch.load("E:/论文/datasets/processed_data/MGTAB/features.pt")
# labels_bot = torch.load("E:/论文/datasets/processed_data/MGTAB/labels_bot.pt")
# edge_index = torch.load("E:/论文/datasets/processed_data/MGTAB/edge_index.pt")
# edge_type = torch.load("E:/论文/datasets/processed_data/MGTAB/edge_type.pt")
# data = Data(x=features,edge_index=edge_index,edge_attr=edge_type,y=labels_bot)

# train_idx = torch.arange(0,data.num_nodes*0.7)
# train_idx = torch.arange(data.num_nodes*0.7)
# train_idx = torch.arange(data.num_nodes*0.7)

def file_load():
    edge_index = torch.load("E:/论文/datasets/processed_data/MGTAB/edge_index.pt")
    edge_type = torch.load("E:/论文/datasets/processed_data/MGTAB/edge_type.pt")
    features = torch.load("E:/论文/datasets/processed_data/MGTAB/features.pt")

    cat_idx = [0,1,2,4,8,15,16,17,18,19]
    num_idx = [3,5,6,7,9,10,11,12,13,14]
    text_idx = [20]
    # 提取 cat 特征
    cat_features = features[:, cat_idx]

    # 提取 num 特征
    num_features = features[:, num_idx]

    # 提取 text 特征
    text_features = features[:, 20:]
    # 保存张量
    torch.save(cat_features, 'cat_features.pt')
    torch.save(num_features, 'num_features.pt')
    torch.save(text_features, 'text_features.pt')
    train_idx = torch.load("E:/论文/datasets/processed_data/TwiBot-20/train_idx.pt")
    val_idx = torch.load("E:/论文/datasets/processed_data/TwiBot-20/val_idx.pt")
    test_idx = torch.load("E:/论文/datasets/processed_data/TwiBot-20/test_idx.pt")
    cat_properties_tensor = torch.load("E:/论文/datasets/processed_data/TwiBot-20/cat_properties_tensor.pt")
    train_idx = torch.load("E:/论文/datasets/processed_data/Cresci-15/train_idx.pt")
    val_idx = torch.load("E:/论文/datasets/processed_data/Cresci-15/val_idx.pt")
    test_idx = torch.load("E:/论文/datasets/processed_data/Cresci-15/test_idx.pt")
    label = torch.load("E:/论文/datasets/processed_data/TwiBot-20/label.pt")
    label = torch.load("E:/论文/datasets/processed_data/Cresci-15/label.pt")
    finetune_train_idx = torch.load("E:/论文/datasets/SeGa_processed_data/finetune_train_idx.pt")
    finetune_val_idx = torch.load("E:/论文/datasets/SeGa_processed_data/finetune_val_idx.pt")
    finetune_test_idx = torch.load("E:/论文/datasets/SeGa_processed_data/finetune_test_idx.pt")

def compare():

    Cresci_edge_index = torch.load("E:/论文/datasets/processed_data/Cresci-15/edge_index.pt")
    Cresci_src = Cresci_edge_index[0]
    Cresci_dst = Cresci_edge_index[1]
    for src_id,dst_id in tqdm(zip(Cresci_src.tolist(),Cresci_dst.tolist())):
        if src_id > 5301 or dst_id > 5301:
            print('edge:{}->{}'.format(src_id,dst_id))

def get_split():
    # 假设我们有 10199 个样本
    num_samples = 10199

    # 设置随机种子,确保每次运行结果一致
    torch.manual_seed(42)

    # 按照 7:2:1 的比例随机分割样本
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # 计算各个子集的大小
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size

    # 使用 random_split() 函数进行随机分割
    train_idx, val_idx, test_idx = torch.utils.data.random_split(
        range(num_samples), [train_size, val_size, test_size]
    )
    # 保存到本地文件
    torch.save(train_idx, 'train_idx.pt')
    torch.save(val_idx, 'val_idx.pt')
    torch.save(test_idx, 'test_idx.pt')

if __name__ == '__main__':
    # file_load()
    # get_split()
    train_ratio,val_ratio=0.8,0.1
    num_samples = 5301
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size
    train_idx, val_idx, test_idx = torch.utils.data.random_split(
        range(num_samples), [train_size, val_size, test_size]
    )
    torch.save(train_idx,'E:\论文\datasets\processed_data\Cresci-15\\train_idx.pt')
    torch.save(val_idx,'E:\论文\datasets\processed_data\Cresci-15\\val_idx.pt')
    torch.save(test_idx,'E:\论文\datasets\processed_data\Cresci-15\\train_idx.pt')