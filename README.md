# DirGCBot
待解决的问题
1. 导入或者重新生成预处理好的数据
   - 数据集组成
     - TwiBot-20
     - Cresci-2015
     - MGTAB
   - 预处理文件组成，前两个文件的预处理好的数据最好重新生成一次
     - edge_index(edge_num,2) 边关系
     - edge_type(edge_num,) 边类型
     - features(node_num,node_feature) 前两个数据集的预处理文件中还包含了组成节点数据的向量（cat，num，tweet，des）
     - train_idx 训练集
     - valid_idx 校验集
     - test_idx 测试集
2. 对比学习和RGT融合的效果还是不好
3. SeGA的模型中的pytorch_lightning的训练写法和数据预处理的代码可以参考

![image](https://github.com/orangeskyyy/DirGCBot/assets/46984272/39b1462a-cd1a-455b-88d0-b649983beb9f)
图片来自于MGTAB论文的实验数据，可以忽略立场检测的数据
