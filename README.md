# DirGCBot
待解决的问题
1. 导入或者重新生成预处理好的数据
   - 数据集组成
     - TwiBot-20
     - Cresci-2015
     - MGTAB **还未处理**
   - 预处理文件组成，前两个文件的预处理好的数据最好重新生成一次
     - edge_index(edge_num,2) 边关系
     - edge_type(edge_num,) 边类型
     - features
       - cat 分类属性embedding
       - num 数值属性embedding
       - tweet tweet文本embedding
       - des 个人描述文本embedding
     - train_idx 训练集
     - valid_idx 校验集
     - test_idx 测试集
2. SeGA的模型中的pytorch_lightning的训练写法和数据预处理的代码可以参考

![image](https://github.com/orangeskyyy/DirGCBot/assets/46984272/39b1462a-cd1a-455b-88d0-b649983beb9f)
图片来自于MGTAB论文的实验数据，可以忽略立场检测的数据

执行预训练
```python main.py --pretrain --dataset TwiBot-20```

加载预训练模型并执行微调
```python main.py --pretrain_load --dataset TwiBot-20```

不同数据集需要的注意点：
1. 设置的user_cat_num参数值不同，参考备注
2. 数据导入时，TwiBot-20数据集需要切片，详见参考load_data()方法注释

