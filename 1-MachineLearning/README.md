# 源代码

源代码放在src目录下

- `main.py`: 主程序文件
- `model`: 定义网络模型，模型通过`ModelWrapper`类包装
- `parameters.py`: 定义配置参数和训练参数
- `data.py`: 数据加载

# 数据集

数据集下载地址：[anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

将`anime.csv`和`rating.csv`两个文件放在dataset目录下

# 指标记录

使用TensorBoard记录训练指标，日志保存在`logs`目录中，使用`tensorboard --logdir=logs`命令查看