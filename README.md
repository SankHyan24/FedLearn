# Federal Learning Lab
## 文件结构说明：

文件结构以及内容更改如下：

```shell
.
├── Data				
│   └── Datasets.py
├── Examples
│   └── MNIST
│       ├── GenClients.py		# 增加了产生noniid测试集的方法
│       └── Model.py			# 添加了一个更复杂的网络
├── Experiment.py				# 程序的入口，加入了较多修改
├── FedLearn.py					# 联邦学习模型框架，加入了较多修改
├── fedper.md					# 联邦学习FedPer算法学习笔记
├── README.md
├── Utils.py					# 相关辅助函数，加入了新的函数
├── draw					## 新增：绘图相关程序
│   ├── draw_fedavg_2nn.py
│   ...
│   └── draw_topk.py
├── requirements.txt			# 新增：依赖列表，多用了个tqdm，问题不大
└── result 						# 新增：放置每次实验输出的结果，用于作图。实际内容我删了，因为没必要上传
    ├── 100_clients_each_round.npy
   	...
    └── old.npy
```

我们代码上的工作（包括但不限于）：

1. 稀疏更新相关：
   1. 在`FederatedScheme`类中添加了对稀疏更新的支持，增加了`train_client`的稀疏更新版本`train_client_sparse`，向类中新增了许多参数和结构。

2. `FedPer`相关：
   1. 在`FederatedScheme`类中添加了对`FedPer`算法的支持，新增了许多参数和结构
   2. 设计了服务器训练框架`fed_per_one_step`和客户端训练框架`train_client_fedper`，以及相应的辅助函数。
   3. 增加了产生noniid测试集的函数（在Genclient中）
   3. 在整个数据集上验证了FedPer算法的客户端在整个数据集上趋向于收敛（代码就是原有的训练框架）
   3. 重新设计了训练与测试框架，使用noniid的本地对应的测试集验证程序的正确性
3. 聚合方式相关：
   1. 在`FederatedScheme`类中添加了一种基于损失大小的聚合方式`fed_avg_one_step_loss`，改进了原有的FedAVG



## 使用方法：
直接执行`Experiment.py`脚本

"draw"一系列文件是独自单独运行，运来绘制每个测试的精确图，我们的测试结果也上传在内，所以可以直接运行各个draw文件。

如果想要测试不同的模型，需要在`Experiment.py`脚本中更改想要调用的函数。

如：测试fedper，调用fed_per_one_step函数；测试fedavg，调用fed_avg_one_step函数，同时还可以在`FedLearn.py`文件里更改fed_avg_one_step函数中train_client为train_clinet_spare来测试稀疏更新。

## 依赖：
- python版本：`3.8.13`

- 参考`requirements.txt`文件


## 算法说明(我的笔记):
### 联邦学习算法
1. FedAvg算法: 每轮：服务器把模型参数发送给客户端，客户端更新参数，然后把更新后的参数发送给服务器，服务器把客户端的参数进行平均，然后更新模型参数。
2. FedSGD算法：每轮：服务器把模型参数发送给客户端，客户端得到模型参数后，计算梯度，然后把梯度发送给服务器，服务器把客户端的梯度进行平均，然后更新模型参数。
3. FedProx算法：每个客户端由于系统异构性设置不同超参数，引入不精确解，改变了本地的目标函数
4. FedPer算法：网络分基础层和个性化层。所有client公用基础层，保留自己的个性化层。FedAvg法更新基础层。

### 稀疏更新算法：
在分布式机器学习中会存在分布式随机梯度下降(Distributed SGD)，client从server处获取到parameters之后算出梯度，再将梯度数据传递给server。而稀疏更新就是说，client计算出权重的梯度矩阵，只将其绝对值大于一个设定值的一部分梯度数据传递给server，也就是传一个稀疏矩阵。但是把那些小的数据直接当成0会影响收敛，所以需要以下算法：
```python
GradDrop.init=0
def GradDrop(Delta:Gradient matrix,R:Dropping rate):
    # 初始化残差矩阵
    if GradDrop.init==0:
        GradDrop.init=1
        GradDrop.residual=np.zeros(Delta.shape)
    # 补偿上一步的残差
    Delta+=GradDrop.residual
    # 绝对值前R%的梯度保留，其余置为0
    threshold=np.percentile(np.abs(Delta),R*100)
    dropped=np.where(np.abs(Delta)>threshold,Delta,0)
    # 将剩余的梯度保存起来，作为补偿下次再加上
    GradDrop.residual=Delta-dropped
    return dropped
```

