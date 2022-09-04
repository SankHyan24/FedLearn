# Federal Learning Lab
## 使用：
直接执行`Experiment.py`脚本

## 依赖：
- python版本：`3.8.13`

- 参考`requirements.txt`文件

## 算法说明；
### 联邦学习算法
1. FedAvg算法: 每轮：服务器把模型参数发送给客户端，客户端更新参数，然后把更新后的参数发送给服务器，服务器把客户端的参数进行平均，然后更新模型参数。
2. FedSGD算法：每轮：服务器把模型参数发送给客户端，客户端得到模型参数后，计算梯度，然后把梯度发送给服务器，服务器把客户端的梯度进行平均，然后更新模型参数。
3. FedProx算法：每个客户端由于系统异构性设置不同超参数，引入不精确解，改变了本地的目标函数
4. FedPer算法：网络分基础层和个性化层。所有client公用基础层，保留自己的个性化层。FedAvg法更新基础层。_有个问题：我怎么测试我的模型准确度？随便找一个客户端测试么？_

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

