from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = ['Arial Unicode MS','Microsoft YaHei','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

acc_5=np.load('./result/5_clients_each_round_CNN.npy')
acc_10=np.load('./result/10_clients_each_round_CNN.npy')
acc_30=np.load('./result/30_clients_each_round_CNN.npy')
acc_100=np.load('./result/100_clients_each_round_CNN.npy')
epoch_list = list(range(len(acc_5)))
lst1 = []
for i in epoch_list:
    lst1.append(i * 50)
epoch_list=lst1

print(epoch_list)
plt.figure(1)
plt.title('fedavg_CNN_idd: clients numbers each round -- accuracy')
plt.xlabel('Epoch')
plt.ylabel("accuracy")
plt.plot(epoch_list,acc_5,'bp-',label=u"5 clients each round")
plt.plot(epoch_list,acc_10,'ro-',label=u"10 clients each round")
plt.plot(epoch_list,acc_30,'g+-',label=u"30 clients each round")
plt.plot(epoch_list,acc_100,'mx-',label=u"100 clients each round")
plt.legend()
plt.show()