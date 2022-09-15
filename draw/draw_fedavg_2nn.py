from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = ['Arial Unicode MS','Microsoft YaHei','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

acc_5=np.load('./result/5_clients_each_round.npy')
acc_10=np.load('./result/10_clients_each_round.npy')
acc_30=np.load('./result/30_clients_each_round.npy')
acc_100=np.load('./result/100_clients_each_round.npy')
epoch_list = list(range(len(acc_5)))
lst1 = []
for i in epoch_list:
    lst1.append(i * 50)
epoch_list=lst1

print(epoch_list)
plt.figure(1)
plt.title('fedavg_2NN_idd: clients numbers each round -- accuracy')
plt.xlabel('Epoch')
plt.ylabel("accuracy")
plt.plot(epoch_list,acc_5,'bp-',label=u"5 clients each round")
plt.plot(epoch_list,acc_10,'ro-',label=u"10 clients each round")
plt.plot(epoch_list,acc_30,'g+-',label=u"30 clients each round")
plt.plot(epoch_list,acc_100,'mx-',label=u"100 clients each round")
plt.legend()

acc_niid_1=np.load('./result/1_classes_per_client.npy')
acc_niid_2=np.load('./result/2_classes_per_client.npy')
acc_niid_3=np.load('./result/3_classes_per_client.npy')
acc_niid_4=np.load('./result/4_classes_per_client.npy')


plt.figure(2)
plt.title('fedavg_2NN_10_Not iid: classes_per_client -- accuracy')
plt.xlabel('Epoch')
plt.ylabel("accuracy")
plt.plot(epoch_list,acc_10,'black',label=u"idd")
plt.plot(epoch_list,acc_niid_1,'bp-',label=u"not idd, 1 classes_per_client")
plt.plot(epoch_list,acc_niid_2,'ro-',label=u"not idd, 2 classes_per_client")
plt.plot(epoch_list,acc_niid_3,'g+-',label=u"not idd, 3 classes_per_client")
plt.plot(epoch_list,acc_niid_4,'mx-',label=u"not idd, 4 clients each round")
plt.legend()
plt.show()