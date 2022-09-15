from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = ['Arial Unicode MS','Microsoft YaHei','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

acc_old=np.load('./result/fedavg_2_noiid.npy')
acc_new=np.load('./result/fedper_2_noiid.npy')

epoch_list_1 = list(range(len(acc_old)))
lst1 = []
for i in epoch_list_1:
    lst1.append(i * 50)
epoch_list_1=lst1

epoch_list_2 = list(range(len(acc_new)))
lst1 = []
for i in epoch_list_2:
    lst1.append(i * 10)
epoch_list_2=lst1

plt.figure(1)
plt.title('noiid: fedavg--fedper')
plt.xlabel('Epoch')
plt.ylabel("accuracy")
plt.plot(epoch_list_1,acc_old,'bp-',label=u"fedavg_noiid")
plt.plot(epoch_list_2,acc_new,'ro-',label=u"fedper_noiid")
plt.legend()
plt.show()