from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = ['Arial Unicode MS','Microsoft YaHei','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

acc__k_1000=np.load('./result/k_1_1000.npy')
acc_k_10=np.load('./result/k_1_10.npy')
acc_k_50=np.load('./result/k_1_50.npy')
acc_k_100=np.load('./result/k_1_100.npy')
acc_k_500=np.load('./result/k_1_500.npy')

acc__k_1000_cnn=np.load('./result/k_1_1000_cnn.npy')
acc_k_10_cnn=np.load('./result/k_1_10_cnn.npy')
acc_k_50_cnn=np.load('./result/k_1_50_cnn.npy')
acc_k_100_cnn=np.load('./result/k_1_100_cnn.npy')
acc_k_500_cnn=np.load('./result/k_1_500_cnn.npy')
acc_k_no=np.load('./result/k_no.npy')

epoch_list = list(range(len(acc__k_1000)))
lst1 = []
for i in epoch_list:
    lst1.append(i * 50)
epoch_list=lst1

plt.figure(1)
plt.title('fedavg_2NN_10_iid: topk -- accuracy')
plt.xlabel('Epoch')
plt.ylabel("accuracy")
plt.plot(epoch_list,acc_k_no,'black',label=u"no use top-k")
plt.plot(epoch_list,acc_k_10,'ro-',label=u"k=1/10")
plt.plot(epoch_list,acc_k_50,'y',label=u"k=1/50")
plt.plot(epoch_list,acc_k_100,'g+-',label=u"k=1/100")
plt.plot(epoch_list,acc_k_500,'mx-',label=u"k=1/500")
plt.plot(epoch_list,acc__k_1000,'bp-',label=u"k=1/1000")
plt.legend()

plt.figure(2)
plt.title('fedavg_CNN_10_iid: topk -- accuracy')
plt.xlabel('Epoch')
plt.ylabel("accuracy")
plt.plot(epoch_list,acc_k_no,'black',label=u"no use top-k")
plt.plot(epoch_list,acc_k_10_cnn,'ro-',label=u"k=1/10")
plt.plot(epoch_list,acc_k_50_cnn,'y',label=u"k=1/50")
plt.plot(epoch_list,acc_k_100_cnn,'g+-',label=u"k=1/100")
plt.plot(epoch_list,acc_k_500_cnn,'mx-',label=u"k=1/500")
plt.plot(epoch_list,acc__k_1000_cnn,'bp-',label=u"k=1/1000")
plt.legend()
plt.show()