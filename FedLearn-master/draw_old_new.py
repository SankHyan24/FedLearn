from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = ['Arial Unicode MS','Microsoft YaHei','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False


#绘制 普通的汇总模式 与 损失占比的汇总模式 的比较图
acc_old=np.load('./result/old.npy')
acc_new=np.load('./result/new.npy')

epoch_list = list(range(len(acc_old)))
lst1 = []
for i in epoch_list:
    lst1.append(i * 50)
epoch_list=lst1

plt.figure(1)
plt.title('old-new')
plt.xlabel('Epoch')
plt.ylabel("accuracy")
plt.plot(epoch_list,acc_old,'bp-',label=u"old")
plt.plot(epoch_list,acc_new,'ro-',label=u"new")
plt.legend()
plt.show()