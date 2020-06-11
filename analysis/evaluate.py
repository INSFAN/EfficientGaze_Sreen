import numpy as np 
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    data1 = np.load('../output/gaze_data1.npy')
    data2 = np.load('../output/gaze_data2.npy')
    data3 = np.load('../output/gaze_data3.npy')
    data4 = np.load('../output/gaze_data4.npy')
    data5 = np.load('../output/gaze_data5.npy')

    data = np.vstack((data1,data2))
    data = np.vstack((data,data3))
    data = np.vstack((data,data4))
    data = np.vstack((data,data5))

    print(np.shape(data))

    cm = confusion_matrix(data[:,0], data[:,1])
    print(cm)

