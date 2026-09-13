import numpy as np
import os
import matplotlib.pyplot as plt
import datetime



def plot_confusion_matrix(cm, dirname, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    """[给定混淆矩阵，绘制并保存]

    Args:
        cm ([二维array]): [混淆矩阵]
        dirname ([str]): [混淆矩阵图要存储的位置]
        labels (list, optional): [混淆矩阵的标签]
        title (str, optional): [description]. Defaults to 'Confusion matrix'.
        cmap ([type], optional): [混淆矩阵图中的颜色]. Defaults to plt.cm.Blues.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    now_time = datetime.datetime.now()
    now_time = now_time.strftime("%m-%d-%H-%M-%S")
    safe_title = title.replace(':', '-')
    plt.savefig(os.path.join(dirname, safe_title + now_time + '.jpg'))


def generate_confusion_matrix(predict, targets, classes):
    """[生成混淆矩阵]

    Args:
        predict ([一维array]]): [网络的得到预测值]]
        targets ([一维array ]): [对应的真实标签]
        classes ([一维array]): 真实类别，str
    """
    predict = np.asarray(predict)
    targets = np.asarray(targets)
    if predict.ndim > 1:
        predict = predict.argmax(axis=1)
    predict = predict.astype(np.int64, copy=False)
    targets = targets.astype(np.int64, copy=False)
    conf = np.zeros([len(classes), len(classes)])
    for i in range(predict.shape[0]):
        j = int(targets[i])
        k = int(predict[i])
        if 0 <= j < len(classes) and 0 <= k < len(classes):
            conf[j][k] += 1
    row_sum = conf.sum(axis=1, keepdims=True)
    confnorm = np.divide(conf, row_sum, out=np.zeros_like(conf), where=row_sum != 0)
    return confnorm