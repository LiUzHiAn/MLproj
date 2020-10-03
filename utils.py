"""
see https://github.com/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

LABLE2SUBMITNAME = {0: "normal", 1: "calling", 2: "smoking"}
LABLE2NAME = {0: "normal", 1: "phone", 2: "smoke"}
NAME2LABLE = {"normal": 0, "phone": 1, "smoke": 2}


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_topk_accuracy(y_pred, y, k=2):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()  # [bs,topK] -> [topK,bs] ,每一行就是第k大各类别的预测分类
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    fig.delaxes(fig.axes[1])  # delete colorbar
    plt.xticks(rotation=90)
    plt.xlabel('Predicted Label', fontsize=50)
    plt.ylabel('True Label', fontsize=50)