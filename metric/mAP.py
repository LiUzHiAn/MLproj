import json
import os
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import pandas as pd
import pickle

SUBMITNAME2LABLE = {"normal": 0, "calling": 1, "smoking": 2}
LABLE2SUBMITNAME = {0: "normal", 1: "calling", 2: "smoking"}


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EveryPointInterpolation = 1
    ElevenPointInterpolation = 2


def calc_mAP(predictions, gts, method=MethodAveragePrecision.EveryPointInterpolation):
    """
    refer to https://github.com/rafaelpadilla/Object-Detection-Metrics#different-competitions-different-metrics

    :param predictions: model output classification score
    :param gts: ground truth labels for all images,a mapping from img name to label
    :param method: (default = EveryPointInterpolation): It can be calculated as the implementation
                    in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
                    interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
                    or EveryPointInterpolation"  (ElevenPointInterpolation);
    :return:
    """
    # list containing metrics (precision, recall, average precision) of each class
    ret = []

    # get all classes
    classes = ["normal", "calling", "smoking"]
    for cls in classes:
        # all the predictions for current class, list of dict
        cls_preds = [item for item in predictions if item["category"] == cls]
        # all the gt images for current class, list of str(image path)
        cls_gt_images = [k for k, v in gts.items() if LABLE2SUBMITNAME[v] == cls]
        cur_cls_num_pos = len(cls_gt_images)
        # sort predictions as per confidence decreasing order
        cls_preds = sorted(cls_preds, key=lambda pred: pred["score"], reverse=True)
        TP = np.zeros(len(cls_preds))
        FP = np.zeros(len(cls_preds))

        for i in range(len(cls_preds)):
            if LABLE2SUBMITNAME[gts[cls_preds[i]["image_name"]]] == cls:
                TP[i] = 1
            else:
                FP[i] = 1

        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        recall = acc_TP / cur_cls_num_pos
        precision = np.divide(acc_TP, (acc_FP + acc_TP))

        # Depending on the method, call the right implementation
        if method == MethodAveragePrecision.EveryPointInterpolation:
            # for pascal VOC 2010 and later
            [ap, mpre, mrec, ii] = CalculateAveragePrecision(recall, precision)
        else:
            # for pascal VOC 2007
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(recall, precision)
        # add class result in the dictionary to be returned
        r = {
            'class': cls,
            'precision': precision,
            'recall': recall,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': cur_cls_num_pos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }
        ret.append(r)
    return ret


def plot_precision_reall_curve(predictions, gts,
                               method=MethodAveragePrecision.EveryPointInterpolation,
                               show_interpolated_precision=False,
                               save_path=None, show_graph=True):
    results = calc_mAP(predictions, gts, method)
    # 保存数据到文件
    pickle.dump(results, open("./pretrained_resnet50_noDataAug_map_stats.json", "wb"))

    colors = ["red", "blue", "green"]
    for idx, cls_result in enumerate(results):
        class_name = cls_result['class']
        precision = cls_result['precision']
        recall = cls_result['recall']
        average_precision = cls_result['AP']
        ap_str = "{0:.4f}%".format(average_precision * 100)
        mpre = cls_result['interpolated precision']
        mrec = cls_result['interpolated recall']
        npos = cls_result['total positives']
        total_tp = cls_result['total TP']
        total_fp = cls_result['total FP']

        if show_interpolated_precision:
            if method == MethodAveragePrecision.EveryPointInterpolation:
                plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
            elif method == MethodAveragePrecision.ElevenPointInterpolation:
                # Uncomment the line below if you want to plot the area
                # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
                # Remove duplicates, getting only the highest precision of each recall value
                nrec = []
                nprec = []
                for idx in range(len(mrec)):
                    r = mrec[idx]
                    if r not in nrec:
                        idxEq = np.argwhere(mrec == r)
                        nrec.append(r)
                        nprec.append(max([mpre[int(id)] for id in idxEq]))
                plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
        plt.plot(recall, precision, color=colors[idx], label='Class: %s, AP:%s' % (class_name, ap_str))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision x Recall curve')
        plt.legend(shadow=True)
        plt.grid()
        ############################################################
        # Uncomment the following block to create plot with points #
        ############################################################
        # plt.plot(recall, precision, 'bo')
        # labels = ['R', 'Y', 'J', 'A', 'U', 'C', 'M', 'F', 'D', 'B', 'H', 'P', 'E', 'X', 'N', 'T',
        # 'K', 'Q', 'V', 'I', 'L', 'S', 'G', 'O']
        # dicPosition = {}
        # dicPosition['left_zero'] = (-30,0)
        # dicPosition['left_zero_slight'] = (-30,-10)
        # dicPosition['right_zero'] = (30,0)
        # dicPosition['left_up'] = (-30,20)
        # dicPosition['left_down'] = (-30,-25)
        # dicPosition['right_up'] = (20,20)
        # dicPosition['right_down'] = (20,-20)
        # dicPosition['up_zero'] = (0,30)
        # dicPosition['up_right'] = (0,30)
        # dicPosition['left_zero_long'] = (-60,-2)
        # dicPosition['down_zero'] = (-2,-30)
        # vecPositions = [
        #     dicPosition['left_down'],
        #     dicPosition['left_zero'],
        #     dicPosition['right_zero'],
        #     dicPosition['right_zero'],  #'R', 'Y', 'J', 'A',
        #     dicPosition['left_up'],
        #     dicPosition['left_up'],
        #     dicPosition['right_up'],
        #     dicPosition['left_up'],  # 'U', 'C', 'M', 'F',
        #     dicPosition['left_zero'],
        #     dicPosition['right_up'],
        #     dicPosition['right_down'],
        #     dicPosition['down_zero'],  #'D', 'B', 'H', 'P'
        #     dicPosition['left_up'],
        #     dicPosition['up_zero'],
        #     dicPosition['right_up'],
        #     dicPosition['left_up'],  # 'E', 'X', 'N', 'T',
        #     dicPosition['left_zero'],
        #     dicPosition['right_zero'],
        #     dicPosition['left_zero_long'],
        #     dicPosition['left_zero_slight'],  # 'K', 'Q', 'V', 'I',
        #     dicPosition['right_down'],
        #     dicPosition['left_down'],
        #     dicPosition['right_up'],
        #     dicPosition['down_zero']
        # ]  # 'L', 'S', 'G', 'O'
        # for idx in range(len(labels)):
        #     box = dict(boxstyle='round,pad=.5',facecolor='yellow',alpha=0.5)
        #     plt.annotate(labels[idx],
        #                 xy=(recall[idx],precision[idx]), xycoords='data',
        #                 xytext=vecPositions[idx], textcoords='offset points',
        #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        #                 bbox=box)
    if save_path is not None:
        plt.savefig(save_path)
    if show_graph is True:
        plt.show()
    plt.close()
    return results


def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


# 11-point interpolated average precision
def ElevenPointInterpolatedAP(rec, prec):
    # def CalculateAveragePrecision2(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]


def plot_curves_of_different_methods_in_single_figure():
    """
    把不同方法的3个类别的Precision-Recall 曲线画在一张图上，便于对比
    :return:
    """

    # 加载数据文件
    vgg19_res = pickle.load(open("./pretrained_vgg19_map_stats.json", "rb"))
    resnet50_res = pickle.load(open("./pretrained_resnet50_map_stats.json", "rb"))
    resnet101_res = pickle.load(open("./pretrained_resnet101_map_stats.json", "rb"))
    methods = [vgg19_res, resnet50_res, resnet101_res]
    # vgg19-red, resnet50-blue , resnet101-green
    colors = [("red", "SCVGG-17"), ("blue", "SCResNet-50"), ("green", "SCResNet-101")]
    class_line_style = {"normal": "solid", "smoking": "dotted", "calling": "dashdot"}

    plt.figure(figsize=(10, 6))
    for i, method_res in enumerate(methods):
        curve_color = colors[i][0]
        for idx, cls_result in enumerate(method_res):
            class_name = cls_result['class']
            precision = cls_result['precision']
            recall = cls_result['recall']
            average_precision = cls_result['AP']
            ap_str = "{0:.2f}%".format(average_precision * 100)
            mpre = cls_result['interpolated precision']
            mrec = cls_result['interpolated recall']
            npos = cls_result['total positives']
            total_tp = cls_result['total TP']
            total_fp = cls_result['total FP']

            plt.plot(recall, precision, color=curve_color, linestyle=class_line_style[class_name],
                     label='%s, Class: %s, AP:%s' % (colors[i][1], class_name, ap_str))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(shadow=True)
            plt.grid()

    plt.savefig("mAP_compare.pdf")


def main():
    df_val = pd.read_csv('../resources/val.csv')
    val_samples = [tuple(x) for x in df_val.to_numpy()]
    # tuple list to dict
    gts = {}
    for (img_path, label) in val_samples:
        gts[img_path] = label

    with open("pretrained_resnet50_noDataAug_validation_result.json", "r") as f:
        predictions = json.load(f)

    # res = calc_mAP(predictions, gts)
    # for cls_res in res:
    #     print("class: %s, AP: %.6f" % (cls_res["class"], cls_res["AP"]))

    plot_precision_reall_curve(predictions, gts, save_path="./pretrained_resnet50_noDataAug.png")


if __name__ == '__main__':
    main()

    # # test case
    # recall = [0.066] * 2 + [0.133] * 7 + [0.2] * 2 + [0.266] + [0.333] + [0.4] * 9 + [0.466] * 2
    # precision = [1, 0.5, 0.666, 0.5, 0.4, 0.333, 0.286, 0.25, 0.222, 0.3, 0.273, 0.333, 0.385,
    #              0.429, 0.4, 0.375, 0.353, 0.333, 0.316, 0.3, 0.286, 0.273, 0.304, 0.292]
    # res = CalculateAveragePrecision(recall, precision)
    # print(res)

    # plot_curves_of_different_methods_in_single_figure()
