############################
# Chance Kim, 2021/05/10
# Copy from here: https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py
# Then, modify some codes to provide multi-label classification ROC curves
# multi-label classification ROC curve reference:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
#
"""
    This file contains evaluation methods that take in a set of predicted labels 
        and a set of ground truth labels and calculate precision, recall, accuracy, f1, and metrics @k
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import time

def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix 
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    # macro
    macro = all_macro(yhat, y)

    # micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    # AUC and @k
    if yhat_raw is not None and calc_auc:
        # allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2 * (prec_at_k * rec_at_k) / (prec_at_k + rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic,
                                                                                                                ymic)


#########################################################################
# MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


###################
# INSTANCE-AVERAGED
###################

def inst_precision(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    # correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)


def inst_recall(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    # correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)


def inst_f1(yhat, y):
    prec = inst_precision(yhat, y)
    rec = inst_recall(yhat, y)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1


##############
# AT-K
##############

def recall_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i, tk].sum()
        denom = y[i, :].sum()
        if denom > 0:
            print('[ERROR] recall_at_k: divide by zero error occurred!')
            vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def precision_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i, tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


##########################################################################
# MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return

    n_classes = y.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}
    # get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(n_classes):
        # only if there are true positives for this label
        if y[:, i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:, i], yhat_raw[:, i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    ################################
    # micro-AUC: just look at each individual prediction
    ################################
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro_fpr"] = fpr["micro"]
    roc_auc["auc_micro_tpr"] = tpr["micro"]
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    ################################
    # macro-AUC
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
    ################################

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    roc_auc["auc_macro_fpr"] = all_fpr
    roc_auc["auc_macro_tpr"] = mean_tpr
    roc_auc["auc_macro"] = auc(all_fpr, mean_tpr)

    return roc_auc


########################
# METRICS BY CODE TYPE
########################

def diag_f1(diag_preds, diag_golds, ind2d, hadm_ids):
    num_labels = len(ind2d)
    yhat_diag = np.zeros((len(hadm_ids), num_labels))
    y_diag = np.zeros((len(hadm_ids), num_labels))
    for i, hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_diag_inds = [1 if ind2d[j] in diag_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_diag_inds = [1 if ind2d[j] in diag_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_diag[i] = yhat_diag_inds
        y_diag[i] = gold_diag_inds
    return micro_f1(yhat_diag.ravel(), y_diag.ravel())


def proc_f1(proc_preds, proc_golds, ind2p, hadm_ids):
    num_labels = len(ind2p)
    yhat_proc = np.zeros((len(hadm_ids), num_labels))
    y_proc = np.zeros((len(hadm_ids), num_labels))
    for i, hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_proc_inds = [1 if ind2p[j] in proc_preds[hadm_id] else 0 for j in range(num_labels)]
        gold_proc_inds = [1 if ind2p[j] in proc_golds[hadm_id] else 0 for j in range(num_labels)]
        yhat_proc[i] = yhat_proc_inds
        y_proc[i] = gold_proc_inds
    return micro_f1(yhat_proc.ravel(), y_proc.ravel())


def metrics_from_dicts(preds, golds, mdir, ind2c):
    with open('%s/pred_100_scores_test.json' % mdir, 'r') as f:
        scors = json.load(f)

    hadm_ids = sorted(set(golds.keys()).intersection(set(preds.keys())))
    num_labels = len(ind2c)
    yhat = np.zeros((len(hadm_ids), num_labels))
    yhat_raw = np.zeros((len(hadm_ids), num_labels))
    y = np.zeros((len(hadm_ids), num_labels))
    for i, hadm_id in tqdm(enumerate(hadm_ids)):
        yhat_inds = [1 if ind2c[j] in preds[hadm_id] else 0 for j in range(num_labels)]
        yhat_raw_inds = [scors[hadm_id][ind2c[j]] if ind2c[j] in scors[hadm_id] else 0 for j in range(num_labels)]
        gold_inds = [1 if ind2c[j] in golds[hadm_id] else 0 for j in range(num_labels)]
        yhat[i] = yhat_inds
        yhat_raw[i] = yhat_raw_inds
        y[i] = gold_inds
    return yhat, yhat_raw, y, all_metrics(yhat, y, yhat_raw=yhat_raw, calc_auc=False)


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def print_metrics(metrics):
    print()
    if "auc_macro" in metrics.keys():
        print("[MACRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (
        metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"]))
    else:
        print("[MACRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (
        metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"]))

    if "auc_micro" in metrics.keys():
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (
        metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"]))
    else:
        print("[MICRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (
        metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]))
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            print("%s: %.4f" % (metric, val))
    print()


def draw_multilabel_roc_curve(_metric, filename=None, line_width=2):
    if filename is None:
        date = time.strftime("%Y%m%d_%H%M%S")
        filename = 'ROC_curve_{}.png'.format(date)

    plt.figure()
    plt.plot(_metric["auc_micro_fpr"], _metric["auc_micro_tpr"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(_metric["auc_micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(_metric["auc_macro_fpr"], _metric["auc_macro_tpr"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(_metric["auc_macro"]),
             color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    ##########################################
    # sample test
    ##########################################
    y_true = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    y_pred = np.array([[0, 0],
                       [1, 1],
                       [0, 0],
                       [0, 1]])
    y_hat_raw = np.array([[0, 0.1],
                          [0.8, 0.9],
                          [0.6, 0.2],
                          [0.5, 0.7]])

    flat_y_true = y_true.ravel()
    flat_y_pred = y_pred.ravel()

    print('=======================================================')
    print('Without consideration of multi-label classification:')
    print(metrics.classification_report(flat_y_true, flat_y_pred))

    print('=======================================================')
    print('With consideration of multi-label classification:')
    metric = all_metrics(y_pred, y_true, k=5, yhat_raw=y_hat_raw)
    print('acc_macro: {}'.format(metric['acc_macro']))
    print('acc_micro: {}'.format(metric['acc_micro']))
    print()
    print('auc_macro: {}'.format(metric['auc_macro']))
    print('auc_micro: {}'.format(metric['auc_micro']))
    print()
    print('f1_at_5: {}'.format(metric['f1_at_5']))
    print('f1_macro: {}'.format(metric['f1_macro']))
    print('f1_micro: {}'.format(metric['f1_micro']))
    print()
    print('prec_at_5: {}'.format(metric['prec_at_5']))
    print('prec_macro: {}'.format(metric['prec_macro']))
    print('prec_micro: {}'.format(metric['prec_micro']))
    print()
    print('rec_at_5: {}'.format(metric['rec_at_5']))
    print('rec_macro: {}'.format(metric['rec_macro']))
    print('rec_micro: {}'.format(metric['rec_micro']))

    # Plot all ROC curves
    draw_multilabel_roc_curve(metric)


    ###########################################
    # draw MultiResCNN ROC curve from files
    ###########################################
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = 50

    y_test = np.loadtxt(open("../../Data/MultiResCNN_result/y.csv", "rb"), delimiter=",")
    y_score = np.loadtxt(open("../../Data/MultiResCNN_result/y_hat.csv", "rb"), delimiter=",")

    for i in range(n_classes):
        # only if there are true positives for this label
        if y_test[:, i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

    ################################
    # micro-AUC: just look at each individual prediction
    ################################
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["auc_micro_fpr"] = fpr["micro"]
    roc_auc["auc_micro_tpr"] = tpr["micro"]
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    ################################
    # macro-AUC
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
    ################################
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    roc_auc["auc_macro_fpr"] = all_fpr
    roc_auc["auc_macro_tpr"] = mean_tpr
    roc_auc["auc_macro"] = auc(all_fpr, mean_tpr)

    draw_multilabel_roc_curve(roc_auc)




