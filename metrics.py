from sklearn.metrics import roc_curve, auc


def acc_calc(true, pred):
    fpr, tpr, _ = roc_curve(true, pred)
    acc = auc(fpr, tpr)
    return fpr, tpr, acc
