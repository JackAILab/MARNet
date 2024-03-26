import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score


def positive(y_true):
    return np.sum((y_true == 1))


def negative(y_true):
    return np.sum((y_true == 0))


def true_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 1))


def false_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 1))


def true_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 0))


def false_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 0))


def Multi_accuracy(y_true, y_pred):
    # y_true [1,2,3,1,2,1]
    # y_pred [0,2,2,1,4,3]
    over_all_bingo_sample = 0
    sample_list = [0, 0, 0, 0, 0]
    # 计算总体的acc
    for i in range(0, len(y_true)):
        if y_true[i] == y_pred[i]:
            over_all_bingo_sample += 1

    for i in range(0, len(y_true)):
        sample_list[y_true[i]] += 1
    sample_true_e_pred_list = [0, 0, 0, 0, 0]
    for i in range(0, len(y_true)):
        if y_pred[i] == y_true[i]:
            sample_true_e_pred_list[y_true[i]] += 1
    acc_list = []
    for i in range(0, 5):
        if sample_list[i] == 0:
            acc_list.append(-1)
        else:
            acc_list.append(sample_true_e_pred_list[i]/sample_list[i])
    over_all_acc = over_all_bingo_sample/len(y_pred)
    return acc_list, over_all_acc


def accuracy1(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count


def accuracy(y_true, y_pred):
    cnt = 0
    for i in range(0, y_true.shape[0]):
        if y_true[i] == y_pred[i]:
            cnt += 1
    return cnt/y_true.shape[0]


def sensitive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    p = positive(y_true)
    return tp / p


def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    n = negative(y_true)
    return tn / n


def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = (2 * prec * reca) / (prec + reca)
    return fs


def dice_image(pred, target):
    n = target.shape[0]
    smooth = 0.0001

    target = np.reshape(target, (n, -1))
    pred = np.reshape(pred, (n, -1))
    intersect = np.sum(target * pred, axis=-1)
    dice = (2 * intersect + smooth) / \
        (np.sum(target, axis=-1) + np.sum(pred, axis=-1) + smooth)
    dice = np.mean(dice)

    return dice


auc_score = roc_auc_score
kappa_score = cohen_kappa_score

if __name__ == '__main__':
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 1])

    sens = sensitive(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = f1_score(y_true, y_pred)

    print(sens)
    print(spec)
    print(prec)
    print(reca)
    print(fs)
