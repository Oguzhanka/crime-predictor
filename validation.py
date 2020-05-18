import sklearn.metrics as mt
import matplotlib.pyplot as plt


def pr_curve(preds, labels):
    """

    :param preds:
    :param labels:
    :return:
    """
    labels[labels >= 1] = 1
    labels[labels < 1] = 0
    labels = labels.numpy().flatten()
    preds = preds.detach().numpy().flatten()

    p, r, t = mt.precision_recall_curve(labels, preds)
    ap = mt.average_precision_score(labels, preds)

    plt.plot(r, p)
    plt.title("AP: {}".format(ap))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
