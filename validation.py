import sklearn.metrics as mt
import matplotlib.pyplot as plt


def pr_curve(preds, labels, ax):
    """
    Computes the precision and recall values for different threshold values
    and plots the PR curve. Also computes the AP score using the precision and
    recall values.

    :param preds: Prediction array.
    :param labels: Labels corresponding to the predictions.
    :param ax: Figure object used for plotting.
    :return: None
    """
    labels[labels >= 1] = 1                     # Binarize the labels.
    labels[labels < 1] = 0                      # Binarize the labels.
    labels = labels.numpy().flatten()           # Flatten the label tensor.
    preds = preds.detach().numpy().flatten()    # Flatten the prediction tensor.

    p, r, t = mt.precision_recall_curve(labels, preds)  # Compute the precision and recall for thresholds.
    ap = mt.average_precision_score(labels, preds)      # Compute the AP score for the given label and predictions.

    ax.clear()
    ax.plot(r, p)                       # Plot the PR curve.
    ax.set_title("AP: {}".format(ap))   # Print the AP score on the PR curve as title.
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.draw()
    plt.pause(0.02)
