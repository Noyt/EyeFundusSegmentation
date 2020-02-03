from sklearn.metrics import *
import numpy as np

class Metrics:
    def __init__(self):
        self.shape = None

    def auc_precision_recall_curve(self, y_true, probas_pred):
        precision, recall, t = precision_recall_curve(y_true, probas_pred)
        precision, indices = np.unique(precision, return_index=True)
        recall = recall[indices]
        return auc(precision, recall)

