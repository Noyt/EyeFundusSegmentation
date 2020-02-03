from sklearn.metrics import *
import numpy as np
import warnings
from .metrics import Metrics
from tqdm import tqdm
from skimage import measure
from scipy.spatial import distance


class Tester:
    def __init__(self):

        m = Metrics()
        self.hard_metrics = {'Accuracy': accuracy_score,
                             'Kappa': cohen_kappa_score,
                             'Precision': precision_score,
                             'Recall': recall_score,
                             'F1': f1_score,
                             'Jaccard':  jaccard_similarity_score
                             }
        self.soft_metrics = {'ROC AUC': roc_auc_score,
                             'Precision/Recall AUC': m.auc_precision_recall_curve}
        self._metrics = None

    #@property
    def metrics(self, path=None, name=None):
        if self._metrics is not None:
            for l in self._metrics:
                print(10*'*', l, 10*'*')
                for _ in self._metrics[l]:
                    print(_, self._metrics[l][_])
        if path is not None:
            complete_name = path + name + ".txt"
            f = open(complete_name, "w+")
            for l in self._metrics:
                f.write("********** {} ********** \n".format(l))
                for _ in self._metrics[l]:
                    f.write(_ + ' ')
                    f.write("{} \n".format(self._metrics[l][_]))
            f.close()

    def evaluate(self, dataset,
                 pred_col,
                 gt_col,
                 pred_index=None,
                 gt_index=None,
                 per_image=True,
                 pred_labels=None):
        """
        :param dataset: Dataset containing both the prediction and the groundtruth
        :param pred_col: Prediction column, expected to be expressed as probabilities CxHxW where C is the number of class
        :param gt_col: Groundtruth column
        :param pred_index: scalar 0<=i<C indicating the class you want to evaluate. If None, score are calculated for all
        classes
        :param gt_index:: Corresponding class in groundtruth. If None, pred_index is taken
        :return:
        """

        if pred_index is not None and not isinstance(pred_index, list):
            pred_index = [pred_index]
        if gt_index is not None and not isinstance(gt_index, list):
            gt_index = [gt_index]
            assert len(pred_index) == len(gt_index), "You need to pass the same number of groundtruth classes than " \
                                                         "predicted ones"
        if gt_index is None:
            gt_index = pred_index

        # Instantiate the result dictionary
        results = dict()

        def classes_pred(hard_pred, c_pred):
            if not isinstance(c_pred, tuple):
                c_pred = [c_pred]
            pred = hard_pred == c_pred[0]
            for c in c_pred:
                pred = np.logical_or(pred, hard_pred==c)
            return pred.flatten()

        if pred_index is not None:
            for m in list(self.hard_metrics.keys()) + list(self.soft_metrics.keys()):
                results[m] = {_:[] for _ in pred_index}
        else:
            for m in list(self.hard_metrics.keys()) + list(self.soft_metrics.keys()):
                results[m] = []

        if per_image:
            gen = dataset.generator(n=1, columns=[pred_col, gt_col])
            i = -1
            for b in tqdm(gen):
                i += 1
                pred = np.squeeze(b[pred_col])
                gt = np.squeeze(b[gt_col])
                probs = pred.dtype == np.float32
                if probs:
                    for metric in self.soft_metrics:

                        if pred_index is not None:
                            for c_pred, c_gt in zip(pred_index, gt_index):
                                if isinstance(c_pred, tuple):
                                    soft_pred = np.sum([pred[_] for _ in c_pred], axis=0).flatten()
                                else:
                                    soft_pred = pred[c_pred].flatten()

                                # soft_pred = np.vstack((1-soft_pred, soft_pred))
                                try:
                                    results[metric][c_pred].append(self.soft_metrics[metric]((gt == c_gt).flatten(), soft_pred))
                                except ValueError:
                                    warnings.warn('Only one class in image %i; replacing AUC by accuracy score ' % i)
                                    results[metric][c_pred].append(accuracy_score(gt == c_gt, classes_pred(hard_pred,
                                                                                                           c_pred)))
                        else:
                            soft_pred = pred.reshape((pred.shape[0], np.prod(pred.shape[1:])))
                            results[metric].append(self.soft_metrics[metric](gt.flatten(), soft_pred))

                if probs:
                    hard_pred = np.argmax(pred, 0)

                else:
                    hard_pred = pred
                assert hard_pred.shape == gt.shape, "Wrong dimensions, prediction has shape %s whereas groundtruth" \
                                                    "has shape %s%" % (hard_pred.shape, gt.shape)
                for metric in self.hard_metrics:
                    if pred_index is not None:
                        for c_pred, c_gt in zip(pred_index, gt_index):
                            results[metric][c_pred].append(self.hard_metrics[metric]((gt == c_gt).flatten(),
                                                                                    classes_pred(hard_pred, c_pred)))
                    else:
                        results[metric].append(self.hard_metrics[metric](gt.flatten(), hard_pred.flatten()))
        else:
            raise NotImplementedError

        final_results = dict()
        if pred_labels is not None:
            for c, l in zip(pred_index, pred_labels):
                final_results[l] = {_: np.mean(results[_][c],0) for _ in list(self.hard_metrics.keys())
                                    +list(self.soft_metrics.keys())}
        else:
            if pred_index is not None:
                for c in pred_index:
                    final_results[c] = {_: np.mean(results[_][c], 0) for _ in list(self.hard_metrics.keys())
                                        +list(self.soft_metrics.keys())}
            else:
                final_results['global'] = {_: np.mean(results[_], 0) for _ in list(self.hard_metrics.keys())
                                        +list(self.soft_metrics.keys())}

        if self._metrics is not None:
            self._metrics = {**self._metrics, **final_results}
        else:
            self._metrics = final_results

    def detection_eval(self, dataset,
                pred_col,
                coordinate_columns,
                pred_index=None,
                mask_gt_col=None,
                gt_index=None,
                per_image=True,
                pred_labels=None):

        if pred_index is None:
            warnings.warn('No index was specified for the prediction, taking 1 as default value')
            pred_index = 1

        if mask_gt_col is not None:
            self.evaluate(dataset, pred_col, mask_gt_col, pred_index, gt_index, per_image, pred_labels)

        print('Evaluating Euclidean Distance between center of gt and pred')
        gen = dataset.generator(n=1, columns=[pred_col, coordinate_columns])

        results = []
        for b in tqdm(gen):
            coord = np.squeeze(b[coordinate_columns]).flatten()
            pred = np.squeeze(b[pred_col])
            if pred.dtype == np.float32:
                pred = np.argmax(pred, 0)

            try:

                props = measure.regionprops((pred==pred_index).astype(np.uint8))
                center = props[0].centroid  #row x col
            except IndexError:
                continue
            results.append(distance.euclidean(center, coord))
        if pred_labels is None:
            pred_labels = pred_index
        if self._metrics is None:
            self._metrics = {pred_labels:{}}

        self._metrics[pred_labels]['Center point euclidean distance'] = np.mean(results)

























