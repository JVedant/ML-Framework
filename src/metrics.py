import numpy as np
from sklearn.metrics import roc_auc_score

class CLFMetrics:
    def __init__(self):
        self.self.tp = 0
        self.self.tn = 0
        self.self.fp = 0
        self.self.fn = 0

    def true_positive(self, y_true, y_pred):
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == 1 and y_p == 1:
                self.tp += 1
        return self.tp / len(y_true)

    def true_negative(self, y_true, y_pred):
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == 0 and y_p == 0:
                self.tn += 1
        return self.tn / len(y_true)

    def false_positive(self, y_true, y_pred):
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == 0 and y_p == 1:
                self.fp += 1
        return self.fp / len(y_true)

    def false_negative(self, y_true, y_pred):
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == 1 and y_p == 0:
                self.fn += 1
        return self.fn / len(y_true)


class ClassificationMetrics:
    def __init__(self):

        self.clf = CLFMetrics()
        self.metrics = {
            "accuracy": self._accuracy,
            "precision": self._precision,
            "logloss": self._logloss,
            "f1": self._f1,
            "recall": self._recall,
            "auc": self._auc
        }

    def __call__(self, metrics, y_true, y_pred, y_proba=None):
        if metrics not in self.metrics:
            raise Exception("Metrics not Identified")
        if metrics == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_pred)
            else:
                raise Exception("y_proba can't be None for auc")
        elif metrics == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_pred)
            else:
                raise Exception("y_proba can't be None for logloss")
        else:
            return self.metrics[metrics](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        true_pos = self.clf.true_positive(y_true, y_pred)
        true_neg = self.clf.true_negative(y_true, y_pred)
        false_pos = self.clf.false_positive(y_true, y_pred)
        false_neg = self.clf.false_negative(y_true, y_pred)
        acc = (true_pos + true_neg) / (true_neg + true_pos + false_neg + false_pos)
        return acc

    @staticmethod
    def _precision(y_true, y_pred):
        true_pos = self.clf.true_positive(y_true, y_pred)
        false_pos = self.clf.false_positive(y_true, y_pred)
        prec = true_pos / (true_pos + false_pos)
        return prec

    @staticmethod
    def _logloss(y_true, y_pred):
        epsilon = 1e-15
        loss = []
        for y_t, y_p in zip(y_true, y_pred):
            y_p = np.clip(y_p, epsilon, 1 - epsilon)
            temp_loss = -1.0 * (y_t * np.log(y_p) + (1-y_t) * np.log(1 - y_p))
            loss.append(temp_loss)
        return np.mean(loss)

    @staticmethod
    def _recall(y_true, y_pred):
        true_pos = self.clf.true_positive(y_true, y_pred)
        false_neg = self.clf.false_negative(y_true, y_pred)
        rec = true_pos / (true_pos + false_neg)
        return rec

    @staticmethod
    def _f1(y_true, y_pred):
        true_pos = self.clf.true_positive(y_true, y_pred)
        false_pos = self.clf.false_positive(y_true, y_pred)
        false_neg = self.clf.false_negative(y_true, y_pred)
        f1 = (2*true_pos) / ((2*true_pos) + false_pos + false_neg)
        return f1

    @staticmethod
    def _auc(y_true, y_pred):

        # Todo: create a logic to represent AUC
        return metrics.roc_auc_score(y_true=y_true, y_pred=y_pred)










'''
############################ Depricate ############################
class ClassificationMetrics:
    def __init__(self):

        self.clf = CLFMetrics()
        self.metrics = {
            "accuracy": self._accuracy,
            "precision": self._precision,
            "logloss": self._logloss,
            "f1": self._f1,
            "recall": self._recall,
            "auc": self._auc
        }

    def __call__(self, metrics, y_true, y_pred, y_proba=None):
        if metrics not in self.metrics:
            raise Exception("Metrics not Identified")
        if metrics == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_pred)
            else:
                raise Exception("y_proba can't be None for auc")
        elif metrics == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_pred)
            else:
                raise Exception("y_proba can't be None for logloss")
        else:
            return self.metrics[metrics](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return metrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return metrics.log_loss(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _recall(y_true, y_pred):
        return metrics.recall_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return metrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return metrics.roc_auc_score(y_true=y_true, y_pred=y_pred)'''