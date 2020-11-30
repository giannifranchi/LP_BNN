import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import anom_utils
import torch
import torch.nn as nn

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist


class Metrics_confidnet:
    def __init__(self, metrics, len_dataset, n_classes):
        self.metrics = metrics
        self.len_dataset = len_dataset
        self.n_classes = n_classes
        self.accurate, self.errors, self.proba_pred = [], [], []
        self.accuracy = 0
        self.current_miou = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, pred, target, confidence):
        self.accurate.extend(pred.eq(target.view_as(pred)).detach().to("cpu").numpy())
        self.accuracy += pred.eq(target.view_as(pred)).sum().item()
        self.errors.extend((pred != target.view_as(pred)).detach().to("cpu").numpy())
        self.proba_pred.extend(confidence.detach().to("cpu").numpy())

        if "mean_iou" in self.metrics:
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            mask = (target >= 0) & (target < self.n_classes)
            hist = np.bincount(
                self.n_classes * target[mask].astype(int) + pred[mask],
                minlength=self.n_classes ** 2,
            ).reshape(self.n_classes, self.n_classes)
            self.confusion_matrix += hist

    def get_scores(self, split="train"):
        self.accurate = np.reshape(self.accurate, newshape=(len(self.accurate), -1)).flatten()
        self.errors = np.reshape(self.errors, newshape=(len(self.errors), -1)).flatten()
        self.proba_pred = np.reshape(self.proba_pred, newshape=(len(self.proba_pred), -1)).flatten()
        #print('GIANNI' , self.metrics)
        scores = {}
        if "accuracy" in self.metrics:
            accuracy = self.accuracy / self.len_dataset
            scores[str(split)+"/accuracy"] = {"value": accuracy, "string": str(accuracy)}
            #print(str(split)+"/accuracy",accuracy)
        if "auc" in self.metrics:
            if len(np.unique(self.accurate)) == 1:
                auc = 1
            else:
                auc = roc_auc_score(self.accurate, self.proba_pred)
            scores[str(split)+"/auc"] = {"value": auc, "string": str(auc)}
            #print(str(split) + "/auc", auc)
        if "ap_success" in self.metrics:
            ap_success = average_precision_score(self.accurate, self.proba_pred)
            scores[str(split)+"/ap_success"] = {"value": ap_success,"string": str(ap_success)}
            #print(str(split)+"/ap_success", ap_success)
        if "accuracy_success" in self.metrics:
            accuracy_success = np.round(self.proba_pred[self.accurate == 1]).mean()
            scores[str(split)+"/accuracy_success"] = {
                "value": accuracy_success,
                "string":str(accuracy_success),
            }
           # print(str(split) + "/accuracy_success", accuracy_success)
        if "ap_errors" in self.metrics:
            ap_errors = average_precision_score(self.errors, -self.proba_pred)
            scores[str(split)+"/ap_errors"] = {"value": ap_errors, "string": str(ap_errors)}
            #print(str(split) + "/ap_errors", ap_errors)
        if "accuracy_errors" in self.metrics:
            accuracy_errors = 1.0 - np.round(self.proba_pred[self.errors == 1]).mean()
            scores[str(split)+"/accuracy_errors"] = {
                "value": accuracy_errors,
                "string": str(accuracy_errors),
            }
        if "fpr_at_95tpr" in self.metrics:
            for i,delta in enumerate(np.arange(
                self.proba_pred.min(),
                self.proba_pred.max(),
                (self.proba_pred.max() - self.proba_pred.min()) / 10000,
            )):
                tpr = len(self.proba_pred[(self.accurate == 1) & (self.proba_pred >= delta)]) / len(
                    self.proba_pred[(self.accurate == 1)]
                )
                if i%100 == 0:
                    print("Thresholde: " + str(delta))
                    print("TPR: " + str(tpr))
                    print("------")
                if 0.9505 >= tpr >= 0.9495:
                    print("Nearest threshold 95% TPR value: "+ str(tpr))

                    print("Threshold 95% TPR value: "+ str(delta))
                    fpr = len(
                        self.proba_pred[(self.errors == 1) & (self.proba_pred >= delta)]
                    ) / len(self.proba_pred[(self.errors == 1)])
                    scores[str(split)+"/fpr_at_95tpr"] = {"value": fpr, "string": str(fpr)}
                    break

        if "mean_iou" in self.metrics:
            iou = np.diag(self.confusion_matrix) / (
                self.confusion_matrix.sum(axis=1)
                + self.confusion_matrix.sum(axis=0)
                - np.diag(self.confusion_matrix)
            )
            mean_iou = np.nanmean(iou)
            scores[str(split)+"/mean_iou"] = {"value": mean_iou, "string":  str(mean_iou)}
            #print(str(split) + "/mean_iou", mean_iou)

        return scores


def eval_ood_measure(conf, seg_label, pred, out_labels=[10], mask=None):
    ECE = _ECELoss()
    pred_ID=pred[seg_label!=out_labels[0]]
    seg_label_ID = seg_label[seg_label != out_labels[0]]
    conf_ID=conf[seg_label != out_labels[0]]
    ece_out = ECE.forward(conf_ID,pred_ID,seg_label_ID)
    conf=conf.cpu().numpy()
    seg_label = seg_label.cpu().numpy()
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]

    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]
    print('out_label',np.sum(out_label) , np.mean(out_scores), np.mean(in_scores))
    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr, ece_out.cpu().item()
    else:
        print("This image does not contain any OOD pixels or is only OOD.",out_labels )
        return None


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, predictions, labels):

        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece