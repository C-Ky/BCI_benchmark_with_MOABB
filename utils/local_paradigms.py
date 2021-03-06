import abc
import logging

from moabb.datasets import utils
from moabb.paradigms.motor_imagery import SinglePass, FilterBank

from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score


log = logging.getLogger(__name__)# MOABB imports

class LeftRightImageryAccuracy(SinglePass):
    """Motor Imagery for left hand/right hand classification
    Metric is 'accuracy'
    """

    def __init__(self, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("LeftRightImageryAccuracy dont accept events"))
        super().__init__(events=["left_hand", "right_hand"], **kwargs)


    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return "accuracy"

class FilterBankLeftRightImageryAccuracy(FilterBank):
    """Filter Bank Motor Imagery for left hand/right hand classification
    Metric is 'accuracy'
    """

    def __init__(self, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("FilterBankLeftRightImageryAccuracy dont accept events"))
        super().__init__(events=["left_hand", "right_hand"], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return "accuracy"
