#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3


# General imports
import os
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
import seaborn as sns

# Machine learning imports
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
from keras.layers import Lambda
from keras.backend import argmax, cast

from sklearn.base import BaseEstimator, TransformerMixin #,ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV

from pyriemann.estimation import Covariances
#from pyriemann.spatialfilters import CSP

import pandas as pd
from joblib import dump, load
import pickle


# MOABB imports
import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery
from moabb.pipelines.utils import FilterBank


# Local imports
from utils.models.tcnet import model_tcnet
from utils.models.shallow_cnn import Shallow_CNN
from utils.models.deep_cnn import Deep_CNN
from utils.models.lstm import LSTM
from utils.utils import auc, auc_score, Estimator
from utils.local_paradigms import LeftRightImageryAccuracy,FilterBankLeftRightImageryAccuracy


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")
    

#############################################################################
# MOABB application

## Parameters
classes = 2 #4
channels = 22
sp = 1001
batch_size = 64
epochs = 750
lr = 0.01
loss = 'sparse_categorical_crossentropy' #categorical_crossentropy
opt = Adam(lr=lr)
met = ['accuracy'] #auc(for roc_auc metrics used for two classes in MOABB) #accuracy

## Making pipelines
print("Making pipelines ")
pipelines={}
clf_shallow_cnn = Shallow_CNN('shallow_cnn')
clf_shallow_cnn = clf_shallow_cnn.create_model(classes, loss=loss, opt=opt, met=met)
pipe = make_pipeline(Estimator(clf_shallow_cnn,'shallow_cnn', batch_size))
pipelines['shallow_cnn'] = pipe
clf_deep_cnn = Deep_CNN('deep_cnn')
clf_deep_cnn = clf_deep_cnn.create_model(classes, loss=loss, opt=opt, met=met)
pipe = make_pipeline(Estimator(clf_deep_cnn,'deep_cnn', batch_size))
pipelines['deep_cnn'] = pipe
clf_lstm = LSTM('lstm')
clf_lstm = clf_lstm.create_model(classes, input_shape=(channels,sp), loss=loss, opt=opt, met=met)
pipe = make_pipeline(Estimator(clf_lstm,'lstm', batch_size))
pipelines['lstm'] = pipe
clf_tcnet = model_tcnet(classes, channels, sp, epochs, loss, opt, met)
pipe = make_pipeline(Estimator(clf_tcnet, 'tcnet', batch_size))
pipelines['tcnet'] = pipe

pipelines_fb={}
clf = LDA()
fbcsp = FilterBank(CSP(n_components=4))
pipe = make_pipeline(fbcsp, clf)
pipelines_fb['fbcsp_lda'] = pipe

## Specifying datasets, paradigm and evaluation
print("Specifying datasets, paradigms and evaluation ")
datasets = [BNCI2014001()]
#datasets[0].subject_list = datasets[0].subject_list[:2]
paradigm = LeftRightImageryAccuracy() #2 classes (right and left hands) with accuracy metric #MotorImagery(events=["left_hand", "right_hand", "feet", "tongue"], n_classes=4) #LeftRightImagery()
evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=False)

filters = [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]
paradigm_fb = FilterBankLeftRightImageryAccuracy(filters=filters)
evaluation_fb = WithinSessionEvaluation(paradigm=paradigm_fb, datasets=datasets, overwrite=False)

## Getting and saving results
print("Calculating results ")
results = evaluation.process(pipelines)
results_fb = evaluation_fb.process(pipelines_fb)
if not os.path.exists("./results"):
    os.mkdir("./results")
results.to_csv("./results/results_benchmark.csv")
results_fb.to_csv("./results/results_fb_benchmark.csv")
results = pd.read_csv("./results/results_benchmark.csv")
