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
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pyriemann.estimation import Covariances

import pandas as pd
from joblib import dump, load
import pickle


# MOABB imports
import moabb
from moabb.evaluations import CrossSessionEvaluation
from moabb.pipelines.utils import FilterBank


# Local imports
from utils.models.tcnet import model_tcnet
from utils.models.shallow_cnn import Shallow_CNN
from utils.models.deep_cnn import Deep_CNN
from utils.models.lstm import LSTM
from utils.utils import Estimator
from utils.local_paradigms import UD_LR_ImageryAccuracy,FilterBank_UD_LR_ImageryAccuracy
from utils.SMR_202104 import SMR_202104


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")
    

#############################################################################
# MOABB application

## Parameters
classes = 2 #4
channels = 60
sp = 4001
batch_size = 64
epochs = 750
lr = 0.01
loss = 'sparse_categorical_crossentropy'
opt = Adam(lr=lr)
met = ['accuracy']

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
fbcsp = FilterBank(CSP())
pipe = make_pipeline(fbcsp, clf)
pipelines_fb['fbcsp_lda'] = pipe

## Specifying datasets, paradigm and evaluation
print("Specifying datasets, paradigms and evaluation ")
datasets = [SMR_202104()]
paradigm = UD_LR_ImageryAccuracy() 
evaluation = CrossSubjectEvaluation(paradigm=paradigm, datasets=datasets, overwrite=False)

filters = [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]]
paradigm_fb = FilterBank_UD_LR_ImageryAccuracy(filters=filters)
evaluation_fb = CrossSubjectEvaluation(paradigm=paradigm_fb, datasets=datasets, overwrite=False)

## Getting and saving results
print("Calculating results ")
results = evaluation.process(pipelines)
results_fb = evaluation_fb.process(pipelines_fb)
if not os.path.exists("./results"):
    os.mkdir("./results")
results.to_csv("./results/results_benchmark.csv")
results_fb.to_csv("./results/results_fb_benchmark.csv")
