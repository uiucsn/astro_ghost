import numpy as np
from astropy.table import Table
import os
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import csv
from astropy.io import ascii
from astropy.table import Table
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
#from astro_ghost import PS1QueryFunctions as ps1
from astropy.utils.data import get_pkg_data_filename
import pkg_resources
from matplotlib import colors
from imblearn.under_sampling import RandomUnderSampler
from scipy import ndimage
from astropy.wcs import WCS
from matplotlib.pyplot import figure
import pickle
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import seaborn as sns
from collections import Counter
from imblearn.datasets import make_imbalance
import random
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from imblearn.pipeline import Pipeline
from rfpimp import *

def classify(dataML, verbose=True):
    feature_list, dataML_preprocessed, labels_df2, names = preprocess_dataframe(dataML)
    dataML_matrix_scaled = preprocessing.scale(dataML_preprocessed)
    rf = loadClassifier()
    class_predictions = rf.predict(dataML_matrix_scaled)
    if verbose:
        for i in np.arange(len(class_predictions)):
            print("%s is predicted to be a %s." % (names[i], class_predictions[i]))
    return class_predictions

def loadClassifier(verbose=True):
    modelName = "BinarySNClassifier.sav"
    stream = pkg_resources.resource_stream(__name__, modelName)
    if verbose:
        print("Loading model %s."%modelName)
    model = pickle.load(stream)
    return model

def plot_ROC_wCV_wMandel(foleySN_matrix_imputed, foleylabels, dataML_matrix_imputed, labels, save=1, balance=1):
    sns.set_context('paper')
    fig = plt.figure(figsize=(10,10), frameon=False)
    ax = plt.gca()
    accuracyF = plot_ROC_wCV(ax, foleySN_matrix_imputed, foleylabels, save=1, balance=balance)
    accuracyM = plot_ROC_wCV(ax, dataML_matrix_imputed, labels, save=0, balance=balance)
    plt.xlabel("False Positive Rate", fontsize=16);
    plt.ylabel("True Positive Rate", fontsize=16);
    plt.legend(loc=4, fontsize=20)

    if save:
        plt.savefig("Combined_MeanROC_Curve_%i_Classes_dataML.png" % len(classes))
    else:
        plt.show()

def plot_ROC_wCV(ax, X, y, names, save=True, balance=True):
    sns.set_context("paper")
    nsplit = 5
    cv = StratifiedKFold(n_splits=nsplit)
    classes = np.unique(y)
    colors = plt.cm.Dark2(np.linspace(0, 1, len(classes)))
    rf = RandomForestClassifier(n_estimators=1400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=90, bootstrap=False)
    tprs = []
    allAcc = []
    aucs = []
    all_confMatrices = []
    mean_fpr = np.linspace(0, 1, 100)
    accuracy_tot = 0
    nclass = len(classes)
    wrong = []
    for j in range(nclass):
        i = 0
        for train, test in cv.split(X, y):
            names_test = names[test]
            if nclass == 3:
                sampling1={'SN Ia': Counter(y[train])['SN Ia'], 'Core Collapse': Counter(y[train])['Core Collapse'], 'SLSN': 1000}
                sampling2={'SN Ia': 1000, 'SLSN': 1000, 'Core Collapse': 1000}
            elif nclass == 4:
                sampling1={'SN Ia':Counter(y[train])['SN Ia'], 'Core Collapse':Counter(y[train])['Core Collapse'] , 'SN Ia Pec': 500, 'SLSN': 500}
                sampling2={'SN Ia': 1000, 'Core Collapse': 1000, 'SN Ia Pec': 500, 'SLSN': 500}
            elif nclass == 2:
                sampling1={'SN Ia': Counter(y[train])['SN Ia'], 'Core Collapse': 3500}
                sampling2={'SN Ia': 3500, 'Core Collapse': 3500}
            if balance:
                over = SMOTE(sampling_strategy=sampling1)
                under = RandomUnderSampler(sampling_strategy=sampling2)
                steps = [('o', over), ('u', under)]
                pipeline = Pipeline(steps=steps)
                Xtrain_resampled, ytrain_resampled = pipeline.fit_resample(X[train], y[train])
            else:
                Xtrain_resampled = X[train]
                ytrain_resampled = y[train]
            print('Distribution after imbalancing: {}'.format(Counter(ytrain_resampled)))
            print('Distribution of test set: {}'.format(Counter(y[test])))

            probas_ = rf.fit(Xtrain_resampled, ytrain_resampled).predict_proba(X[test])
            predictions = rf.predict(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, j], pos_label=classes[j])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            i += 1
            tempAccuracy =  np.sum(predictions == y[test])/len(y[test])*100
            wrong.append(names_test[y[test] != predictions])
            print(tempAccuracy)
            allAcc.append(tempAccuracy)
            matr = sklearn.metrics.confusion_matrix(y[test], predictions, normalize='true')
            all_confMatrices.append(matr)
            print(matr)
            accuracy_tot += tempAccuracy
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        accuracy = accuracy_tot / (nsplit*len(classes))
        if True:
            if classes[j] == 'Core Collapse':
                ax.plot(mean_fpr, mean_tpr, color=colors[j],
                         label=r'CC (%0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                         lw=2, alpha=.8)
            elif classes[j] == 'SLSN':
                ax.plot(mean_fpr, mean_tpr, color=colors[j],
                         label=r'%s (%0.2f $\pm$ %0.2f)' % (classes[j], mean_auc, std_auc),
                         lw=2, alpha=.8)
            else:
                ax.plot(mean_fpr, mean_tpr, color=colors[j],
                         label=r'%s (%0.2f $\pm$ %0.2f)' % (classes[j].strip("SN "), mean_auc, std_auc),
                         lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[j], alpha=.05)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',alpha=.8)
    #if ~foley:
    ax.set_xlabel("False Positive Rate", fontsize=16);
    ax.set_ylabel("True Positive Rate", fontsize=16);
        #plt.title("ROC Curve, %i Classes" % (len(classes)), fontsize=26)
    ax.legend(loc=4,fontsize=12)
    #    plt.text(0.1, 0.9, r'$N_{tot} = %i$'%len(y),fontsize=12)
    plt.text(0.1, 0.9, r'$N_{train} = 7000$')
    plt.text(0.1, 0.82, r'$N_{test} = 2226$')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
        #plt.savefig("Combined_MeanROC_Curve_%i_Classes_dataML_noOverlapCuts.png" % len(classes))
    return accuracy, rf, all_confMatrices, allAcc, wrong


def plot_ROC(train_features, test_features, train_labels, test_labels, save):
    rf = RandomForestClassifier(n_estimators = 1000, bootstrap = True, max_features = 'sqrt')
    rf.fit(train_features, train_labels);

    predictions = rf.predict(test_features)# Calculate the absolute errors

    fpr = dict()
    tpr = dict()
    ROC = dict()
    classes = np.unique(test_labels)
    plt.figure(figsize=(10,10))
    for i in range(len(classes)):
        rf_probs = rf.predict_proba(test_features)[:, i]
        fpr[i], tpr[i], _ = roc_curve(test_labels, rf_probs, pos_label=classes[i])
        ROC[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i],
                 lw=4, label='%s (area = %0.2f)' % (classes[i], ROC[i]))
    accuracy = np.sum(predictions == test_labels)/len(test_labels)*100
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc=4, fontsize=10)
    plt.xlabel("False Positive Rate", fontsize=16);
    plt.ylabel("True Positive Rate", fontsize=16);
    plt.title("ROC Curve, %i Classes, Accuracy = %.1f%%" % (len(classes), accuracy), fontsize=26)
    #os.chdir('/Users/alexgagliano/Documents/Research/Transient_ML/plots')
    if save:
        plt.savefig("ROC_Curve_%i_Classes_dataML.png" % len(classes))
    else:
        plt.show()
    return rf, predictions

#dataML = pd.read_csv("../database/GHOST.csv")

def condense_labels(dataML, nclass):
        # Labels are the values we want to predict
        dataML.loc[dataML['TransientClass'] == 'SN Ib\n SN Ib', 'TransientClass'] = 'SN Ib'
        dataML.loc[dataML['TransientClass'] == 'SN Ia\n SN Ia', 'TransientClass'] = 'SN Ia'
        dataML.loc[dataML['TransientClass'] == 'SN Ibn', 'TransientClass'] = 'SN Ib'
        dataML.loc[dataML['TransientClass'] == 'SN Ib', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'SN Ic', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'SLSN-I', 'TransientClass'] = 'SLSN'
        dataML.loc[dataML['TransientClass'] == 'SN I', 'TransientClass'] = 'SN I?'
        dataML.loc[dataML['TransientClass'] == 'SN Ib', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'SN Ic', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'SLSN-II', 'TransientClass'] = 'SLSN'
        dataML.loc[dataML['TransientClass'] == 'SLSN-II', 'TransientClass'] = 'SLSN'
        dataML.loc[dataML['TransientClass'] == 'CC', 'TransientClass'] = 'SN II'
        dataML.loc[dataML['TransientClass'] == 'II', 'TransientClass'] = 'SN II'
        dataML.loc[dataML['TransientClass'] == 'SLSN-I-R', 'TransientClass'] = 'SLSN'
        dataML.loc[dataML['TransientClass'] == 'SLSN-R', 'TransientClass'] = 'SLSN'
        dataML.loc[dataML['TransientClass'] == 'II/IIb', 'TransientClass'] = 'SN II'
        dataML.loc[dataML['TransientClass'] == 'II P', 'TransientClass'] = 'SN IIP'
        dataML.loc[dataML['TransientClass'] == 'Ib', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'Ic', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'II-p', 'TransientClass'] = 'SN II P'
        dataML.loc[dataML['TransientClass'] == 'II/LBV', 'TransientClass'] = 'SN II'
        dataML.loc[dataML['TransientClass'] == 'IIb', 'TransientClass'] = 'SN IIb'
        dataML.loc[dataML['TransientClass'] == 'Ic BL', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'Ia', 'TransientClass'] = 'SN Ia'
        dataML.loc[dataML['TransientClass'] == 'Ib/c', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'Ib/c', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'IIn', 'TransientClass'] = 'SN IIn'
        dataML.loc[dataML['TransientClass'] == 'Ibn', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'IIn Pec', 'TransientClass'] = 'SN IIn'
        dataML.loc[dataML['TransientClass'] == 'Ia/Ic', 'TransientClass'] = 'SN Ia/c'
        dataML.loc[dataML['TransientClass'] == 'SN II P', 'TransientClass'] = 'SN IIP'
        dataML.loc[dataML['TransientClass'] == 'Ia/c', 'TransientClass'] = 'SN Ia/c'
        dataML.loc[dataML['TransientClass'] == 'I', 'TransientClass'] = 'SN I'

        dataML.loc[dataML['TransientClass'] == 'LRV?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'II Pec?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'I?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'SLSN-II?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'IIb/Ib', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ib/Ic (Ca rich?)?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ic?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'IIn?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'PISN?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'SLSN?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ib/c?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'II?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'IIb?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ia?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'SN I?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ii', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'LBV to IIn', 'TransientClass'] = 'LBV'
        dataML.loc[dataML['TransientClass'] == 'II/Ib/c', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ca-rich', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'SN', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'SN Ia/c', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ib/IIb', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'IIn/LBV', 'TransientClass'] = 'IIn'
        dataML.loc[dataML['TransientClass'] == 'IIn', 'TransientClass'] = 'SN IIn'
        dataML.loc[dataML['TransientClass'] == 'IIn/LBV', 'TransientClass'] = 'IIn'
        dataML.loc[dataML['TransientClass'] == 'CV', 'TransientClass'] = 'Other'
        dataML.loc[dataML['TransientClass'] == 'Pec', 'TransientClass'] = 'Other'
        dataML.loc[dataML['TransientClass'] == 'LBV', 'TransientClass'] = 'Other'
        dataML.loc[dataML['TransientClass'] == 'IIn/LBV', 'TransientClass'] = 'IIn'
        dataML.loc[dataML['TransientClass'] == 'Ic Pec', 'TransientClass'] = 'Other'
        dataML.loc[dataML['TransientClass'] == 'CN', 'TransientClass'] = 'Other'
        dataML.loc[dataML['TransientClass'] == 'Ib-Ca', 'TransientClass'] = 'SN Ib/c'
        dataML.loc[dataML['TransientClass'] == 'SLSN-I?', 'TransientClass'] = 'SLSN'
        dataML.loc[dataML['TransientClass'] == 'SLSN-IIn', 'TransientClass'] = 'SLSN'
        dataML.loc[dataML['TransientClass'] == 'nIa', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'II L?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'Ia-09dc', 'TransientClass'] = 'SN Ia'
        dataML.loc[dataML['TransientClass'] == 'CC?', 'TransientClass'] = 'Unknown'
        dataML.loc[dataML['TransientClass'] == 'SN Ia-pec', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'SN Ia-91T-like', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'SN Iax[02cx-like]', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'SN Ia-91bg-like', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'SN Ia-CSM', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'SN Ia-91bg', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'Ia Pec', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'Ia*', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'Ia-02cx', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'Ia-91T', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'Ia-91bg', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'Ia-99aa', 'TransientClass'] = 'SN Ia Pec'#
        dataML.loc[dataML['TransientClass'] == 'Ia CSM', 'TransientClass'] = 'SN Ia Pec'#
        if nclass != 4:
            dataML = dataML[dataML['TransientClass'] != 'SN Ia Pec']

        # specific to the four-class
        if nclass == 5:
            dataML = dataML[dataML['TransientClass'] != 'SLSN']
            #dataML.loc[dataML['TransientClass'] == 'SN IIb', 'TransientClass'] = 'SN II'
            dataML = dataML[dataML['TransientClass'] != 'SN IIb']
            #dataML.loc[dataML['TransientClass'] == 'SN IIP', 'TransientClass'] = 'SN II'
            #dataML.loc[dataML['TransientClass'] == 'SN IIn', 'TransientClass'] = 'SN II'
            #II, IIP, Ia, IIn, Ib/c,
        # specific to the two-class
        elif nclass == 4:
            dataML.loc[dataML['TransientClass'] == 'SN II', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIP', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIb', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIn', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN Ib/c', 'TransientClass'] = 'Core Collapse'#
            #dataML = dataML[dataML['TransientClass'] != 'SN IIb']
            #dataML = dataML[dataML['TransientClass'] != 'SN Ib/c']
            #dataML = dataML[dataML['TransientClass'] != 'SN IIP']
            #dataML = dataML[dataML['TransientClass'] != 'SN IIn']
        elif nclass == 3:
            dataML.loc[dataML['TransientClass'] == 'SN II', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIP', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIb', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIn', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN Ib/c', 'TransientClass'] = 'Core Collapse'#

            #dataML = dataML[dataML['TransientClass'] != 'SN Ia']
            #dataML = dataML[dataML['TransientClass'] != 'Core Collapse']

            #SN II, SN Ib/c, SN Ia
        elif nclass == 2:
            dataML = dataML[dataML['TransientClass'] != 'SLSN']
            dataML.loc[dataML['TransientClass'] == 'SN II', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIP', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIb', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN IIn', 'TransientClass'] = 'Core Collapse'
            dataML.loc[dataML['TransientClass'] == 'SN Ib/c', 'TransientClass'] = 'Core Collapse'#

        dataML = dataML[dataML['TransientClass'] != 'SN II-pec']
        dataML = dataML[dataML['TransientClass'] != 'SLSN-I']
        dataML = dataML[dataML['TransientClass'] != 'SN Ic-BL']
        dataML = dataML[dataML['TransientClass'] != 'SN Ib-pec']
        dataML = dataML[dataML['TransientClass'] != 'SN Ib-Ca-rich']
        dataML = dataML[dataML['TransientClass'] != 'SN Ic-pec']
        dataML = dataML[dataML['TransientClass'] != 'SN IIL']
        dataML = dataML[dataML['TransientClass'] != 'II Pec']
        dataML = dataML[dataML['TransientClass'] != 'II L']
        dataML = dataML[dataML['TransientClass'] != 'Ib Pec']
        dataML = dataML[dataML['TransientClass'] != 'SN I']
        dataML = dataML[dataML['TransientClass'] != 'Other']
        dataML = dataML[dataML['TransientClass'] != 'Unknown']
        dataML = dataML[dataML['TransientClass'] != 'Ib-IIb']
        dataML = dataML[dataML['TransientClass'] != 'SNSN-II']

        return dataML

def preprocess_dataframe(dataML, nclass=2):
    dataML.replace(-999, np.nan, inplace=True)

    if 'TransientRedshift' in dataML.columns:
        trueIDXs = dataML.dropna(subset=['TransientRedshift', 'NED_redshift']).index
        naIDXs = set(dataML.index) - set(trueIDXs)

        dataML_na = dataML.loc[naIDXs]
        dataML_nona = dataML.loc[trueIDXs]

        pdiff = np.abs(dataML_nona['TransientRedshift'] - dataML_nona['NED_redshift'])/dataML_nona['TransientRedshift']*100
        dataML_nona = dataML_nona.loc[pdiff < 5]
        dataML = pd.concat([dataML_na, dataML_nona], ignore_index=True)

    #ADDING IN SNR IN TWO BANDS - I AND Z
    dataML["gSNR"] = 1/dataML["gApMagErr"]
    dataML["rSNR"] = 1/dataML["rApMagErr"]
    dataML["iSNR"] = 1/dataML["iApMagErr"]
    dataML["zSNR"] = 1/dataML["zApMagErr"]
    dataML["ySNR"] = 1/dataML["yApMagErr"]

    dataML = dataML.drop(['objAltName1', 'objAltName2','objAltName3'], axis=1)
    dataML = dataML.drop(['objName','uniquePspsOBid','ippObjID','surveyID','htmID','zoneID','tessID','projectionID','skyCellID'], axis=1)
    dataML = dataML.drop(['randomID','batchID','dvoRegionID','processingVersion','objInfoFlag','qualityFlag','raStack','decStack'], axis=1)
    dataML = dataML.drop(['raStackErr', 'decStackErr', 'raMean', 'decMean', 'raMeanErr', 'decMeanErr'], axis=1)
    dataML = dataML.drop(['gra', 'gdec', 'graErr', 'gdecErr', 'rra', 'rdec', 'rraErr', 'rdecErr','ira', 'idec', 'iraErr', 'idecErr','zra', 'zdec', 'zraErr', 'zdecErr','yra', 'ydec', 'yraErr', 'ydecErr'], axis=1)
    dataML = dataML.drop(['l','b','nStackObjectRows'],axis=1)
    dataML = dataML.drop(['nStackDetections','nDetections'],axis=1)
    dataML = dataML.drop(['gippDetectID', 'gstackDetectID', 'gstackImageID','rippDetectID', 'rstackDetectID', 'rstackImageID','iippDetectID', 'istackDetectID', 'istackImageID','zippDetectID', 'zstackDetectID', 'zstackImageID','yippDetectID', 'ystackDetectID', 'ystackImageID'], axis=1)
    dataML = dataML.drop(['bestDetection'],axis=1)
    dataML = dataML.drop(['epochMean'],axis=1)
    dataML = dataML.drop(['ng','nr','ni','nz'],axis=1)
    dataML = dataML.drop(['ny'],axis=1)
    dataML = dataML.drop(['uniquePspsSTid','primaryDetection','gEpoch'],axis=1)
    dataML = dataML.drop(['rEpoch','iEpoch','zEpoch', 'yEpoch'],axis=1)
    dataML = dataML.drop(['cx','cy'],axis=1)
    dataML = dataML.drop(['cz'],axis=1)
    dataML = dataML.drop(['lambda','beta'],axis=1)
    dataML = dataML.drop(['gpsfChiSq','rpsfChiSq','ipsfChiSq','zpsfChiSq','ypsfChiSq', 'ginfoFlag', 'ginfoFlag2', 'ginfoFlag3',  'rinfoFlag', 'rinfoFlag2', 'rinfoFlag3',  'iinfoFlag', 'iinfoFlag2', 'iinfoFlag3',  'zinfoFlag', 'zinfoFlag2', 'zinfoFlag3',  'yinfoFlag', 'yinfoFlag2', 'yinfoFlag3'],axis=1)
    dataML = dataML.drop(['gxPos', 'gxPosErr','rxPos', 'rxPosErr','ixPos', 'ixPosErr','zxPos', 'zxPosErr','yxPos', 'yxPosErr' ],axis=1)
    dataML = dataML.drop(['gyPos', 'gyPosErr','ryPos', 'ryPosErr','iyPos', 'iyPosErr','zyPos', 'zyPosErr','yyPos', 'yyPosErr' ],axis=1)
    dataML = dataML.drop(['gexpTime','rexpTime','iexpTime','zexpTime','yexpTime','gnFrames','rnFrames','inFrames','znFrames','ynFrames'],axis=1)
    dataML = dataML.drop(['gzp','rzp','izp','zzp','yzp'],axis=1)
    dataML = dataML.drop(['gPlateScale','rPlateScale','iPlateScale','zPlateScale','yPlateScale'],axis=1)
    dataML = dataML.drop(['posMeanChisq'],axis=1)
    dataML = dataML.drop(['gpsfQf','ipsfQf', 'zpsfQf', 'ypsfQf'], axis=1)
    dataML = dataML.drop(['gApFillFac', 'yApFillFac', 'iApFillFac', 'zApFillFac'], axis=1)
    dataML = dataML.drop(['gpsfQfPerfect', 'ipsfQfPerfect', 'zpsfQfPerfect', 'ypsfQfPerfect'], axis=1)
    dataML = dataML.drop(['gpsfTheta', 'ipsfTheta', 'zpsfTheta', 'ypsfTheta'], axis=1)
    dataML = dataML.drop(['gsky', 'isky', 'zsky', 'ysky'], axis=1)
    dataML = dataML.drop(['gskyErr', 'iskyErr', 'zskyErr', 'yskyErr'], axis=1)
    dataML = dataML.drop(['gpsfCore', 'ipsfCore', 'zpsfCore', 'ypsfCore'], axis=1)
    dataML = dataML.drop(['rpsfTheta', 'rsky', 'rskyErr', 'rpsfCore'], axis=1)
    dataML = dataML.drop(['gpsfLikelihood', 'rpsfLikelihood', 'ipsfLikelihood', 'zpsfLikelihood','ypsfLikelihood'], axis=1)
    dataML = dataML.drop(['rpsfQf'], axis=1)
    #dataML = dataML.drop(['host_logmass', 'host_logmass_min', 'host_logmass_max','Hubble Residual', 'Transient AltName'],axis=1)
    dataML = dataML.drop(['rpsfQfPerfect'], axis=1)
    dataML = dataML.drop(['rApFillFac'], axis=1)
    dataML = dataML.drop(['TransientRA', 'TransientDEC','NED_type', 'NED_name', 'class'], axis=1)

    #try dropping NED info now:
    dataML = dataML.drop(['NED_vel', 'NED_mag'], axis=1)
    dataML = dataML.drop(['NED_redshift'], axis=1)
    #dataML = dataML.drop(['TransientRedshift'], axis=1)
    #dataML.drop(['TransientDiscoveryDate',  'TransientDiscoveryMag', 'TransientDiscoveryYear'], axis=1, inplace=True)
    dataML = dataML.drop(['objID'],axis=1)

    #dataML

    dataML = condense_labels(dataML, nclass=nclass)
    names = dataML['TransientName']
    dataML = dataML.drop(['TransientName'], axis=1)
    labels_df = dataML['TransientClass']# Remove the labels from the features
    labels = np.array(labels_df)
    classes = np.unique(labels)

    #order in the same way the classifier training data was labeled
    dataML = dataML[['gPSFMag', 'gPSFMagErr', 'gApMag', 'gApMagErr', 'gKronMag',
       'gKronMagErr', 'gpsfMajorFWHM', 'gpsfMinorFWHM', 'gmomentXX',
       'gmomentXY', 'gmomentYY', 'gmomentR1', 'gmomentRH', 'gPSFFlux',
       'gPSFFluxErr', 'gApFlux', 'gApFluxErr', 'gApRadius', 'gKronFlux',
       'gKronFluxErr', 'gKronRad', 'gExtNSigma', 'rPSFMag', 'rPSFMagErr',
       'rApMag', 'rApMagErr', 'rKronMag', 'rKronMagErr', 'rpsfMajorFWHM',
       'rpsfMinorFWHM', 'rmomentXX', 'rmomentXY', 'rmomentYY',
       'rmomentR1', 'rmomentRH', 'rPSFFlux', 'rPSFFluxErr', 'rApFlux',
       'rApFluxErr', 'rApRadius', 'rKronFlux', 'rKronFluxErr', 'rKronRad',
       'rExtNSigma', 'iPSFMag', 'iPSFMagErr', 'iApMag', 'iApMagErr',
       'iKronMag', 'iKronMagErr', 'ipsfMajorFWHM', 'ipsfMinorFWHM',
       'imomentXX', 'imomentXY', 'imomentYY', 'imomentR1', 'imomentRH',
       'iPSFFlux', 'iPSFFluxErr', 'iApFlux', 'iApFluxErr', 'iApRadius',
       'iKronFlux', 'iKronFluxErr', 'iKronRad', 'iExtNSigma', 'zPSFMag',
       'zPSFMagErr', 'zApMag', 'zApMagErr', 'zKronMag', 'zKronMagErr',
       'zpsfMajorFWHM', 'zpsfMinorFWHM', 'zmomentXX', 'zmomentXY',
       'zmomentYY', 'zmomentR1', 'zmomentRH', 'zPSFFlux', 'zPSFFluxErr',
       'zApFlux', 'zApFluxErr', 'zApRadius', 'zKronFlux', 'zKronFluxErr',
       'zKronRad', 'zExtNSigma', 'yPSFMag', 'yPSFMagErr', 'yApMag',
       'yApMagErr', 'yKronMag', 'yKronMagErr', 'ypsfMajorFWHM',
       'ypsfMinorFWHM', 'ymomentXX', 'ymomentXY', 'ymomentYY',
       'ymomentR1', 'ymomentRH', 'yPSFFlux', 'yPSFFluxErr', 'yApFlux',
       'yApFluxErr', 'yApRadius', 'yKronFlux', 'yKronFluxErr', 'yKronRad',
       'yExtNSigma', 'i-z', 'g-r', 'r-i', 'g-i', 'z-y', 'g-rErr',
       'r-iErr', 'i-zErr', 'z-yErr', 'gApMag_gKronMag', 'rApMag_rKronMag',
       'iApMag_iKronMag', 'zApMag_zKronMag', 'yApMag_yKronMag', '7DCD',
       'dist/DLR', 'dist', 'gSNR', 'rSNR', 'iSNR', 'zSNR', 'ySNR', 'TransientClass']]

    dataML.dropna(axis=0, inplace=True)

    feature_list = list(dataML.columns) # Convert to numpy array

    dataML_noLabels = dataML.drop('TransientClass', axis=1)

    print('Distribution before imbalancing: {}'.format(Counter(labels)))

    labels_df.reset_index(inplace=True, drop=True)
    return feature_list, dataML_noLabels, labels_df, names

#feature_list, dataML_preprocessed2, labels_df2, names = preprocess_dataframe(dataML, nclass=2, PCA=False)
#dataML_matrix_scaled = preprocessing.scale(dataML_preprocessed2)
#labels = labels_df2.values

#acc, rf, all_confMatrices, accTot, wrong = plot_ROC_wCV(ax, dataML_matrix_scaled, labels.ravel(), names.values, save=0, balance=True)
