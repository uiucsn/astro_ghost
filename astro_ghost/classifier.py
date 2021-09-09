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
from astropy.utils.data import get_pkg_data_filename
import astro_ghost
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
import random
from sklearn import preprocessing

def downloadClassifier(fname='BinarySNClassifier.sav'):
    url = 'http://ghost.ncsa.illinois.edu/static/BinarySNClassifier.sav'
    response = requests.get(url, stream=True)

    install_path = astro_ghost.__file__
    install_path = install_path.split("/")[:-1]
    install_path = "/".join(install_path)
    fullPath = install_path + "/" + fname

    if response.status_code == 200:
        with open(fullPath, 'wb') as f:
            f.write(response.raw.read())
    print("Binary classification model downloaded.")
    return

def classify(dataML, verbose=True):
    downloadClassifier()
    feature_list, dataML_preprocessed, labels_df2, names = preprocess_dataframe(dataML)
    dataML_matrix_scaled = preprocessing.scale(dataML_preprocessed)
    rf = loadClassifier()
    class_predictions = rf.predict(dataML_matrix_scaled)
    dataML['predictedClass'] = ''
    for i in np.arange(len(class_predictions)):
        if verbose:
            print("%s is predicted to be a %s." % (names[i], class_predictions[i]))
        dataML.loc[dataML['TransientName'] == names[i], 'predictedClass'] = class_predictions[i]
    return dataML

def loadClassifier(verbose=True):
    modelName = "BinarySNClassifier.sav"
    stream = pkg_resources.resource_stream(__name__, modelName)
    if verbose:
        print("Loading model %s."%modelName)
    model = pickle.load(stream)
    return model

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

    dataML = condense_labels(dataML, nclass=nclass)

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
       'dist/DLR', 'dist', 'gSNR', 'rSNR', 'iSNR', 'zSNR', 'ySNR', 'TransientClass', 'TransientName']]

    dataML.dropna(axis=0, inplace=True)
    dataML.reset_index(inplace=True, drop=True)

    names = dataML['TransientName']
    dataML = dataML.drop(['TransientName'], axis=1)
    labels_df = dataML['TransientClass']# Remove the labels from the features
    labels = np.array(labels_df)
    classes = np.unique(labels)

    feature_list = list(dataML.columns) # Convert to numpy array

    dataML_noLabels = dataML.drop('TransientClass', axis=1)

    print('Distribution before imbalancing: {}'.format(Counter(labels)))

    labels_df.reset_index(inplace=True, drop=True)
    return feature_list, dataML_noLabels, labels_df, names

#feature_list, dataML_preprocessed2, labels_df2, names = preprocess_dataframe(dataML, nclass=2, PCA=False)
#dataML_matrix_scaled = preprocessing.scale(dataML_preprocessed2)
#labels = labels_df2.values

#acc, rf, all_confMatrices, accTot, wrong = plot_ROC_wCV(ax, dataML_matrix_scaled, labels.ravel(), names.values, save=0, balance=True)
