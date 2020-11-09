# taken from https://github.com/teddyroland/python-biplot/blob/master/biplot.py
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
from collections import Counter
from matplotlib import cm
import matplotlib.cm as cm
import joypy
from sklearn import preprocessing
import matplotlib
import sys
import joypy

def heatmap(corr, size):
    x = corr['x']
    y = corr['y']

    fig, ax = plt.subplots(figsize=(24,20))

    # Mapping from column names to integer coordinates
    x_labels = [v for v in x.unique()]
    y_labels = [v for v in y.unique()]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)}

    size_scale = 450
    s = ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        c=corr['value'],
        cmap='coolwarm',
        marker='s', # Use square as scatterplot marker
        vmin=-1,
        vmax=1
    )
    cbar = fig.colorbar(s, orientation='vertical')
    cbar.ax.tick_params(size=0)
    #cbar.set_label('Correlation', rotation=270)
    cbarax = cbar.ax
    cbarax.text(3,-0.12,'Correlation',rotation=-90, fontsize=30)
    # Show column labels on the axes
    #print([x_to_num[v]+0.5 for v in x_labels])
    plt.xlim((-0.5, 24.5))
    plt.ylim((-0.5, 24.5))
    ax.set_xticks([x_to_num[v]+0.5 for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='left')
    ax.invert_yaxis()
    ax.set_yticks([y_to_num[v]+0.5 for v in y_labels])
    ##plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, ha="left" )
    ax.set_yticklabels(y_labels)
    dx = -0.3; dy = 0;
    offset_x = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    dx = 0; dy = +0.2;
    offset_y = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset_x)
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset_y)

def preprocess_df(dataML, save=0):
    dataML['rAp-Kron'] = dataML['rApMag_rKronMag']
    del dataML['rApMag_rKronMag']
    #del dataML['7DCD']
    dataML = dataML.drop(['objAltName1','objAltName2','objAltName3'], axis=1)
    dataML = dataML.drop(['objID'], axis=1)
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
    #dataML = dataML.drop(['host_logmass', 'host_logmass_min', 'host_logmass_max'],axis=1)
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
    #dataML = dataML.drop(['level_0'], axis=1)
    dataML = dataML.drop(['gpsfTheta', 'ipsfTheta', 'zpsfTheta', 'ypsfTheta'], axis=1)
    dataML = dataML.drop(['gsky', 'isky', 'zsky', 'ysky'], axis=1)
    dataML = dataML.drop(['gskyErr', 'iskyErr', 'zskyErr', 'yskyErr'], axis=1)
    dataML = dataML.drop(['gpsfCore', 'ipsfCore', 'zpsfCore', 'ypsfCore'], axis=1)
    dataML = dataML.drop(['rpsfTheta', 'rsky', 'rskyErr', 'rpsfCore'], axis=1)
    dataML = dataML.drop(['gpsfLikelihood', 'rpsfLikelihood', 'ipsfLikelihood', 'zpsfLikelihood','ypsfLikelihood'], axis=1)
    dataML = dataML.drop(['rpsfQf'], axis=1)
    dataML = dataML.drop(['rpsfQfPerfect'], axis=1)
    dataML = dataML.drop(['rApFillFac'], axis=1)
    #dataML.drop(['objID'], inplace=True, axis=1)
    dataML = dataML.drop(['NED_redshift', 'NED_type', 'NED_mag', 'NED_name', 'NED_vel', 'TransientDEC', 'TransientDiscoveryDate', 'TransientDiscoveryMag', 'TransientDiscoveryYear', 'TransientRA'], axis=1)
    dataML = dataML.drop(['TransientRedshift',
       'TransientRedshift', 'Transient AltName',
       'host_logmass', 'host_logmass_min', 'host_logmass_max',
       'Hubble Residual'], axis=1)

    dataML = dataML.dropna()

    # Labels are the values we want to predict
    dataML.loc[dataML['TransientClass'] == 'SN Ib\n SN Ib', 'TransientClass'] = 'SN Ib'
    dataML.loc[dataML['TransientClass'] == 'SN Ia\n SN Ia', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SN II-pec', 'TransientClass'] = 'SN II Pec'
    dataML.loc[dataML['TransientClass'] == 'SN Ic-BL', 'TransientClass'] = 'SN Ic'
    dataML.loc[dataML['TransientClass'] == 'SN Ia-91T-like', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SN Ia-pec', 'TransientClass'] = 'SN Ia Pec'
    dataML.loc[dataML['TransientClass'] == 'SN Ib-pec', 'TransientClass'] = 'SN Ib'
    dataML.loc[dataML['TransientClass'] == 'SN Ic', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'SN Iax[02cx-like]', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SN Ib-Ca-rich', 'TransientClass'] = 'SN Ib'
    dataML.loc[dataML['TransientClass'] == 'SN Ia-91bg-like', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SN Ia-CSM', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SN Ic-pec', 'TransientClass'] = 'SN Ic'
    dataML.loc[dataML['TransientClass'] == 'SN IIn', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'SN Ibn', 'TransientClass'] = 'SN Ib'
    dataML.loc[dataML['TransientClass'] == 'SN Ib', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'SN Ia Pec', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SN Ia-91bg', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SN Ic', 'TransientClass'] = 'SN Ib/c'
    #dataML.loc[dataML['TransientClass'] == 'SN Ib/c', 'TransientClass'] = 'SN Ib'
    dataML.loc[dataML['TransientClass'] == 'SN IIP', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'SN IIL', 'TransientClass'] = 'SN II'
    #dataML.loc[dataML['TransientClass'] == 'SLSN-I', 'TransientClass'] = 'SLSN'
    #dataML.loc[dataML['TransientClass'] == 'SN IIb', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'SN I', 'TransientClass'] = 'SN I?'
    #dataML.loc[dataML['TransientClass'] == 'SN II', 'TransientClass'] = 'Core Collapse'
    dataML.loc[dataML['TransientClass'] == 'SN Ib', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'SN Ic', 'TransientClass'] = 'SN Ib/c'
    #dataML.loc[dataML['TransientClass'] == 'SN Ib/c', 'TransientClass'] = 'Core Collapse'
    dataML.loc[dataML['TransientClass'] == 'SLSN-I', 'TransientClass'] = 'SLSN'
    dataML.loc[dataML['TransientClass'] == 'SLSN-II', 'TransientClass'] = 'SLSN'
    dataML.loc[dataML['TransientClass'] == 'Ia Pec', 'TransientClass'] = 'SN Ia Pec'
    dataML.loc[dataML['TransientClass'] == 'II Pec', 'TransientClass'] = 'SN II Pec'
    dataML.loc[dataML['TransientClass'] == 'Ia*', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'Ia-02cx', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'Ia-91T', 'TransientClass'] = 'SN Ia-91T-like'
    dataML.loc[dataML['TransientClass'] == 'Ia-91bg', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'Ia-99aa', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'SLSN-II', 'TransientClass'] = 'SLSN'
    dataML.loc[dataML['TransientClass'] == 'CC', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'II', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'SLSN-I-R', 'TransientClass'] = 'SLSN'
    dataML.loc[dataML['TransientClass'] == 'SLSN-R', 'TransientClass'] = 'SLSN'
    dataML.loc[dataML['TransientClass'] == 'II/IIb', 'TransientClass'] = 'SN II'
    #dataML.loc[dataML['TransientClass'] == 'II L', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'II P', 'TransientClass'] = 'SN IIP'
    dataML.loc[dataML['TransientClass'] == 'Ib', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'Ic', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'II-p', 'TransientClass'] = 'SN II P'
    #dataML.loc[dataML['TransientClass'] == 'II/LBV', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'IIb', 'TransientClass'] = 'SN IIb'
    #dataML.loc[dataML['TransientClass'] == 'Ic Pec', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'SN Ia Pec', 'TransientClass'] = 'SN Ia'
    #dataML.loc[dataML['TransientClass'] == 'Ib/Ic (Ca rich?)?', 'TransientClass'] = 'SN Ib/c'
    #dataML.loc[dataML['TransientClass'] == 'Ia CSM', 'TransientClass'] = 'SN Ia'
    #dataML.loc[dataML['TransientClass'] == 'Ic BL', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'Ia', 'TransientClass'] = 'SN Ia'
    dataML.loc[dataML['TransientClass'] == 'Ib/c', 'TransientClass'] = 'SN Ib/c'
    dataML.loc[dataML['TransientClass'] == 'IIn', 'TransientClass'] = 'SN IIn'
    #dataML.loc[dataML['TransientClass'] == 'Ib Pec', 'TransientClass'] = 'SN Ib/c'
    #dataML.loc[dataML['TransientClass'] == 'Ibn', 'TransientClass'] = 'SN Ib/c'
    #dataML.loc[dataML['TransientClass'] == 'IIn Pec', 'TransientClass'] = 'SN IIn'
    dataML.loc[dataML['TransientClass'] == 'Ia/Ic', 'TransientClass'] = 'SN Ia/c'
    dataML.loc[dataML['TransientClass'] == 'SN II P', 'TransientClass'] = 'SN IIP'
    dataML.loc[dataML['TransientClass'] == 'SN II Pec', 'TransientClass'] = 'SN II'
    dataML.loc[dataML['TransientClass'] == 'Ia/c', 'TransientClass'] = 'SN Ia/c'
    dataML.loc[dataML['TransientClass'] == 'I', 'TransientClass'] = 'SN I'
    #dataML.loc[dataML['TransientClass'] == 'SN Ia-91T-like', 'TransientClass'] = 'SN Ia Pec'
    #dataML.loc[dataML['TransientClass'] == 'SN Iax[02cx-like]', 'TransientClass'] = 'SN Ia Pec'

    #dataML.loc[dataML['TransientClass'] == 'SLSN-I?', 'TransientClass'] = 'SLSN'

    #put that back in there, along with the SLSN-I-R
    #dataML.loc[dataML['TransientClass'] == 'SLSN-IIn', 'TransientClass'] = 'SLSN'

    # only for the 2 component class
    #dataML.loc[dataML['TransientClass'] == 'SN Ib/c', 'TransientClass'] = 'Core Collapse'
    #dataML.loc[dataML['TransientClass'] == 'SN IIP', 'TransientClass'] = 'Core Collapse'
    #dataML.loc[dataML['TransientClass'] == 'SN IIb', 'TransientClass'] = 'Core Collapse'
    #dataML.loc[dataML['TransientClass'] == 'SN IIn', 'TransientClass'] = 'Core Collapse'
    #dataML.loc[dataML['TransientClass'] == 'SN II', 'TransientClass'] = 'Core Collapse'

    #'SN Ia': 6279, 'SN II': 2061, 'SN Ib/c': 528, 'SN IIP': 307, 'SN IIn': 265, 'SN IIb': 94, 'SLSN': 38

    # 'SN II', 'SN IIb', 'SN IIn',
    # Delete the rows where we have no set detection
    dataML = dataML[dataML['TransientClass'] != 'SN']
    #dataML = dataML[dataML['TransientClass'] != 'SLSN']
    dataML = dataML[dataML['TransientClass'] != 'Ia-09dc']
    dataML = dataML[dataML['TransientClass'] != 'SN I?']
    dataML = dataML[dataML['TransientClass'] != 'SNIa?']
    dataML = dataML[dataML['TransientClass'] != 'SLSN?']
    dataML = dataML[dataML['TransientClass'] != 'Ic Pec']
    dataML = dataML[dataML['TransientClass'] != 'PISN?']
    dataML = dataML[dataML['TransientClass'] != 'I?']
    dataML = dataML[dataML['TransientClass'] != 'Ib/Ic (Ca rich?)?']
    dataML = dataML[dataML['TransientClass'] != 'II Pec?']
    dataML = dataML[dataML['TransientClass'] != 'IIn?']
    dataML = dataML[dataML['TransientClass'] != 'IIb?']
    dataML = dataML[dataML['TransientClass'] != 'SLSN?']
    dataML = dataML[dataML['TransientClass'] != 'SN Ic-BL']

    #dataML = dataML[dataML['TransientClass'] != 'SN IIb']
    #dataML = dataML[dataML['TransientClass'] != 'SN Ia Pec']
    #dataML = dataML[dataML['TransientClass'] != 'SN IIn']

    #'SLSN', 'SN II', 'SN II Pec', 'SN IIP', 'SN IIb', 'SN IIn',
    #       'SN Ia', 'SN Ia Pec', 'SN Ia-91T-like', 'SN Ib/c'

    dataML = dataML[dataML['TransientClass'] != 'Ic?']
    dataML = dataML[dataML['TransientClass'] != 'II?']
    dataML = dataML[dataML['TransientClass'] != 'Ib/IIb']
    dataML = dataML[dataML['TransientClass'] != 'IIb/Ib']
    dataML = dataML[dataML['TransientClass'] != 'II/Ib/c']
    dataML = dataML[dataML['TransientClass'] != 'Ib/c?']
    dataML = dataML[dataML['TransientClass'] != 'SLSN-II?']
    dataML = dataML[dataML['TransientClass'] != 'Ia?']
    # only for the 4-component class
    dataML = dataML[dataML['TransientClass'] != 'SN I']

    dataML = dataML[dataML['TransientClass'] != 'LBV to IIn']
    dataML = dataML[dataML['TransientClass'] != 'LBV']
    dataML = dataML[dataML['TransientClass'] != 'SN Ia/c']

    dataML = dataML[dataML['TransientClass'] != 'Ca-rich']
    dataML = dataML[dataML['TransientClass'] != 'Pec']
    dataML = dataML[dataML['TransientClass'] != 'CN']
    dataML = dataML[dataML['TransientClass'] != 'II L?']
    dataML = dataML[dataML['TransientClass'] != 'Ib-Ca']
    dataML = dataML[dataML['TransientClass'] != 'Pec']
    dataML = dataML[dataML['TransientClass'] != 'nIa']
    dataML = dataML[dataML['TransientClass'] != 'SLSN-I?']
    dataML = dataML[dataML['TransientClass'] != 'SLSN-IIn']
    dataML = dataML[dataML['TransientClass'] != 'SN Iax[02cx-like]']
    dataML = dataML[dataML['TransientClass'] != 'SN Ia-91T-like']
    dataML = dataML[dataML['TransientClass'] != 'Ia-02cx']
    dataML = dataML[dataML['TransientClass'] != 'Ia-91T']
    dataML = dataML[dataML['TransientClass'] != 'Ia-91bg']
    dataML = dataML[dataML['TransientClass'] != 'IIn Pec']

    dataML_orig = dataML

    dataML = dataML.drop(['TransientName'],axis=1)
    dataML = dataML.drop(['TransientClass'],axis=1)

    dataML.dropna(inplace=True)

    if save:
        dataML.to_csv("pre_PCA_features.csv",index=False)
    return dataML_orig, dataML

def plot_heatmap(dataML, save=0, corr_type='pearson'):
    dataML["gSNR"] = 1/dataML["gApMagErr"]
    dataML["rSNR"] = 1/dataML["rApMagErr"]
    dataML["iSNR"] = 1/dataML["iApMagErr"]
    dataML["zSNR"] = 1/dataML["zApMagErr"]
    dataML["ySNR"] = 1/dataML["yApMagErr"]

    dataML['4DCD'] = dataML['7DCD']
    dataML[r'$\theta$'] = dataML['dist']
    dataML[r'$\theta/d_{DLR}$'] = dataML['dist/DLR']
    dataML[r'Ap - Kron'] = dataML['gApMag_gKronMag']
    dataML[r'KronRad'] = dataML['gKronRad']
    dataML[r'PSFMag'] = dataML['gPSFMag']
    dataML[r'PSFMagErr'] = dataML['gPSFMagErr']
    dataML[r'ApMag'] = dataML['gApMag']
    dataML[r'ApMagErr'] = dataML['gApMagErr']
    dataML[r'KronMag'] = dataML['gKronMag']
    dataML[r'KronMagErr'] = dataML['gKronMagErr']
    dataML[r'psfMajorFWHM'] = dataML['gpsfMajorFWHM']
    dataML[r'psfMinorFWHM'] = dataML['gpsfMinorFWHM']
    dataML[r'momentXX'] = dataML['gmomentXX']
    dataML[r'momentXY'] = dataML['gmomentXY']
    dataML[r'momentYY'] = dataML['gmomentYY']

    dataML[r'momentR1'] = dataML['gmomentR1']
    dataML[r'momentRH'] = dataML['gmomentRH']
    dataML[r'ApRadius'] = dataML['gApRadius']
    dataML[r'ExtNSigma'] = dataML['gExtNSigma']
    dataML[r'PSFFlux'] = dataML['gPSFFlux']
    dataML[r'PSFFluxErr'] = dataML['gPSFFluxErr']
    dataML[r'ApFlux'] = dataML['gApFlux']
    dataML[r'ApFluxErr'] = dataML['gApFluxErr']
    dataML[r'KronFlux'] = dataML['gKronFlux']
    dataML[r'KronFluxErr'] = dataML['gKronFluxErr']

    dataML_corr = dataML[[r'PSFMag',r'PSFMagErr','ApMag','ApMagErr',r'KronMag','KronMagErr','psfMajorFWHM','psfMinorFWHM',
    'momentXX','momentXY','momentYY','momentR1','momentRH','PSFFlux','PSFFluxErr','ApFlux','ApFluxErr','ApRadius',
    'KronFlux','KronFluxErr',r'KronRad','ExtNSigma',r'Ap - Kron',
    'g-r','r-i','i-z','z-y','4DCD',r'$\theta$',r'$\theta/d_{DLR}$',r'g-rErr']]

    dataML.drop(['rAp-Kron', 'ApRadius', 'Ap - Kron','KronRad','PSFMag','PSFMagErr','ApMag','ApMagErr','KronMag','KronMagErr','psfMajorFWHM','psfMinorFWHM','momentXX','momentXY','momentYY','momentR1','momentRH'],axis=1, inplace=True)
    dataML.drop(['ExtNSigma','PSFFlux','PSFFluxErr','ApFlux','ApFluxErr','KronFlux','KronFluxErr', '4DCD', r'$\theta$', r'$\theta/d_{DLR}$'], axis=1, inplace=True)

    cols = dataML.columns.values
    cols = ['g-r', 'gApFlux', 'gApFluxErr', 'gApMag', 'gApMagErr', #g
       'gApMag_gKronMag', 'gApRadius', 'gExtNSigma', 'gKronFlux',
       'gKronFluxErr', 'gKronMag', 'gKronMagErr', 'gKronRad', 'gPSFFlux',
       'gPSFFluxErr', 'gPSFMag', 'gPSFMagErr', 'gmomentR1', 'gmomentRH',
       'gmomentXX', 'gmomentXY', 'gmomentYY', 'gpsfMajorFWHM',
       'gpsfMinorFWHM','g-rErr',
       'r-i', 'rApFlux', 'rApFluxErr', #r
       'rApMag', 'rApMagErr', 'rApRadius', 'rExtNSigma', 'rKronFlux',
       'rKronFluxErr', 'rKronMag', 'rKronMagErr', 'rKronRad', 'rPSFFlux',
       'rPSFFluxErr', 'rPSFMag', 'rPSFMagErr', 'rmomentR1', 'rmomentRH',
       'rmomentXX', 'rmomentXY', 'rmomentYY', 'rpsfMajorFWHM',
       'rpsfMinorFWHM','r-iErr',
       'i-z', 'iApFlux', 'iApFluxErr', 'iApMag', #i
       'iApMagErr', 'iApMag_iKronMag', 'iApRadius', 'iExtNSigma',
       'iKronFlux', 'iKronFluxErr', 'iKronMag', 'iKronMagErr', 'iKronRad',
       'iPSFFlux', 'iPSFFluxErr', 'iPSFMag', 'iPSFMagErr', 'imomentR1',
       'imomentRH', 'imomentXX', 'imomentXY', 'imomentYY',
       'ipsfMajorFWHM', 'ipsfMinorFWHM','i-zErr',
       'z-y','zApFlux', 'zApFluxErr', 'zApMag', 'zApMagErr', #z
       'zApMag_zKronMag', 'zApRadius', 'zExtNSigma', 'zKronFlux',
       'zKronFluxErr', 'zKronMag', 'zKronMagErr', 'zKronRad', 'zPSFFlux',
       'zPSFFluxErr', 'zPSFMag', 'zPSFMagErr', 'zmomentR1', 'zmomentRH',
       'zmomentXX', 'zmomentXY', 'zmomentYY', 'zpsfMajorFWHM',
       'zpsfMinorFWHM','z-yErr',
       'yApFlux', 'yApFluxErr', 'yApMag', 'yApMagErr', #y
       'yApMag_yKronMag', 'yApRadius', 'yExtNSigma', 'yKronFlux',
       'yKronFluxErr', 'yKronMag', 'yKronMagErr', 'yKronRad', 'yPSFFlux',
       'yPSFFluxErr', 'yPSFMag', 'yPSFMagErr', 'ymomentR1', 'ymomentRH',
       'ymomentXX', 'ymomentXY', 'ymomentYY', 'ypsfMajorFWHM',
       'ypsfMinorFWHM', 'dist/DLR', 'dist']

    dataML_shifted = dataML[cols]


    sns.set_context("poster")
    plt.figure(figsize=(250, 200))
    sns.heatmap(dataML_shifted.corr(method=corr_type), annot=False, cmap='coolwarm', vmin=-1, vmax=1, linecolor='white', linewidths=0.3,annot_kws={"fontsize": "5"},cbar_kws={'label': 'Correlation'})
    if save:
        plt.savefig("heatmap_fullTable_%s.png"%corr_type, bbox_inches='tight')
        dataML.corr(method=corr_type).to_csv("GHOST_fullTable_correlations_%s.tar.gz"%corr_type, index=False)

    #dataML_corr_sorted = dataML_corr[['g-r', 'momentR1', 'KronRad', 'Ap - Kron', 'momentXX', '4DCD',
    #    'ExtNSigma', 'PSFFlux', 'r-i', 'momentYY','momentRH',
    #    'PSFMag', 'PSFMagErr', 'ApMag',
    #    'ApMagErr', 'KronMag',
    #    'KronMagErr', 'psfMajorFWHM', 'psfMinorFWHM',
    #    'momentXY',
    #    'PSFFluxErr', 'ApFlux', 'ApFluxErr', 'ApRadius', 'KronFlux',
    #    'KronFluxErr',
    #    'i-z', 'z-y', '$\\theta$', '$\\theta/d_{DLR}$']]
    dataML_corr_sorted = dataML_corr[['g-r', 'ApFlux', 'ApFluxErr', 'ApMag', 'ApMagErr', #g
       'Ap - Kron', 'ApRadius', 'ExtNSigma', 'KronFlux',
       'KronFluxErr', 'KronMag', 'KronMagErr', 'KronRad', 'PSFFlux',
       'PSFFluxErr', 'PSFMag', 'PSFMagErr', 'momentR1', 'momentRH',
       'momentXX', 'momentXY', 'momentYY', 'psfMajorFWHM',
       'psfMinorFWHM','g-rErr']]

    data = dataML_corr_sorted

    columns = dataML_corr_sorted.columns
    corr = data[columns].corr(method=corr_type)
    corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr,
        size=corr['value'].abs()
    )
    if save:
        plt.savefig("heatmap_withScaledSquares_%s.png" %corr_type, dpi=300, bbox_inches='tight')

    #matrix = np.triu(dataML_corr.dropna().corr(method=corr_type))
    #plt.figure(figsize=(20, 14))
    #sns.heatmap(dataML_corr.corr(method=corr_type), annot=False, mask=matrix, cmap='coolwarm', vmin=-1, vmax=1, linecolor='white', linewidths=0.3,annot_kws={"fontsize": "30"},cbar_kws={'label': 'Correlation'})
    #if save:
    #    plt.savefig("heatmap_triangle_%s.pdf"%corr_type,dpi=300, bbox_inches='tight')

    dataML = dataML.drop(['gSNR', 'rSNR', 'iSNR', 'zSNR', 'ySNR'], axis=1)
    dataML_sub = dataML[['r-i', 'iApMag_iKronMag', 'zApMag_zKronMag', 'g-r']]
    bestFeatures = np.array(['r-i', 'iApMag_iKronMag', 'zApMag_zKronMag', 'g-r', 'gApMag_gKronMag', 'yApMag_yKronMag', 'gExtNSigma', '7DCD', 'rPSFMag_rKronMag', 'yExtNSigma', 'zExtNSigma', 'iPSFMag_zPSFMag', 'gmomentRH', 'i-z'])
    #del dataML['Unnamed: 0']
    dataML['4DCD'] = dataML['7DCD']
    del dataML['7DCD']
    #get rid of color for now
    del dataML['4DCD']
    dataML[r'$\theta$'] = dataML['dist']
    del dataML['dist']
    dataML[r'$\theta/d_{DLR}$'] = dataML['dist/DLR']
    del dataML['dist/DLR']

def tsne_ghost(dataML):
    dataML_orig, dataML = preprocess_df(dataML)
    plot_heatmap(dataML, save=0, corr_type='spearman')

    #scaling with standardscaler
    dataML_scaled = preprocessing.scale(dataML)
    ## perform PCA
    n = len(dataML.columns)
    pca = PCA(n_components = 2)
    df_plot = pca.fit(dataML_scaled)

    ## project data into PC space
    xvector = pca.components_[0] # see 'prcomp(my_data)$rotation' in R
    yvector = pca.components_[1]

    xs = pca.transform(dataML_scaled)[:,0] # see 'prcomp(my_data)$x' in R
    ys = pca.transform(dataML_scaled)[:,1]

    plt.figure(figsize=(10,7))
    sns.kdeplot(df_plot.loc[df_plot['TransientClass']=='SN Ia','xs'], shade=True, shade_lowest=False, alpha=0.6, label='SN Ia',color='tab:blue')
    plt.axvline(np.median(df_plot.loc[df_plot['TransientClass']=='SN Ia','xs']), linestyle='--',color='tab:blue')
    sns.kdeplot(df_plot.loc[df_plot['TransientClass']=='SN Ib/c','xs'], shade=True, shade_lowest=False, alpha=0.6, label='SN Ib/c',color='tab:green')
    plt.axvline(np.median(df_plot.loc[df_plot['TransientClass']=='SN Ib/c','xs']), linestyle='--',color='tab:green')
    sns.kdeplot(df_plot.loc[df_plot['TransientClass']=='SLSN','xs'], shade=True, shade_lowest=False, alpha=0.6, label='SLSN', color='tab:orange')
    plt.axvline(np.median(df_plot.loc[df_plot['TransientClass']=='SLSN','xs']), linestyle='--',color='tab:orange')
    plt.legend(fontsize=16)
    plt.xlabel("PC1 (71.5%)",fontsize=16)
    plt.xlim((-20,20))
    plt.savefig("PCA_axis1Only_withMedians_scaler.png", dpi=300)

    Counter(df_plot['TransientClass'])
    dropClass = np.array(['II L', 'II/LBV', 'Ia CSM', 'Ib Pec', 'Ibn', 'Ic BL','SN IIL','SN Ia-91bg-like', 'SN Ia-CSM', 'SN Ib-Ca-rich', 'SN Ib-pec', 'SN Ibn', 'SN Ic-pec'])
    df_plot = df_plot[~df_plot['TransientClass'].isin(dropClass)]

    # red purple blue
    Counter(df_plot['TransientClass'])
    cols = ['#ff6666', '#861388', '#0d3b66']
    #classes = ['SN Ia', 'SN II','SN Ib/c']
    #classes = ['SN Ib/c','SN II','SN Ia']
    classes = ['SN Ib/c','SLSN', 'SN Ia']

    length = np.sqrt(xvector**2 + yvector**2)

    def checkFeature(feature):
        if feature.endswith("Err"):
            return False
        if feature.startswith("4"):
            return False
        if (not feature.startswith('g') and  not feature.startswith('i') and not feature.startswith('z') and not feature.startswith('y')):
            return True
        return False

    sns.set_context("poster")
    #pl = sns.color_palette(['#f77189', '#3ba3ec''#50b131'])
    pl = [sns.light_palette("firebrick", as_cmap=True), sns.light_palette("purple", as_cmap=True), sns.light_palette("orange", as_cmap=True)]
    plt.figure(figsize=(10,10))
    plt.xlim((-75, 75))
    plt.ylim((-30,40))
    for i in np.arange(len(cols)):
        if i == 2:
            shadeval = False
            ls = '--'
        else:
            shadeval=True
            ls = '-'
        alpha=1.
        sns.kdeplot(df_plot.loc[df_plot['TransientClass'] == classes[i], 'xs'], df_plot.loc[df_plot['TransientClass'] == classes[i],'ys'], alpha=alpha, linewidths=5, n_levels=5, linestyles=ls,shade=shadeval, shade_lowest=False,label=classes[i])
    facs = ['rKronMag', 'rKronMag', 'rmomentR1', 'rExtNSigma']
    j = 0

    for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
        length = np.sqrt(xvector[i]**2 + yvector[i]**2)
        if length > 0.1:
            if checkFeature(dataML.columns.values[i]):
                if dataML.columns.values[i].endswith("Mag"):
                    xtext = xvector[i]*max(xs) - 9
                    ytext = yvector[i]*max(ys) - 0.25
                elif dataML.columns.values[i].endswith("Flux"):
                    xtext = xvector[i]*max(xs) + 0.5
                    ytext = yvector[i]*max(ys)
                elif dataML.columns.values[i] == 'rmomentR1':
                    xtext = xvector[i]*max(xs)+1.5
                    ytext = yvector[i]*max(ys) - 1.8
                elif dataML.columns.values[i] == 'rExtNSigma':
                    xtext = xvector[i]*max(xs)
                    ytext = yvector[i]*max(ys) - 2
                elif dataML.columns.values[i] == 'rmomentRH':
                    xtext = xvector[i]*max(xs) + 0.5
                    ytext = yvector[i]*max(ys) - 0.75
                elif dataML.columns.values[i] == 'rKronRad':
                    xtext = xvector[i]*max(xs)+2.5
                    ytext = yvector[i]*max(ys)
                elif dataML.columns.values[i] == 'rAp-Kron':
                    #dataML.columns.values[i] = 'rAp-Kron'
                    xtext = xvector[i]*max(xs) + 2
                    ytext = yvector[i]*max(ys) - 0.5
                elif dataML.columns.values[i] == 'rmomentXX':
                    xtext = xvector[i]*max(xs) + 2
                    ytext = yvector[i]*max(ys) - 1.5
                elif dataML.columns.values[i] == 'rmomentYY':
                    xtext = xvector[i]*max(xs) + 2
                    ytext = yvector[i]*max(ys) - 0.75
                else:
                    xtext = xvector[i]*max(xs)
                    ytext = yvector[i]*max(ys)

                if dataML.columns.values[i] == 'rKronFlux':
                    xtext = xvector[i]*max(xs) + 0.5
                    ytext = yvector[i]*max(ys) -1
                elif dataML.columns.values[i] == 'rKronMag':
                    xtext = xvector[i]*max(xs) - 13
                    ytext = yvector[i]*max(ys) - 0.25
                plt.text(xtext, ytext,
                    list(dataML.columns.values)[i], fontsize=26, color='#2e2e3a')
        #    j += 1
                plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
                color='#2e2e3a', width=0.05, head_width=0.25, zorder=10, alpha=1, lw=3)
    #plt.text(-5, -22,'SN Ia', fontsize=20, color='#ff6666')
    #plt.text(-60, 10,'SN II', fontsize=20, color='#861388')
    #plt.text(55, 20,'SN Ib/c', fontsize=20, color='#0d3b66')
    #re-calculate DLR!
    plt.xlim((-22, 20))
    plt.ylim((-12, 12))
    plt.xlabel("PC 1 (44.5%)",fontsize=20)
    #0.44575393, 0.10369499
    plt.ylabel("PC 2 (10.4%)",fontsize=20)
    plt.legend(fontsize=20, loc='lower left')
    plt.savefig("PCA_Ia_Ibc_II_scaler.png", dpi=300,bbox_inches='tight')

    fig = plt.figure(figsize = (8,8))
    plt.plot(np.arange(10), pca.explained_variance_ratio_,'o--',markersize=15)
    plt.xticks(np.arange(10))
    plt.xlabel("Principal Component",fontsize=18)
    plt.ylabel("Explained Variance (%)",fontsize=18)
    np.sum(pca.explained_variance_ratio_)

    #create data frame with xs, ys, dataML_orig['TransientClass']
    dataML_orig.reset_index(drop=True, inplace=True)

    df_plot_subset = df_plot[['ys', 'TransientClass']]
    df_plot_subset2 = df_plot[['xs', 'TransientClass']]

    sns.set_context("notebook", font_scale=1.5)
    fig, axes = joypy.joyplot(df_plot_subset2, by="TransientClass", ylim='own',fade=True, colormap=cm.Dark2, kind="kde")
    fig.set_figheight(10)
    fig.set_figwidth(10)
    axes[0].set_xlim((-25,25))
    fig.savefig("PCA2_joyplot2_scaler_dropOutliers.png", dpi=200)

    df_plot_lowZ = df_plot.loc[df_plot['TransientRedshift'] < 0.1]
    df_plot_subset_lowZ = df_plot_lowZ[['ys', 'TransientClass']]
    df_plot_subset_lowZ2 = df_plot_lowZ[['xs', 'TransientClass']]

    fig, axes = joypy.joyplot(df_plot_subset_lowZ2, by="TransientClass", ylim='own', fade=True, colormap=cm.Dark2, kind="kde")
    fig.set_figheight(10)
    fig.set_figwidth(10)
    axes[0].set_xlim(xmin=-20, xmax=20)
    fig.savefig("PCA2_joyplot2_lowZ_dropOutliers.pdf")

    fig = plt.figure(figsize = (8,8))
    plt.plot(np.arange(2), pca.explained_variance_ratio_,'o--',markersize=15)
    plt.xticks(np.arange(2))
    plt.xlabel("Principal Component",fontsize=18)
    plt.ylabel("Explained Variance (%)",fontsize=18)
    plt.savefig("Explained_variance.png", dpi=300, bbox_inches='tight')
