from astropy.io import ascii
from astropy.table import Table
import pkg_resources

import sys
import re
import numpy as np
import pylab
import json
import requests
from astropy.io import fits

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from concurrent.futures import ThreadPoolExecutor
import asyncio
import asyncio
import codecs
#import aiohttp

import time
import sfdmap
import os
import tarfile

def build_sfd_dir(fname='./sfddata-master.tar.gz'):
    url = 'http://ghost.ncsa.illinois.edu/static/sfddata-master.tar.gz'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(fname, 'wb') as f:
            f.write(response.raw.read())
    tar = tarfile.open(fname)
    tar.extractall()
    tar.close()
    os.remove(fname)
    print("Done creating dust directory.")
    return

def get_photoz_weights(fname='./MLP_lupton.hdf5'):
    url = 'http://ghost.ncsa.illinois.edu/static/MLP_lupton.hdf5'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(fname, 'wb') as f:
            f.write(response.raw.read())
    print("Done getting photo-z weights.")
    return

def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a cone search of the PS1 catalog

    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)
    """

    ra=list(ra)
    dec=list(dec)
    radius = list(radius)

    assert len(ra) == len(dec) == len(radius)
    #this is a dictionary... we want a list of dictionaries
    data_list=[kw.copy() for i in range(len(ra))]

    for i in range(len(data_list)):
        data_list[i]['ra'] = ra[i]
        data_list[i]['dec'] = dec[i]
        data_list[i]['radius'] = radius[i]

    urls = []
    datas = []
    for i in range(len(ra)):
        url, data = ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data_list[i])

        urls.append(url)
        datas.append(data)

    return urls, datas

def ps1objIDsearch(objID,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do an object lookup by objID

    Parameters
    ----------
    objID (big int): list of
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)
    """
    #this is a dictionary... we want a list of dictionaries
    objID=list(objID)

    data_list=[kw.copy() for i in range(len(objID))]
    assert len(data_list)==len(objID)

    for i in range(len(data_list)):
        data_list[i]['objID'] = objID[i]

    urls = []
    datas = []
    for i in range(len(objID)):
        url, data = ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data_list[i])

        urls.append(url)
        datas.append(data)

    return urls, datas


def ps1search(table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius)

    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2).  Note this is required!
    """

    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    checklegal(table,release)
    if format not in ("csv","votable","json"):
        raise ValueError("Bad value for format")
    url = "{baseurl}/{release}/{table}.{format}".format(**locals())
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

        # either get or post works
        #r = requests.post(url, data=data)
        return url, data
    return url, data

def fetch_information_serially(url,data,verbose=False,format='csv'):
    results = []
    for i in range(len(url)):
        r = requests.get(url[i], params=data[i])
        if verbose:
            print(r.url)
        r.raise_for_status()
        if format == "json":
            results.append(r.json())
        else:
            results.append(r.text)

    return results

def checklegal(table,release):
    """Checks if this combination of table and release is acceptable

    Raises a VelueError exception if there is problem
    """

    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection", "forced_mean")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))


def ps1metadata(table="mean",release="dr1",
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table

    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    baseurl: base URL for the request

    Returns an astropy table with columns name, type, description
    """

    checklegal(table,release)
    url = "{baseurl}/{release}/{table}/metadata".format(**locals())
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab

def post_url_parallel(results,YSE_ID):
    if type(results) != str:
        results = codecs.decode(results,'UTF-8')
    lines = results.split('\n')
    print(lines)
    if len(lines) > 2:
        values = [line.strip().split(',') for line in lines]
        DF = pd.DataFrame(values[1:-1],columns=values[0])
    else:
        print('No Matches')
        DF = pd.DataFrame()
    DF['YSE_id'] = np.ones(len(DF))*YSE_ID
    return DF

def post_url_serial(results,YSE_ID):
    if type(results) != str:
        results = codecs.decode(results,'UTF-8')
    lines = results.split('\r\n')
    #print(lines)
    if len(lines) > 2:
        values = [line.strip().split(',') for line in lines]
        DF = pd.DataFrame(values[1:-1],columns=values[0])
    else:
        DF = pd.DataFrame()
    DF['id'] = np.ones(len(DF))*YSE_ID
    return DF

def serial_objID_search(objIDs,table='forced_mean',release='dr2',columns=None,verbose=False,**constraints):
    constrains=constraints.copy()
    URLS, DATAS = ps1objIDsearch(objID=objIDs,table='forced_mean',release=release,columns=columns,verbose=verbose,**constraints)
    Return = fetch_information_serially(URLS,DATAS)
    DFs=[]
    for i in range(len(Return)):
        DFs.append(post_url_serial(Return[i],i))

    return DFs

def post_url_parallel(results,YSE_ID):
    results = codecs.decode(results,'UTF-8')
    lines = results.split('\n')
    if len(lines) > 2:
        values = [line.strip().split(',') for line in lines]
        DF = pd.DataFrame(values[1:-1],columns=values[0])
    else:
        DF = pd.DataFrame()
    DF['id'] = np.ones(len(DF))*YSE_ID
    return DF

def get_common_constraints_columns():
    '''
    Because I am so lazy
    '''
    constraints = {'nDetections.gt':1} #objects with n_detection=1 sometimes just an artifact.
    # strip blanks and weed out blank and commented-out values
    columns ="""objID, raMean, decMean, gFKronFlux, rFKronFlux, iFKronFlux, zFKronFlux, yFKronFlux,
    gFPSFFlux, rFPSFFlux, iFPSFFlux, zFPSFFlux, yFPSFFlux,
    gFApFlux, rFApFlux, iFApFlux, zFApFlux, yFApFlux,
    gFmeanflxR5, rFmeanflxR5, iFmeanflxR5, zFmeanflxR5, yFmeanflxR5,
    gFmeanflxR6, rFmeanflxR6, iFmeanflxR6, zFmeanflxR6, yFmeanflxR6,
    gFmeanflxR7, rFmeanflxR7, iFmeanflxR7, zFmeanflxR7, yFmeanflxR7""".split(',')
    columns = [x.strip() for x in columns]
    columns = [x for x in columns if x and not x.startswith('#')]

    return constraints, columns

def preprocess(DF,PATH='../DATA/sfddata-master/', ebv=True):
    if ebv:
        m = sfdmap.SFDMap(PATH)
        EBV = m.ebv(DF['raMean'].values.astype(np.float32),DF['decMean'].values.astype(np.float32))

        DF['ebv'] = EBV
    else:
        DF['ebv'] = 0.0

    def convert_flux_to_luptitude(f,b,f_0=3631):
        return -2.5/np.log(10) * (np.arcsinh((f/f_0)/(2*b)) + np.log(b))

    b_g = 1.7058474723241624e-09
    b_r = 4.65521985283191e-09
    b_i = 1.2132217745483221e-08
    b_z = 2.013446972858555e-08
    b_y = 5.0575501316874416e-08


    MEANS = np.array([18.70654578, 17.77948707, 17.34226094, 17.1227873 , 16.92087669,
           19.73947441, 18.89279411, 18.4077393 , 18.1311733 , 17.64741402,
           19.01595669, 18.16447837, 17.73199409, 17.50486095, 17.20389615,
           19.07834251, 18.16996592, 17.71492073, 17.44861273, 17.15508793,
           18.79100201, 17.89569908, 17.45774026, 17.20338482, 16.93640741,
           18.62759241, 17.7453392 , 17.31341498, 17.06194499, 16.79030564,
            0.02543223])

    STDS = np.array([1.7657395 , 1.24853534, 1.08151972, 1.03490545, 0.87252421,
           1.32486758, 0.9222839 , 0.73701807, 0.65002723, 0.41779001,
           1.51554956, 1.05734494, 0.89939638, 0.82754093, 0.63381611,
           1.48411417, 1.05425943, 0.89979008, 0.83934385, 0.64990996,
           1.54735158, 1.10985163, 0.96460099, 0.90685922, 0.74507053,
           1.57813401, 1.14290345, 1.00162105, 0.94634726, 0.80124359,
           0.01687839])

    data_columns = ['gFKronFlux', 'rFKronFlux', 'iFKronFlux', 'zFKronFlux', 'yFKronFlux',
    'gFPSFFlux', 'rFPSFFlux', 'iFPSFFlux', 'zFPSFFlux', 'yFPSFFlux',
    'gFApFlux', 'rFApFlux', 'iFApFlux', 'zFApFlux', 'yFApFlux',
    'gFmeanflxR5', 'rFmeanflxR5', 'iFmeanflxR5', 'zFmeanflxR5', 'yFmeanflxR5',
    'gFmeanflxR6', 'rFmeanflxR6', 'iFmeanflxR6', 'zFmeanflxR6', 'yFmeanflxR6',
    'gFmeanflxR7', 'rFmeanflxR7', 'iFmeanflxR7', 'zFmeanflxR7', 'yFmeanflxR7', 'ebv']

    X = DF[data_columns].values.astype(np.float32)
    X[:,0:30:5] = convert_flux_to_luptitude(X[:,0:30:5],b=b_g)
    X[:,1:30:5] = convert_flux_to_luptitude(X[:,1:30:5],b=b_r)
    X[:,2:30:5] = convert_flux_to_luptitude(X[:,2:30:5],b=b_i)
    X[:,3:30:5] = convert_flux_to_luptitude(X[:,3:30:5],b=b_z)
    X[:,4:30:5] = convert_flux_to_luptitude(X[:,4:30:5],b=b_y)

    X = (X-MEANS)/STDS
    X[X>20] = 20
    X[X<-20] = -20
    X[np.isnan(X)] = -20

    return X

def load_lupton_model(model_path):
    build_sfd_dir()
    get_photoz_weights(model_path)
    def model():
        INPUT = tf.keras.layers.Input(31)

        DENSE1 = tf.keras.layers.Dense(256,activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),kernel_regularizer=tf.keras.regularizers.l2(1e-5))(INPUT)
        DROP1 = tf.keras.layers.Dropout(0.05)(DENSE1)

        DENSE2 = tf.keras.layers.Dense(1024,activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),kernel_regularizer=tf.keras.regularizers.l2(1e-5))(DROP1)
        DROP2 = tf.keras.layers.Dropout(0.05)(DENSE2)

        DENSE3 = tf.keras.layers.Dense(1024,activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),kernel_regularizer=tf.keras.regularizers.l2(1e-5))(DROP2)
        DROP3 = tf.keras.layers.Dropout(0.05)(DENSE3)

        DENSE4 = tf.keras.layers.Dense(1024,activation=tf.keras.layers.LeakyReLU(),kernel_initializer=tf.keras.initializers.he_normal(),kernel_regularizer=tf.keras.regularizers.l2(1e-5))(DROP3)

        OUTPUT = tf.keras.layers.Dense(360,activation=tf.keras.activations.softmax)(DENSE4)

        model = tf.keras.Model(INPUT,OUTPUT)

        return model
    mymodel = model()
    #stream = pkg_resources.resource_stream(__name__, model_path)
    mymodel.load_weights(model_path)

    NB_BINS = 360
    ZMIN = 0.0
    ZMAX = 1.0
    BIN_SIZE = (ZMAX - ZMIN) / NB_BINS
    range_z = np.linspace(ZMIN, ZMAX, NB_BINS + 1)[:NB_BINS]

    return mymodel, range_z

def evaluate(X,mymodel,range_z):
    posteriors = mymodel(X,training=False).numpy()
    point_estimates = np.sum(posteriors*range_z,axis=1)
    for i in range(len(posteriors)):
        posteriors[i,:] /= np.sum(posteriors[i,:])
    errors=np.ones(len(posteriors))
    for i in range(len(posteriors)):
        errors[i] = (np.std(np.random.choice(a=range_z,size=1000,p=posteriors[i,:],replace=True)))

    return posteriors, point_estimates, errors


#PhotoZ beta: not tested for missing objids.
#photo-z uses a artificial neural network to estimate P(Z) in range Z = (0 - 1)
#range_z is the value of z
#posterior is an estimate PDF of the probability of z
#point estimate uses the mean to find a single value estimate
#error is an array that uses sampling from the posterior to estimate a STD

#relies upon the sfdmap package, (which is compatible with both unix and windows)
#https://github.com/kbarbary/sfdmap

#'id' column in DF is the 0th ordered index of hosts. missing rows are therefore signalled
#    by skipped numbers in index
def calc_photoz(hosts):
    objIDs = hosts['objID'].values.tolist()
    constraints, columns = get_common_constraints_columns()
    DFs = serial_objID_search(objIDs,columns=columns,**constraints)
    DF = pd.concat(DFs)

    #The function load_lupton_model downloads the necessary dust models and
    #weights from the ghost server.
    dust_PATH = './sfddata-master'
    model_PATH = './MLP_lupton.hdf5'

    mymodel, range_z = load_lupton_model(model_PATH)
    X = preprocess(DF,dust_PATH)
    posteriors, point_estimates, errors = evaluate(X,mymodel,range_z)
    successIDs = DF['objID'].values

    for i in np.arange(len(successIDs)):
        objID = int(successIDs[i])
        hosts.loc[hosts['objID']==objID, 'photo_z'] = point_estimates[i]
    return hosts
