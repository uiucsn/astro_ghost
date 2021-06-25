import numpy as np
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import pathlib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import re
from astro_ghost.PS1QueryFunctions import getcolorim, get_hosts
from astro_ghost.hostMatching import build_ML_df
from astro_ghost.NEDQueryFunctions import getNEDInfo
from astro_ghost.gradientAscent import gradientAscent
from astro_ghost.starSeparation import separateStars_STRM, separateStars_South
from astro_ghost.sourceCleaning import clean_dict, removePS1Duplicates, getColors, makeCuts
from astro_ghost.stellarLocus import calc_7DCD
from astro_ghost.DLR import chooseByDLR
import requests
import pickle
import os
import glob
from datetime import datetime
import astro_ghost

def getGHOST(real=False, verbose=False):
    install_path = astro_ghost.__file__
    install_path = install_path.split("/")[:-1]
    install_path = "/".join(install_path)
    if not os.path.exists(install_path + '/database'):
        os.makedirs(install_path + '/database')
    if real:
        url = 'http://ghost.ncsa.illinois.edu/static/database/GHOST.csv'
        r = requests.get(url)
        fname = install_path + '/database/GHOST.csv'
        open(fname , 'wb').write(r.content)
        if verbose:
            print("Successfully downloaded GHOST database from ghost.ncsa.illinois.edu.\n")
    else:
        #create dummy database
        colnames = ['objName', 'objAltName1', 'objAltName2', 'objAltName3', 'objID',
       'uniquePspsOBid', 'ippObjID', 'surveyID', 'htmID', 'zoneID',
       'tessID', 'projectionID', 'skyCellID', 'randomID', 'batchID',
       'dvoRegionID', 'processingVersion', 'objInfoFlag', 'qualityFlag',
       'raStack', 'decStack', 'raStackErr', 'decStackErr', 'raMean',
       'decMean', 'raMeanErr', 'decMeanErr', 'epochMean', 'posMeanChisq',
       'cx', 'cy', 'cz', 'lambda', 'beta', 'l', 'b', 'nStackObjectRows',
       'nStackDetections', 'nDetections', 'ng', 'nr', 'ni', 'nz', 'ny',
       'uniquePspsSTid', 'primaryDetection', 'bestDetection',
       'gippDetectID', 'gstackDetectID', 'gstackImageID', 'gra', 'gdec',
       'graErr', 'gdecErr', 'gEpoch', 'gPSFMag', 'gPSFMagErr', 'gApMag',
       'gApMagErr', 'gKronMag', 'gKronMagErr', 'ginfoFlag', 'ginfoFlag2',
       'ginfoFlag3', 'gnFrames', 'gxPos', 'gyPos', 'gxPosErr', 'gyPosErr',
       'gpsfMajorFWHM', 'gpsfMinorFWHM', 'gpsfTheta', 'gpsfCore',
       'gpsfLikelihood', 'gpsfQf', 'gpsfQfPerfect', 'gpsfChiSq',
       'gmomentXX', 'gmomentXY', 'gmomentYY', 'gmomentR1', 'gmomentRH',
       'gPSFFlux', 'gPSFFluxErr', 'gApFlux', 'gApFluxErr', 'gApFillFac',
       'gApRadius', 'gKronFlux', 'gKronFluxErr', 'gKronRad', 'gexpTime',
       'gExtNSigma', 'gsky', 'gskyErr', 'gzp', 'gPlateScale',
       'rippDetectID', 'rstackDetectID', 'rstackImageID', 'rra', 'rdec',
       'rraErr', 'rdecErr', 'rEpoch', 'rPSFMag', 'rPSFMagErr', 'rApMag',
       'rApMagErr', 'rKronMag', 'rKronMagErr', 'rinfoFlag', 'rinfoFlag2',
       'rinfoFlag3', 'rnFrames', 'rxPos', 'ryPos', 'rxPosErr', 'ryPosErr',
       'rpsfMajorFWHM', 'rpsfMinorFWHM', 'rpsfTheta', 'rpsfCore',
       'rpsfLikelihood', 'rpsfQf', 'rpsfQfPerfect', 'rpsfChiSq',
       'rmomentXX', 'rmomentXY', 'rmomentYY', 'rmomentR1', 'rmomentRH',
       'rPSFFlux', 'rPSFFluxErr', 'rApFlux', 'rApFluxErr', 'rApFillFac',
       'rApRadius', 'rKronFlux', 'rKronFluxErr', 'rKronRad', 'rexpTime',
       'rExtNSigma', 'rsky', 'rskyErr', 'rzp', 'rPlateScale',
       'iippDetectID', 'istackDetectID', 'istackImageID', 'ira', 'idec',
       'iraErr', 'idecErr', 'iEpoch', 'iPSFMag', 'iPSFMagErr', 'iApMag',
       'iApMagErr', 'iKronMag', 'iKronMagErr', 'iinfoFlag', 'iinfoFlag2',
       'iinfoFlag3', 'inFrames', 'ixPos', 'iyPos', 'ixPosErr', 'iyPosErr',
       'ipsfMajorFWHM', 'ipsfMinorFWHM', 'ipsfTheta', 'ipsfCore',
       'ipsfLikelihood', 'ipsfQf', 'ipsfQfPerfect', 'ipsfChiSq',
       'imomentXX', 'imomentXY', 'imomentYY', 'imomentR1', 'imomentRH',
       'iPSFFlux', 'iPSFFluxErr', 'iApFlux', 'iApFluxErr', 'iApFillFac',
       'iApRadius', 'iKronFlux', 'iKronFluxErr', 'iKronRad', 'iexpTime',
       'iExtNSigma', 'isky', 'iskyErr', 'izp', 'iPlateScale',
       'zippDetectID', 'zstackDetectID', 'zstackImageID', 'zra', 'zdec',
       'zraErr', 'zdecErr', 'zEpoch', 'zPSFMag', 'zPSFMagErr', 'zApMag',
       'zApMagErr', 'zKronMag', 'zKronMagErr', 'zinfoFlag', 'zinfoFlag2',
       'zinfoFlag3', 'znFrames', 'zxPos', 'zyPos', 'zxPosErr', 'zyPosErr',
       'zpsfMajorFWHM', 'zpsfMinorFWHM', 'zpsfTheta', 'zpsfCore',
       'zpsfLikelihood', 'zpsfQf', 'zpsfQfPerfect', 'zpsfChiSq',
       'zmomentXX', 'zmomentXY', 'zmomentYY', 'zmomentR1', 'zmomentRH',
       'zPSFFlux', 'zPSFFluxErr', 'zApFlux', 'zApFluxErr', 'zApFillFac',
       'zApRadius', 'zKronFlux', 'zKronFluxErr', 'zKronRad', 'zexpTime',
       'zExtNSigma', 'zsky', 'zskyErr', 'zzp', 'zPlateScale',
       'yippDetectID', 'ystackDetectID', 'ystackImageID', 'yra', 'ydec',
       'yraErr', 'ydecErr', 'yEpoch', 'yPSFMag', 'yPSFMagErr', 'yApMag',
       'yApMagErr', 'yKronMag', 'yKronMagErr', 'yinfoFlag', 'yinfoFlag2',
       'yinfoFlag3', 'ynFrames', 'yxPos', 'yyPos', 'yxPosErr', 'yyPosErr',
       'ypsfMajorFWHM', 'ypsfMinorFWHM', 'ypsfTheta', 'ypsfCore',
       'ypsfLikelihood', 'ypsfQf', 'ypsfQfPerfect', 'ypsfChiSq',
       'ymomentXX', 'ymomentXY', 'ymomentYY', 'ymomentR1', 'ymomentRH',
       'yPSFFlux', 'yPSFFluxErr', 'yApFlux', 'yApFluxErr', 'yApFillFac',
       'yApRadius', 'yKronFlux', 'yKronFluxErr', 'yKronRad', 'yexpTime',
       'yExtNSigma', 'ysky', 'yskyErr', 'yzp', 'yPlateScale', 'distance',
       'NED_name', 'NED_type', 'NED_vel', 'NED_redshift', 'NED_mag',
       'i-z', 'g-r', 'r-i', 'g-i', 'z-y', 'g-rErr', 'r-iErr', 'i-zErr',
       'z-yErr', 'gApMag_gKronMag', 'rApMag_rKronMag', 'iApMag_iKronMag',
       'zApMag_zKronMag', 'yApMag_yKronMag', '7DCD', 'class', 'dist/DLR',
       'dist', 'TransientClass', 'TransientRA', 'TransientDEC',
       'TransientDiscoveryDate', 'TransientDiscoveryMag',
       'TransientRedshift', 'TransientDiscoveryYear', 'Transient AltName',
       'host_logmass', 'host_logmass_min', 'host_logmass_max',
       'Hubble Residual', 'TransientName']
        df = pd.DataFrame(columns = colnames)
        df.to_csv(install_path + "/database/GHOST.csv",index=False)
        if verbose:
            print("Successfully created dummy database.\n")

# Calculates the fraction of supernovae
# in our dictionary that have at least
# one candidate host associated with
# them.
# input: transient_dict, the dictionary
#        of supernovae and their host
#        galaxy candidate objIDs from PS1
# output: the fraction of supernovae
#         with candidate hosts
def fracWithHosts(transient_dict):
    count = 0
    for name, host in transient_dict.items():
        # only do matching if there's a found host
        if isinstance(host, list) or isinstance(host, np.ndarray):
            if len(host) > 0 and np.any(np.array(host == host)):
                count += 1
        #elif isinstance(host, np.ndarray):
        #    if len(host) > 0 and np.any(np.array(host == host)):
        #        count += 1
        else:
            if host == host:
                count+=1
    return count/len(transient_dict.keys())

# Removes the prefix from a string
# input - text and prefix to remove from text
# output - text without prefix
# very useful for removing the 'SN' from
# supernova names!
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

# Gets the host associated with a supernova in
# the GHOST database, if it exists. If not,
# this method runs through the pipeline to
# find a host for a new event, and returns all
# data for the transient host.
# input - TransientCoord, the coordinates of
#         a transient being queried
# output - A data frame of the host
#          associated with the transient
#          being queried, with PS1 information.
#          If the supernova is currently in the
#          database, return the database object.
#          If not, complete the pipeline for
#          host association and return the database
#          object.
#          If no host is found, return None

# Returns all data for a transient host
# currently in the database based on its
# coordinates.
# input - TransientCoord, the coordinates of
#         a new transient being queried
# output - A data frame of the host
#          associated with the transient
#          being queried, with PS1 information.
#          If the supernova is currently in the
#          database, return the database object.
#          If not found, return None
def getDBHostFromTransientCoords(transientCoords):
    fullTable = fullData()
    notFound = []
    host_DF = None
    hostList = []

    #a little wrapper so that this function can take lists of coords in addition to one coord
    for transientCoord in transientCoords:
        smallTable = fullTable[np.abs(fullTable['TransientRA'] - transientCoord.ra.degree)<0.1]
        smallTable = smallTable[np.abs(smallTable['TransientDEC'] - transientCoord.dec.degree)<0.1]
        if len(smallTable) < 1:
            notFound.append(transientCoord)
            continue
        c2 = SkyCoord(smallTable['TransientRA']*u.deg, smallTable['TransientDEC']*u.deg, frame='icrs')
        sep = np.array(transientCoord.separation(c2).arcsec)
        if np.nanmin(sep) <= 1:
            host_idx = np.where(sep == np.nanmin(sep))[0][0]
            host = smallTable.iloc[[host_idx]]
            hostList.append(host)
        else:
            notFound.append(transientCoord)
        #    print("Sorry, that supernova was not found in our database! The closest supernova is %.2f arcsec away.\n" %np.nanmin(sep))
    if len(hostList) > 0:
        host_DF = pd.concat(hostList, ignore_index=True)
    return host_DF, notFound

# Returns all data for transient host,
# based on supernova name
# input - TransientCoord, the coordinates of
#         a new transient being queried
# output - A data frame of the host
#          associated with the transient
#          being queried, with PS1 information.
#          If the supernova is currently in the
#          database, return the database object.
#          If no host is found, return None
def getDBHostFromTransientName(SNNames):
    fullTable = fullData()
    allHosts = []
    notFound = []
    host_DF = None
#    if isinstance(SNName, list) or isinstance(SNName, np.ndarray):
#        SNNames = SNName
#    else:
#        SNNames = np.array([SNname])
    for SNName in SNNames:
        SNName = re.sub(r"\s+", "", str(SNName))
        host = None
        possibleNames = [SNName, SNName.upper(), SNName.lower(), "SN"+SNName]
        for name in possibleNames:
            if len(fullTable[fullTable['TransientName'] == name])>0:
                host = fullTable[fullTable['TransientName'] == name]
                allHosts.append(host)
                break
        if host is None:
            notFound.append(SNName)
    if len(allHosts) > 0:
        host_DF = pd.concat(allHosts, ignore_index=True)
    return host_DF, notFound

# Returns all data for transient and host,
# based on host name
# input - hostName
# output - A data frame of the host
#          associated with the transient
#          being queried, with PS1 information.
#          If the supernova is currently in the
#          database, return the database object.
#          If no host is found, return None
def getHostFromHostName(hostNames):
    fullTable = fullData()
    possibleNames = []

    for name in hostNames:
        possibleNames.append(name)
        possibleNames.append(name.upper())
        possibleNames.append(re.sub(r"\s+", "", str(name)))
        possibleNames.append(name.lower())

    host = fullTable[fullTable['NED_name'].isin(possibleNames)]
    if host is None:
        print("Sorry, no hosts were found in our database!\n")
    return host

def getHostFromHostCoords(hostCoords):
    fullTable = fullData()
    c2 = SkyCoord(fullTable['raMean']*u.deg, fullTable['decMean']*u.deg, frame='icrs')
    host = []
    for hostCoord in hostCoords:
        sep = np.array(hostCoord.separation(c2).arcsec)
        if np.nanmin(sep) <= 1:
            host_idx = np.where(sep == np.nanmin(sep))[0][0]
            host.append(fullTable.iloc[[host_idx]])

    if len(host) < 1:
        print("Sorry, No hosts found in our database!\n")
    else:
        host = pd.concat(host, ignore_index=True)
    return host

# Returns basic statistics for transient,
# based on a query of the coordinates of
# its host
# inputs - the position of the host
# outputs - A printout of name,
#           discovery year and discovery mag
#           for all transients associated
#           with a host at that coordinates

def getTransientStatsFromHostCoords(hostCoord):
    host = getHostFromHostCoords(hostCoord)
    i = 0
    if len(host) > 0:
        print("Found info for host %s.\n"%host['NED_name'].values[0])
        print("Associated supernovae: ")
        for SN in np.array(host['TransientName'].values):
            SN_frame = host.loc[host['TransientName'] == SN]
            print("%i. %s"%(i+1, SN_frame['TransientName'].values[0]))
            print("RA, DEC (J2000): %f, %f"%(SN_frame['TransientRA'].values[0], SN_frame['TransientDEC'].values[0]))
            print("Redshift: %f"%SN_frame['TransientRedshift'].values[0])
            print("Discovery Date: %s"%SN_frame['TransientDiscoveryDate'].values[0].split(" ")[0])
            print("Discovery Mag: %.2f"%SN_frame['TransientDiscoveryMag'].values[0])
            i+= 1
    return

# Returns basic statistics for transient,
# based on a query of the coordinates of
# its host
# inputs - the position of the host
# outputs - A printout of name,
#           discovery year and discovery mag
#           for all transients associated
#           with a host at that coordinates
def getTransientStatsFromHostName(hostName):
    host = getHostFromHostName(hostName)
    hostCoord = SkyCoord(np.unique(host['raMean'])*u.deg, np.unique(host['decMean'])*u.deg, frame='icrs')
    getTransientStatsFromHostCoords(hostCoord)
    return

# Returns basic statistics for the most likely
# host of a previously identified transient, using
# the pipeline outlined above.
# inputs - the coordinates of the transient to search
def getHostStatsFromTransientCoords(transientCoordsList):
    fullTable = fullData()
    names = []
    for transientCoords in transientCoordsList:
        c2 = SkyCoord(fullTable['TransientRA']*u.deg, fullTable['TransientDEC']*u.deg, frame='icrs')
        sep = np.array(transientCoords.separation(c2).arcsec)
        host = None
        if np.nanmin(sep) <= 1:
            host_idx = np.where(sep == np.nanmin(sep))[0][0]
            host = fullTable.iloc[[host_idx]]
            names.append(host['TransientName'].values[0])
    getHostStatsFromTransientName(names)

# Returns basic statistics for the most likely
# host of a previously identified transient, using
# the pipeline outlined above.
# inputs - the coordinates of the transient to search
def getHostStatsFromTransientName(SNName):
    SNName = np.array(SNName)
    fullTable = fullData()
    host, notFound = getDBHostFromTransientName(SNName)
    if host is not None:
        for idx, row in host.iterrows():
            if np.unique(row['NED_name']) != "":
                print("Found host %s.\n"%row['NED_name'])
            else:
                print("Found host with PS1 ID %s"%row['objID'])
            print("RA, DEC (J2000): %f, %f"%(row['raMean'], row['decMean']))
            if row['NED_redshift'] != '':
                print("Redshift: %f"%row['NED_redshift'])
            print("PS1 rMag: %.2f"%row['rApMag'])
            print("g-r          r-i          i-z")
            print("%.2f+/-%.3f   %.2f+/-%.3f   %.2f+/-%.3f"%(row['g-r'],row['g-rErr'], row['r-i'], row['r-iErr'], row['i-z'], row['i-zErr']))
        #    print("g-r: %.2f."%host['g-r'].values[0])
        #    print("r-i: %.2f."%host['r-i'].values[0])
        #    print("i-z: %.2f."%host['i-z'].values[0])
            #print("There are %i supernovae associated with this object.\n"%len(host))
            print("Associated supernovae: ")
            if np.unique(row['NED_name']) != "":
                tempHost = fullTable[fullTable['NED_name'] == row['NED_name']]
            else:
                tempHost = fullTable[fullTable['objID'] == row['objID']]
            for SN in np.array(tempHost['TransientName'].values):
                print(SN)
    else:
        print("No host info found!")
    return

# Returns a postage stamp of the most likely
# host in one of the PS1 bands - g,r,i,z,y - as a
# fits file with radius rad, and plots the image.
# inputs
# outputs
def getHostImage(transientName='', band="grizy", rad=60, save=0):
    if transientName == '':
        print("Error! Please enter a supernova!\n")
        return
    fullTable = fullData()
    host, notFound = getDBHostFromTransientName(transientName)
    if host is not None:
        host.reset_index(drop=True, inplace=True)
        tempSize = int(4*float(rad))
        fn_save = host['objID'].values[0]
        if np.unique(host['NED_name']) != "":
            fn_save = host['NED_name'].values[0]
            print("Showing postage stamp for %s"%np.unique(host['NED_name'])[0])
        ra = np.unique(host['raMean'])[0]
        dec = np.unique(host['decMean'])[0]
        tempSize = int(4*rad)
        img = getcolorim(ra, dec, output_size=tempSize, size=tempSize, filters=band, format="png")
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        if save:
            img.save("%s.png" % fn_save)
        return
    else:
        print("Transient host not found!")

# description
# inputs
# outputs
def getTransientSpectra(path, SNname):#
    SNname = remove_prefix(SNname, 'SN')
    files = glob.glob(path+"*%s*"%SNname)
    specFiles = []
    if len(files) > 0:
        for file in files:
            if remove_prefix(file, path).startswith("osc_"):
                spectra = pd.read_csv(file, header=None, names=['Wave(A)', 'I', 'Ierr'])
            else:
                spectra = pd.read_csv(file, delim_whitespace=True, header=None, names=['x', 'y', 'z'])
            #spectra = spectra.astype('float')
            specFiles.append(spectra)
    else:
        print("Sorry! No spectra found.")
    return specFiles

# description
# inputs
# outputs
def getHostSpectra(SNname, path):#
    SNname = remove_prefix(SNname, 'SN')
    files = glob.glob(path+"*%s_hostSpectra.csv*"%SNname)
    specFiles = []
    if len(files) > 0:
        for file in files:
            spectra = pd.read_csv(file)
            #spectra = spectra.astype('float')
            specFiles.append(spectra)
    else:
        print("Sorry! No spectra found.")
    return specFiles

# A cone search for all transient-host pairs
# within a certain radius, returned as a pandas dataframe
# inputs location, in astropy coordinates, and radius in arcsec
# outputs the data frame of all nearby pairs
def coneSearchPairs(coord, radius):
    fullTable = fullData()
    c2 = SkyCoord(fullTable['TransientRA']*u.deg, fullTable['TransientDEC']*u.deg, frame='icrs')
    sep = np.array(coord.separation(c2).arcsec)
    hosts = None
    if np.nanmin(sep) < radius:
        host_idx = np.where(sep < radius)[0]
        hosts = fullTable.iloc[host_idx]
        #hosts = pd.concat(hosts)
    return hosts

# Returns the full table of data
# inputs: none
# outputs: the full GHOST database
def fullData():
    install_path = astro_ghost.__file__
    install_path = install_path.split("/")[:-1]
    install_path = "/".join(install_path)
    fullTable = pd.read_csv(install_path+"/database/GHOST.csv")
    return fullTable

# The wrapper function for the
# host association pipeline.
# inputs: the location of the supernova
#         as an astropy SkyCoord object
# output: A pandas dataframe of
#         the most likely host in PS1,
#         with stats provided at
#         printout
def getTransientHosts(snName=[''], snCoord=[''], snClass=[''], verbose=0, starcut='normal', ascentMatch=False):
    hostDB = None
    tempHost1 = tempHost2 = tempHost3 = None
    found_by_name = found_by_coord = found_by_manual = 0
    if not isinstance(snCoord, list) and not isinstance(snCoord, np.ndarray):
        snCoord = [snCoord]
        snName = [snName]
        snClass = [snClass]
    if len(snClass) != len(snName):
        snClass = ['']*len(snName)

    snName = [x.replace(" ", "") for x in snName]
    df_transients = pd.DataFrame({'Name':np.array(snName), 'snCoord':np.array(snCoord), 'snClass':np.array(snClass)})

    tempHost1, notFoundNames = getDBHostFromTransientName(snName)
    found_by_name = len(snName) - len(notFoundNames)

    if tempHost1 is None or len(notFoundNames) > 0:
        if verbose:
            print("%i transients not found in GHOST by name, trying a coordinate search..."%len(notFoundNames))

        df_transients_remaining = df_transients[df_transients['Name'].isin(notFoundNames)]

        snCoord_remaining =  df_transients_remaining['snCoord'].values

        tempHost2, notFoundCoords = getDBHostFromTransientCoords(snCoord_remaining);
        found_by_coord = len(snName) - len(notFoundCoords) - found_by_name
        if tempHost2 is None or len(notFoundCoords) > 0:
            if verbose:
                print("%i transients not found in GHOST by name or coordinates, manually associating..."% len(notFoundCoords))

            df_transients_remaining = df_transients_remaining[df_transients_remaining['snCoord'].isin(notFoundCoords)]

            snName_remaining =  df_transients_remaining['Name'].values
            snCoord_remaining = df_transients_remaining['snCoord'].values
            snClass_remaining = df_transients_remaining['snClass'].values

            tempHost3 = findNewHosts(snName_remaining, snCoord_remaining, snClass_remaining, verbose, starcut, ascentMatch)
            found_by_manual = len(tempHost3)
    hostDB = pd.concat([tempHost1, tempHost2, tempHost3], ignore_index=True)
    hostDB.replace(-999.0, np.nan, inplace=True)
    if verbose:
        print("%i transients found by name, %i transients found by coordinates, %i transients manually associated."% (found_by_name, found_by_coord, found_by_manual))
    return hostDB

# The wrapper function for the
# host association pipeline.
# inputs: the location of the supernova
#         as an astropy SkyCoord object
# output: A pandas dataframe of
#         the most likely host in PS1,
#         with stats provided at
#         printout
def findNewHosts(snName, snCoord, snClass, verbose=0, starcut='gentle', ascentMatch=False):
    if isinstance(snName, str):
        snName = snName.replace(" ", "")
        snRA = snCoord.ra.degree
        snDEC = snCoord.dec.degree
    elif isinstance(snName, list) or isinstance(snName, np.ndarray):
        snName = [x.replace(" ", "") for x in snName]
        snRA = [x.ra.degree for x in snCoord]
        snDEC = [x.dec.degree for x in snCoord]
    else:
        print("Error! Please pass in your transient name as either a string or a list/array of strings.\n")
        #return None

    snName_arr = np.array(snName)
    snRA_arr = np.array(snRA)
    snDEC_arr = np.array(snDEC)
    snClass_arr = np.array(snClass)

    #now = datetime.now()
    #dateStr = "%i%.02i%.02i" % (now.year,now.month,now.day)
    dateStr = str(datetime.today()).replace("-", '').replace(".", '').replace(":", "").replace(" ", '')

    rad = 30 #arcsec
    fn_Host = "SNe_TNS_%s_PS1Hosts_%iarcsec.csv" % (dateStr, rad)
    fn_SN = 'transients_%s.csv' % dateStr
    fn_Dict = fn_Host[:-4] + ".p"
    dir_name = fn_SN[:-4]
    if not os.path.exists('./' + dir_name):
        os.makedirs('./' + dir_name)
    path = './'+dir_name+'/'
    #create temp dataframe with RA and DEC corresponding to the transient

    snDF = pd.DataFrame({'Name':snName_arr, 'RA':snRA_arr, 'DEC':snDEC_arr, 'HostName':['']*len(snDEC_arr), 'Obj. Type':snClass_arr})
    snDF.to_csv(path+fn_SN, index=False)

    #begin doing the heavy lifting to associate transients with hosts
    host_DF = get_hosts(path, fn_SN, fn_Host, rad)

    if len(host_DF) < 1:
        print("ERROR: Found no hosts in cone search during manual association!")
        return None

    #turn off all cuts now for debugging
    cuts = ["n", "quality", "coords", "duplicate"]
    #cuts = []

    transient_dict = pickle.load(open(path+'/dictionaries/'+fn_Dict, "rb"))
    transient_dict = {k.replace(' ', ''): v for k, v in transient_dict.items()}
    # check how many supernovae we have potential hosts for - should be nearly all of em

    host_DF_north = host_DF[host_DF['decMean']>-30].reset_index()
    host_DF_south = host_DF[host_DF['decMean']<=-30].reset_index()

    host_DF_north = makeCuts(host_DF_north, cuts, transient_dict)

    host_DF = pd.concat([host_DF_north, host_DF_south], ignore_index=True)

    cut_dict = clean_dict(transient_dict, host_DF, [])
    hostFrac = fracWithHosts(cut_dict)*100
    if verbose:
        print("Associated fraction after quality cuts: %.2f%%."%hostFrac)

    #automatically add to host association for gradient ascent
    lost = np.array([k for k, v in cut_dict.items() if len(v) <1])
    #lost = set(snName_arr) - set(cut_dict.keys())

    host_DF_north = getColors(host_DF_north)
    host_DF_north = removePS1Duplicates(host_DF_north)
    host_DF_north = calc_7DCD(host_DF_north)

    host_DF = pd.concat([host_DF_north, host_DF_south], ignore_index=True)
    host_DF = getNEDInfo(host_DF)

    host_DF_north = host_DF[host_DF['decMean']>-30].reset_index()
    host_DF_south = host_DF[host_DF['decMean']<=-30].reset_index()

    host_gals_DF_north, stars_DF_north = separateStars_STRM(host_DF_north, plot=0, verbose=verbose, starcut=starcut)
    host_gals_DF_south, stars_DF_south = separateStars_South(host_DF_south, plot=0, verbose=verbose, starcut=starcut)
    host_gals_DF = pd.concat([host_gals_DF_north, host_gals_DF_south],ignore_index=True)
    stars_DF = pd.concat([stars_DF_north, stars_DF_south],ignore_index=True)

    if verbose:
        print("Removed %i stars. We now have %i candidate host galaxies."%(len(stars_DF), len(host_gals_DF)))

    cut_dict = clean_dict(cut_dict, host_gals_DF, [])
    stars_DF.to_csv(path+"removedStars.csv",index=False)

    #plotLocus(host_gals_DF, color=1, save=1, type="Gals", timestamp=dateStr)
    #plotLocus(stars_DF, color=1, save=1, type="Stars", timestamp=dateStr)

    host_dict_nospace = {k.replace(' ', ''): v for k, v in cut_dict.items()}

    fn = "DLR.txt"
    transients = pd.read_csv(path+fn_SN)
#    clean_dict(host_dict_nospace, host_gals_DF, [])
#    fracWithHosts(host_dict_nospace) 
#    print(len(host_gals_DF[host_gals_DF['decMean'] < -30]))

    with open(path+"/dictionaries/checkpoint_preDLR.p", 'wb') as fp:
           pickle.dump(host_dict_nospace, fp, protocol=pickle.HIGHEST_PROTOCOL)

    host_DF, host_dict_nospace_postDLR, noHosts, GD_SN = chooseByDLR(path, host_gals_DF, transients, fn, host_dict_nospace, host_dict_nospace, todo="r")

    if len(noHosts) > 0:
        with open(path+"/noHosts_fromDLR_griCut.txt", 'wb') as fp:
            pickle.dump(noHosts, fp)

    if len(GD_SN) > 0:
        with open(path+"/badInfo_fromDLR_griCut.txt", 'wb') as fp:
            pickle.dump(GD_SN, fp)

    #gradient ascent algorithm for the SNe that didn't pass this stage
    SN_toReassociate = np.concatenate([np.array(noHosts), np.array(GD_SN), np.array(list(lost))])
    #SN_toReassociate = np.concatenate([np.array(noHosts), np.array(list(lost))])
    #SN_toReassociate = np.array(SN_toReassociate)

    if (len(SN_toReassociate) > 0) and (ascentMatch):
        if verbose:
            print("%i transients with no host found with DLR, %i transients with bad host data with DLR." %(len(noHosts), len(GD_SN)))
            print("Running gradient ascent for %i remaining transients."%len(SN_toReassociate))
            print("See GradientAscent.txt for more information.")

        fn_GD= path+'/GradientAscent.txt'

        host_dict_nospace_postDLR_GD, host_DF, unchanged = gradientAscent(path, transient_dict,  host_dict_nospace_postDLR, SN_toReassociate, host_DF, transients, fn_GD, plot=verbose)

        with open(path+"/dictionaries/gals_postGD.p", 'wb') as fp:
            pickle.dump(host_dict_nospace_postDLR_GD, fp, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print("Hosts not found for %i transients in gradient ascent. Storing names in GD_unchanged.txt" %(len(unchanged)))

        with open(path+"/GD_unchanged.txt", 'wb') as fp:
            pickle.dump(unchanged, fp)

        hostFrac = fracWithHosts(host_dict_nospace_postDLR_GD)*100

        if verbose:
            print("Associated fraction after gradient ascent: %.2f%%."%hostFrac)

        final_dict = host_dict_nospace_postDLR_GD.copy()

    else:
        final_dict = host_dict_nospace_postDLR.copy()

    host_DF = build_ML_df(final_dict, host_DF, transients)

    with open(path+"/dictionaries/" + "Final_Dictionary.p", 'wb') as fp:
           pickle.dump(final_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #a few final cleaning steps
    host_DF.drop_duplicates(subset=['TransientName'], inplace=True)
    host_DF = host_DF[host_DF['TransientName'] != ""]
    host_DF.reset_index(inplace=True, drop=True)
    host_DF['TransientName'] = [x.replace(" ", "") for x in host_DF['TransientName']]

    matchFrac = len(host_DF)/len(transients)*100
    print("Found matches for %.1f%% of events."%matchFrac)
    if verbose:
        print("Saving table of hosts to %s."%(path+"tables/FinalAssociationTable.csv"))

    host_DF.to_csv(path+"FinalAssociationTable.csv", index=False)

    #hSpecPath = path+"/hostSpectra/"
    #tSpecPath = path+"/SNspectra/"
    #psPath = path+"/hostPostageStamps/"
    tablePath = path+"/tables/"
    printoutPath = path+'/printouts/'

    #paths = [hSpecPath, tSpecPath, psPath, tablePath, printoutPath]
    paths = [tablePath, printoutPath]

    for tempPath in paths:
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)

    #move tables to the tables directory, and printouts to the printouts directory
    printouts = glob.glob(path+'/*.txt')
    for p in printouts:
        fn = remove_prefix(p, path)
        os.rename(p, printoutPath+fn)

    table_ext = ['csv', 'gz']
    for end in table_ext:
        tables = glob.glob(path+'/*.%s'%end)
        for t in tables:
            fn = remove_prefix(t, path)
            os.rename(t, tablePath+fn)

    return(host_DF)
