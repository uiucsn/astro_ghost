import numpy as np
from astropy.table import Table
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
import astro_ghost
from astro_ghost.PS1QueryFunctions import *
from astro_ghost.hostMatching import build_ML_df
from astro_ghost.NEDQueryFunctions import getNEDInfo
from astro_ghost.SimbadQueryFunctions import getSimbadInfo
from astro_ghost.gradientAscent import gradientAscent
from astro_ghost.starSeparation import separateStars_STRM, separateStars_South
from astro_ghost.sourceCleaning import clean_dict, removePS1Duplicates, getColors, makeCuts
from astro_ghost.stellarLocus import calc_7DCD
from astro_ghost.DLR import chooseByDLR, chooseByGladeDLR
import requests
import pickle
import pyvo
import glob
from datetime import datetime
from joblib import dump, load
import pandas as pd

#we do a lot of copies, sub-selects and rewrites - no need to warn about everything!
pd.options.mode.chained_assignment = None  # default='warn'

def cleanup(path):
    """Cleans up the working directory.

    This is done after host-galaxy association is completed.

    :param path: filepath where association files are saved
    :type path: str
    """

    tablePath = path+"/tables/"
    printoutPath = path+'/printouts/'

    paths = [tablePath, printoutPath]

    for tempPath in paths:
        if not os.path.exists(tempPath):
            os.mkdir(tempPath)

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

def getGHOST(real=False, verbose=False, installpath='', clobber=False):
    """Downloads the GHOST database.

    :param real: If True, download the GHOST database. If False, write an empty files with
        relevant columns (so that every transient is manually associated).
    :type real: bool, optional
    :param verbose: If True, print debugging info.
    :type verbose: bool, optional
    :param installpath: Filepath where GHOST database will be installed.
    :type installpath: str
    :param clobber: If True, write new GHOST database even if it already exists at installpath.
    :type clobber: bool, optional
    """

    if not installpath:
        try:
            installpath = os.environ['GHOST_PATH']
        except:
            print("Couldn't find where you want to save GHOST. Setting location to package path...")
            installpath = astro_ghost.__file__
            installpath = installpath.split("/")[:-1]
            installpath = "/".join(installpath)
    if not os.path.exists(installpath + '/database'):
        os.mkdir(installpath + '/database')
    else:
        if os.path.exists(installpath + '/database/GHOST.csv') & (clobber == False):
            print("GHOST database already exists in the install path!")
            return
    if real:
        url = 'https://www.dropbox.com/s/a0fufc3827pfril/GHOST.csv?dl=1'
        r = requests.get(url)
        fname = installpath + '/database/GHOST.csv'
        open(fname , 'wb').write(r.content)
        if verbose:
            print("Successfully downloaded GHOST database.\n")
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
        df.to_csv(installpath + "/database/GHOST.csv",index=False)
        if verbose:
            print("Successfully created dummy database.\n")

def fracWithHosts(transient_dict):
    """Calculates transient fraction with at least one candidate host.

    :param transient_dict: The dictionary of supernovae and their host galaxy candidate objIDs from PS1.
    :type transient_dict: dictionary
    :return: The fraction of supernovae with at least one candidate host galaxy.
    :rtype: dictionary
    """

    count = 0
    for name, host in transient_dict.items():
        # only do matching if there's a found host
        if isinstance(host, list) or isinstance(host, np.ndarray):
            if len(host) > 0 and np.any(np.array(host == host)):
                count += 1
        else:
            if host == host:
                count+=1
    return count/len(transient_dict.keys())

def remove_prefix(text, prefix):
    """Removes the prefix from a string.

    Very useful for removing the 'SN' from supernova names!

    :param text: The full text.
    :type text: str
    :param prefix: The prefix to remove from the text.
    :type prefix: str
    :return: The input text, with prefix removed.
    :rtype: str
    """

    return text[text.startswith(prefix) and len(prefix):]

def checkSimbadHierarchy(df, verbose=False):
    """Throw a warning if the source has a parent in simbad!

    :param df: The final associated data frame containing host and transient features.
    :type df: Pandas DataFrame
    :return: The dataframe, after correcting for SIMBAD hierarchical information.
    :rtype: Pandas DataFrame
    """
    host_DF = df.copy()
    HierarchicalHostedSNe = []
    parents = []
    for idx, row in host_DF.iterrows():
        # cone search of the best-fit host in SIMBAD - if it gets it right,
        #replace the info with the parent information!
        tap_simbad = pyvo.dal.TAPService("https://simbad.u-strasbg.fr/simbad/sim-tap")
        query = """SELECT main_id, otype, basic.ra, basic.dec,
        DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', {0}, {1})) AS dist,
        h_link.membership, cluster.id AS cluster
        FROM (SELECT oid, id FROM basic JOIN ident ON oidref = oid WHERE
        CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{0},{1},0.000556))=1 ) AS cluster, basic
        JOIN h_link ON basic.oid = h_link.parent WHERE h_link.child = cluster.oid ORDER BY dist ASC;
        """.format(row.raMean, row.decMean)
        result = tap_simbad.search(query)
        tap_pandas = result.to_table().to_pandas().reset_index(drop=True)
        tap_pandas.dropna(subset=['membership'], inplace=True)
        if ((not tap_pandas.empty) and (tap_pandas.loc[0, 'membership'] > 50)):
            tap_pandas.drop_duplicates(subset=['main_id'], inplace=True)
            tap_pandas.reset_index(drop=True, inplace=True)
 
            if tap_pandas['main_id'].values[0].startswith("VIRTUAL PARENT"):
                continue

            if verbose:
                print("Warning! Host of %s is the hierarchical child of another object in Simbad, choosing parent as host instead..." % row.TransientName)

            # query PS1 for correct host
            a = ps1cone(tap_pandas.loc[0, 'ra'], tap_pandas.loc[0, 'dec'], 10./3600)
            if a:
                a = ascii.read(a)
                a = a.to_pandas()
                parent = a.iloc[[0]]
                parent['TransientName'] = row.TransientName
                parent['TransientClass'] = row.TransientClass
                parent['TransientRA'] = row.TransientRA
                parent['TransientDEC'] = row.TransientDEC
                parent = getNEDInfo(parent)
                HierarchicalHostedSNe.append(row.TransientName)
                parents.append(parent)
    if len(parents)>0:
        parentDF = pd.concat(parents)
    else:
        parentDF = pd.DataFrame({})
    finalHosts_traditional = host_DF.loc[~host_DF['TransientName'].isin(HierarchicalHostedSNe)]
    host_DF = pd.concat([finalHosts_traditional, parentDF], ignore_index=True)
    return host_DF

def getDBHostFromTransientCoords(transientCoords, GHOSTpath=''):
    """Gets the host of a GHOST transient by position.

    :param transientCoords: A list of astropy SkyCoord coordinates of transients.
    :type transientCoords: array-like
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str

    :return: The PS1 objects associated with the queried transients in GHOST.
    :rtype: Pandas DataFrame
    :return: A list of the coordinates of transients not found in the database.
    :rtype: host_DF : array-like
    """

    fullTable = fullData(GHOSTpath)
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
        c2 = SkyCoord(smallTable['TransientRA'].values*u.deg, smallTable['TransientDEC'].values*u.deg, frame='icrs')
        sep = np.array(transientCoord.separation(c2).arcsec)
        if np.nanmin(sep) <= 1:
            host_idx = np.where(sep == np.nanmin(sep))[0][0]
            host = smallTable.iloc[[host_idx]]
            hostList.append(host)
        else:
            notFound.append(transientCoord)
    if len(hostList) > 0:
        host_DF = pd.concat(hostList, ignore_index=True)
    return host_DF, notFound

def getDBHostFromTransientName(transientNames, GHOSTpath=''):
    """Gets the host of a GHOST transient by transient name.

    :param transientNames: A list of transient names.
    :type transientNames: array-like
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    :return: The PS1 objects associated with the queried transients in GHOST.
    :rtype: Pandas DataFrame
    :return: A list of the coordinates of transients not found in the database.
    :rtype: array-like
    """

    fullTable = fullData(GHOSTpath)
    allHosts = []
    notFound = []
    host_DF = None
    for transientName in transientNames:
        transientName = re.sub(r"\s+", "", str(transientName))
        host = None
        possibleNames = [transientName, transientName.upper(), transientName.lower(), "SN"+transientName]
        for name in possibleNames:
            if len(fullTable[fullTable['TransientName'] == name])>0:
                host = fullTable[fullTable['TransientName'] == name]
                allHosts.append(host)
                break
        if host is None:
            notFound.append(transientName)
    if len(allHosts) > 0:
        host_DF = pd.concat(allHosts, ignore_index=True)
    return host_DF, notFound

def getHostFromHostName(hostNames, GHOSTpath=''):
    """Gets hosts in the GHOST database by name.

    :param hostNames: A list of host galaxy names.
    :type hostNames: array-like
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    :return: The host galaxies found in GHOST.
    :rtype: Pandas DataFrame
    """

    fullTable = fullData(GHOSTpath)
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

def getHostFromHostCoords(hostCoords, GHOSTpath=''):
    """Gets hosts in the GHOST database by coordinates.

    :param hostCoords : A list of astropy SkyCoord coordinates of host galaxies.
    :type hostCoords: array-like
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    :return: The subset of discovered hosts.
    :rtype: Pandas DataFrame
    """

    fullTable = fullData(GHOSTpath)
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

def getTransientStatsFromHostCoords(hostCoord):
    """Prints basic statistics for transient,
       based on a query of the coordinates of
       its host.

    :param hostCoord: The position of the host galaxy.
    :type hostCoord: Astropy SkyCoord Object
    """

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

def getTransientStatsFromHostName(hostName):
    """Returns basic statistics for transient,
       based on a query of the name of
       its host.

    :param hostName: The name of the host galaxy.
    :type hostName: str
    """

    host = getHostFromHostName(hostName)
    hostCoord = SkyCoord(np.unique(host['raMean'])*u.deg, np.unique(host['decMean'])*u.deg, frame='icrs')
    getTransientStatsFromHostCoords(hostCoord)
    return

def getHostStatsFromTransientCoords(transientCoordsList, GHOSTpath=''):
    """ Returns basic statistics for the most likely
        host of a previously identified transient, from the
        transient's coordinates.

    :param transientCoordsList: A list of astropy SkyCoord coordinates of transients.
    :type transientCoordsList: array-like
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    """

    fullTable = fullData(GHOSTpath)
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

def getHostStatsFromTransientName(transientName, GHOSTpath=''):
    """Returns basic statistics for the most likely host of a previously identified transient.

    :param transientName: Array of transient names.
    :type transientName: array-like
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    """

    transientName = np.array(transientName)
    fullTable = fullData(GHOSTpath)
    host, notFound = getDBHostFromTransientName(transientName, GHOSTpath)
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

def getHostImage(transientName='', band="grizy", rad=60, save=False, GHOSTpath=''):
    """Returns a postage stamp of the most likely host in one of the PS1 bands - g,r,i,z,y - as a
       fits file with radius rad, and plots the image.

    :param transientName: Name of queried transient.
    :type transientName: str
    :param band: Band for host-galaxy image.
    :type band: str
    :param rad: Size of the image, in arcsec.
    :type rad: float
    :param save: If True, save host image.
    :type save: bool, optional
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    """

    if transientName == '':
        print("Error! Please enter a supernova!\n")
        return
    fullTable = fullData(GHOSTpath)
    host, notFound = getDBHostFromTransientName(transientName, GHOSTpath)
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

def getTransientSpectra(path, transientName):
    """Retrieves all saved spectra associated with a transient.

    :param path: Filepath to spectra.
    :type path: str
    :param transientName: Name of transient to query.
    :type transientName: str
    :return: Saved spectra.
    :rtype: list of Pandas DataFrames
    """

    transientName = remove_prefix(transientName, 'SN')
    files = glob.glob(path+"*%s*"%transientName)
    specFiles = []
    if len(files) > 0:
        for file in files:
            if remove_prefix(file, path).startswith("osc_"):
                spectra = pd.read_csv(file, header=None, names=['Wave(A)', 'I', 'Ierr'])
            else:
                spectra = pd.read_csv(file, delim_whitespace=True, header=None, names=['x', 'y', 'z'])
            specFiles.append(spectra)
    else:
        print("Sorry! No spectra found.")
    return specFiles

def getHostSpectra(transientName, path):
    """Retrieves all saved spectra associated with a host galaxy.

    :param path: Filepath to spectra.
    :type path: str
    :param transientName: Name of transient to query.
    :type transientName: str
    :return: Saved spectra.
    :rtype: list of Pandas DataFrames
    """

    transientName = remove_prefix(transientName, 'SN')
    files = glob.glob(path+"*%s_hostSpectra.csv*"%transientName)
    specFiles = []
    if len(files) > 0:
        for file in files:
            spectra = pd.read_csv(file)
            specFiles.append(spectra)
    else:
        print("Sorry! No spectra found.")
    return specFiles

def coneSearchPairs(coord, radius, GHOSTpath=''):
    """A cone search for all transient-host pairs within a certain radius, returned as a pandas dataframe.

    :param coord: Astropy SkyCoord
    :type coord: Position for cone search.
    :param radius: Search radius, in arcsec.
    :type radius: float
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    :return: GHOST galaxies within search radius.
    :rtype: Pandas DataFrame
    """

    fullTable = fullData(GHOSTpath)
    c2 = SkyCoord(fullTable['TransientRA']*u.deg, fullTable['TransientDEC']*u.deg, frame='icrs')
    sep = np.array(coord.separation(c2).arcsec)
    hosts = None
    if np.nanmin(sep) < radius:
        host_idx = np.where(sep < radius)[0]
        hosts = fullTable.iloc[host_idx]
    return hosts

def fullData(GHOSTpath=''):
    """Returns the full GHOST database.

    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    :return: GHOST database.
    :rtype: Pandas DataFrame
    """

    if not GHOSTpath:
        GHOSTpath = os.getenv('GHOST_PATH')
        if not GHOSTpath:
            try:
                GHOSTpath = astro_ghost.__file__
                GHOSTpath = GHOSTpath.split("/")[:-1]
                GHOSTpath = "/".join(GHOSTpath)
            except:
                print("Error! I don't know where you installed GHOST -- set GHOST_PATH as an environmental variable or pass in the GHOSTpath parameter.")
    fullTable = pd.read_csv(GHOSTpath+"/database/GHOST.csv")
    return fullTable

def getTransientHosts(transientName=[''], snCoord=[''], snClass=[''], verbose=False, starcut='normal', ascentMatch=False, GLADE=True, px=800, savepath='./', GHOSTpath='', redo_search=True):
    """The wrapper function for the main host association pipeline. The function first
       searches the pre-existing GHOST database by transient name, then by transient coordinates, and finally
       associates the remaining transients not found.

    :param transientName: List of transients to associate.
    :type transientName: array-like
    :param snCoord: List of astropy SkyCoord transient positions.
    :type snCoord: array-like
    :param snClass: List of transient classifications (if they exist).
    :type snClass: array-like
    :param verbose: If True, print logging information.
    :type verbose: bool, optional
    :param starcut: Strings corresponding to the classification thresholds required to classify a star as such.
        Options are \\'gentle\\' (P>0.8), normal (P>0.5), and aggressive (P>0.3).
    :type starcut: str
    :param ascentMatch: If True, run the gradient ascent algorithm for the transients not matched with the
        Directional Light Radius algorithm.
    :type ascentMatch: bool
    :param px: Size of the image used in gradient ascent (ignored if ascentMatch=False).
    :type px: int
    :param savepath: Filepath where dataframe of associated hosts will be saved.
    :type savepath: str
    :param GHOSTpath: The path to the saved GHOST database.
    :type GHOSTpath: str
    :param redo_search: If True, redo the search with 150\\" cone search radius if hosts were not
        found for any of the queried transients.
    :type redo_search: bool
    :return: Final dataframe of associated transients and host galaxies.
    :rtype: Pandas DataFrame
    """

    #if no names were passed in, add placeholder names for each transient in the search
    if len(transientName) < 1:
        transientName = []
        print("No transient names listed, adding placeholder names...")
        for i in np.arange(len(snCoord)):
            transientName.append("Transient_%i" % (i+1))
    hostDB = None
    tempHost1 = tempHost2 = tempHost3 = None
    found_by_name = found_by_coord = found_by_manual = 0
    if not isinstance(snCoord, list) and not isinstance(snCoord, np.ndarray):
        snCoord = [snCoord]
        transientName = [transientName]
        snClass = [snClass]
    if len(snClass) != len(transientName):
        snClass = ['']*len(transientName)

    transientName = [x.replace(" ", "") for x in transientName]
    df_transients = pd.DataFrame({'Name':np.array(transientName), 'snCoord':np.array(snCoord), 'snClass':np.array(snClass)})

    tempHost1, notFoundNames = getDBHostFromTransientName(transientName, GHOSTpath)
    found_by_name = len(transientName) - len(notFoundNames)

    if tempHost1 is None or len(notFoundNames) > 0:
        if verbose:
            print("%i transients not found in GHOST by name, trying a coordinate search..."%len(notFoundNames))

        df_transients_remaining = df_transients[df_transients['Name'].isin(notFoundNames)]

        snCoord_remaining =  df_transients_remaining['snCoord'].values

        tempHost2, notFoundCoords = getDBHostFromTransientCoords(snCoord_remaining, GHOSTpath);
        found_by_coord = len(transientName) - len(notFoundCoords) - found_by_name
        if tempHost2 is None or len(notFoundCoords) > 0:
            if verbose:
                print("%i transients not found in GHOST by name or coordinates, manually associating..."% len(notFoundCoords))

            df_transients_remaining = df_transients_remaining[df_transients_remaining['snCoord'].isin(notFoundCoords)]

            transientName_remaining =  df_transients_remaining['Name'].values
            snCoord_remaining = df_transients_remaining['snCoord'].values
            snClass_remaining = df_transients_remaining['snClass'].values

            tempHost3 = findNewHosts(transientName_remaining, snCoord_remaining, snClass_remaining, verbose, starcut, ascentMatch, px, savepath, GLADE=GLADE)

            if (len(transientName_remaining) > 0) and (len(tempHost3)==0) and (redo_search):
                 #bump up the search radius to 150 arcsec for extremely low-redshift hosts...
                 if verbose:
                     print("Couldn't find any hosts! Trying again with a search radius of 150''.")
                 tempHost3 = findNewHosts(transientName_remaining, snCoord_remaining, snClass_remaining, verbose, starcut, ascentMatch, px, savepath, 150)
            found_by_manual = len(tempHost3)
    hostDB = pd.concat([tempHost1, tempHost2, tempHost3], ignore_index=True)
    hostDB.replace(-999.0, np.nan, inplace=True)
    if verbose:
        print("%i transients found by name, %i transients found by coordinates, %i transients manually associated."% (found_by_name, found_by_coord, found_by_manual))
    return hostDB

def findNewHosts(transientName, snCoord, snClass, verbose=False, starcut='gentle', ascentMatch=False, px=800, savepath='./', rad=60, GLADE=True):
    """Associates hosts of transients not in the GHOST database.

    :param transientName: List of transients to associate.
    :type transientName: array-like
    :param snCoord: List of astropy SkyCoord transient positions.
    :type snCoord: array-like
    :param snClass: List of transient classifications (if they exist).
    :type snClass: array-like
    :param verbose: If True, print logging information.
    :type verbose: bool, optional
    :param starcut: Strings corresponding to the classification thresholds required to classify a star as such.
        Options are \\'gentle\\' (P>0.8), normal (P>0.5), and aggressive (P>0.3).
    :type starcut: str, optional
    :param ascentMatch: If True, run the gradient ascent algorithm for the transients not matched with the
        Directional Light Radius algorithm.
    :type ascentMatch: bool, optional
    :param px: Size of the image used in gradient ascent (ignored if ascentMatch=False).
    :type px: int
    :param savepath: Filepath where dataframe of associated hosts will be saved.
    :type savepath: str
    :param rad: The search radius around each transient position, in arcseconds.
    :type rad: float
    :return: Final dataframe of associated transients and host galaxies.
    :rtype: Pandas DataFrame
    """

    if isinstance(transientName, str):
        transientName = transientName.replace(" ", "")
        snRA = snCoord.ra.degree
        snDEC = snCoord.dec.degree
    elif isinstance(transientName, list) or isinstance(transientName, np.ndarray):
        transientName = [x.replace(" ", "") for x in transientName]
        snRA = [x.ra.degree for x in snCoord]
        snDEC = [x.dec.degree for x in snCoord]
    else:
        print("Error! Please pass in your transient name as either a string or a list/array of strings.\n")

    transientName_arr = np.array(transientName)
    snRA_arr = np.array(snRA)
    snDEC_arr = np.array(snDEC)
    snClass_arr = np.array(snClass)

    dateStr = str(datetime.today()).replace("-", '').replace(".", '').replace(":", "").replace(" ", '')

    fn_host = "SNe_TNS_%s_PS1Hosts_%iarcsec.csv" % (dateStr, rad)
    fn_transients = 'transients_%s.csv' % dateStr
    fn_dict = fn_host[:-4] + ".p"
    dir_name = fn_transients[:-4]
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        os.chmod(savepath, 0o777)
    os.makedirs(savepath + dir_name)
    path = savepath+dir_name+'/'

    #create temp dataframe with RA and DEC corresponding to the transient
    snDF = pd.DataFrame({'Name':transientName_arr, 'RA':snRA_arr, 'DEC':snDEC_arr, 'HostName':['']*len(snDEC_arr), 'Obj. Type':snClass_arr})
    snDF.to_csv(path+fn_transients, index=False)

    #new low-z method (beta) - before we do anything else, find and associate with GLADE
    if GLADE: 
        fn_glade = "gladeDLR.txt"
        foundGladeHosts, noGladeHosts = chooseByGladeDLR(path, fn_glade, snDF, verbose=verbose, todo="r")

        #open transients df and drop the transients already found in GLADE. We'll add these back in at the end
        snDF = snDF[snDF['Name'].isin(noGladeHosts)]
        fn_transients_preGLADE = fn_transients
        fn_transients = 'transients_%s_postGlade.csv' % dateStr
        snDF.to_csv(path+fn_transients, index=False)

    else: 
        foundGladeHosts = []
        noGladeHosts = snDF['Name'].values
        fn_transients_preGLADE = fn_transients

    if len(noGladeHosts) < 1:
        #just return the GLADE df!
        foundGladeHosts['GLADE_source'] = True
        host_DF = foundGladeHosts 
    else:
        #begin doing the heavy lifting after GLADE to associate transients with hosts
        host_DF = get_hosts(path, fn_transients, fn_host, rad)

        if len(host_DF) < 1:
            print("ERROR: Found no hosts in cone search during manual association!")
            return None

        cuts = ["n", "coords", "quality", "duplicate"]

        transient_dict =[]
        # this bit of trickery is required to combine northern-hemisphere and
        # southern-hemisphere source dictionaries
        f = open(path+'/dictionaries/'+fn_dict, "rb")
        try:
            while True:
                transient_dict.append(pickle.load(f))
        except EOFError:
            pass
        temp = transient_dict[0]
        if len(transient_dict) > 1:
            for i in np.arange(len(transient_dict)-1):
                temp.update(transient_dict[i+1])
        transient_dict = {k.replace(' ', ''): v for k, v in temp.items()}
        desperate_dict = transient_dict.copy()

        host_DF = getNEDInfo(host_DF)

        host_DF_north = host_DF[host_DF['decMean']>-30].reset_index(drop=True)
        host_DF_south = host_DF[host_DF['decMean']<=-30].reset_index(drop=True)

        host_DF_north = makeCuts(host_DF_north, cuts, transient_dict)

        host_DF = pd.concat([host_DF_north, host_DF_south], ignore_index=True)

        cut_dict = clean_dict(transient_dict, host_DF, [])
        desperate_dict = cut_dict.copy()

        hostFrac = fracWithHosts(cut_dict)*100
        if verbose:
            print("Associated fraction after quality cuts: %.2f%%."%hostFrac)

        #automatically add to host association for gradient ascent
        lost = np.array([k for k, v in cut_dict.items() if len(v) <1])

        host_DF_north = getColors(host_DF_north)
        host_DF_north = removePS1Duplicates(host_DF_north)
        host_DF_north = calc_7DCD(host_DF_north)

        host_DF = pd.concat([host_DF_north, host_DF_south], ignore_index=True)
        host_DF = getSimbadInfo(host_DF)
        host_DF.to_csv(path+"/candidateHosts_NEDSimbad.csv", index=False)

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

        #in debugging mode, plotting the ML-identified stars and galaxies along the
        #stellar locus can be super helpful!!
        #plotLocus(host_gals_DF, color=1, save=1, type="Gals", timestamp=dateStr)
        #plotLocus(stars_DF, color=1, save=1, type="Stars", timestamp=dateStr)

        host_dict_nospace = {k.replace(' ', ''): v for k, v in cut_dict.items()}

        fn = "DLR.txt"
        transients = pd.read_csv(path+fn_transients)

        with open(path+"/dictionaries/checkpoint_preDLR.p", 'wb') as fp:
               dump(host_dict_nospace, fp)

        host_DF, host_dict_nospace_postDLR, noHosts, GD_SN = chooseByDLR(path, host_gals_DF, transients, fn, host_dict_nospace, todo="r")

        #last-ditch effort -- for the ones with no found host, just pick the nearest NED galaxy.
        for transient in noHosts:
              tempDF = host_gals_DF[host_gals_DF['objID'].isin(desperate_dict[transient])]
              tempDF_gals = tempDF[tempDF['NED_type'].isin(['G', 'PofG', 'GPair', 'GGroup', 'GClstr'])].reset_index()
              if len(tempDF_gals) < 1:
                 continue
              transientRA = transients.loc[transients['Name'] == transient, 'RA'].values[0]
              transientDEC = transients.loc[transients['Name'] == transient, 'DEC'].values[0]
              transientCoord = SkyCoord(transientRA*u.deg, transientDEC*u.deg, frame='icrs')
              tempHostCoords = SkyCoord(tempDF_gals['raMean'].values*u.deg, tempDF_gals['decMean'].values*u.deg, frame='icrs')
              sep = transientCoord.separation(tempHostCoords)
              desperateMatch = tempDF_gals.iloc[[np.argmin(sep.arcsec)]]
              host_DF = pd.concat([host_DF, desperateMatch], ignore_index=True)
              host_dict_nospace_postDLR[transient] = desperateMatch['objID'].values[0]
              if verbose:
                   print("Desperate match found for %s, %.2f arcsec away." % (transient, sep[np.argmin(sep.arcsec)].arcsec))

        if len(noHosts) > 0:
            with open(path+"/dictionaries/noHosts_fromDLR.p", 'wb') as fp:
                dump(noHosts, fp)

        if len(GD_SN) > 0:
            with open(path+"/dictionaries/badInfo_fromDLR.p", 'wb') as fp:
                 dump(GD_SN, fp)

        #gradient ascent algorithm for the SNe that didn't pass this stage
        SN_toReassociate = np.concatenate([np.array(noHosts), np.array(GD_SN), np.array(list(lost))])

        if (len(SN_toReassociate) > 0) and (ascentMatch):
            if verbose:
                print("%i transients with no host found with DLR, %i transients with bad host data with DLR." %(len(noHosts), len(GD_SN)))
                print("Running gradient ascent for %i remaining transients."%len(SN_toReassociate))
                print("See GradientAscent.txt for more information.")

            fn_GD= path+'/GradientAscent.txt'

            host_dict_nospace_postDLR_GD, host_DF, unchanged = gradientAscent(path, transient_dict,  host_dict_nospace_postDLR, SN_toReassociate, host_DF, transients, fn_GD, plot=verbose, px=px)

            with open(path+"/dictionaries/gals_postGD.p", 'wb') as fp:
                dump(host_dict_nospace_postDLR_GD, fp)

            if verbose:
                print("Hosts not found for %i transients in gradient ascent. Storing names in GD_unchanged.txt" %(len(unchanged)))

            with open(path+"/GD_unchanged.txt", 'wb') as fp:
                dump(unchanged, fp)

            hostFrac = fracWithHosts(host_dict_nospace_postDLR_GD)*100

            if verbose:
                print("Associated fraction after gradient ascent: %.2f%%."%hostFrac)

            final_dict = host_dict_nospace_postDLR_GD.copy()

        else:
            final_dict = host_dict_nospace_postDLR.copy()

        host_DF = build_ML_df(final_dict, host_DF, transients)

        # add the glade sources back in!
        if len(foundGladeHosts) > 0:
            if verbose:
                print("Adding %i sources from GLADE back into the catalog..."%len(foundGladeHosts))

            host_DF['GLADE_source'] = False
            foundGladeHosts['GLADE_source'] = True

            #get PS1 photometry for GLADE sources by crossmatching
            ps1matches = []
            for idx, row in foundGladeHosts.iterrows():
                a = ps1cone(row.raMean, row.decMean, 10./3600)
                if a:
                    a = ascii.read(a)
                    a = a.to_pandas()
                    ps1match = a.iloc[[0]] 
                    #get rid of coord info - GLADE properties are better!
                    ps1match.drop(['raMean', 'decMean'], axis=1, inplace=True)
                    foundGladeHosts.loc[foundGladeHosts.index == idx, 'objID'] = ps1match['objID'].values[0]
                    ps1matches.append(ps1match)
            ps1matches = pd.concat(ps1matches)
            foundGladeHosts = foundGladeHosts.merge(ps1matches, on=['objID'])


            #combine
            host_DF = pd.concat([host_DF, foundGladeHosts], ignore_index=True)

        with open(path+"/dictionaries/" + "Final_Dictionary.p", 'wb') as fp:
               dump(final_dict, fp)

    host_DF = checkSimbadHierarchy(host_DF, verbose=verbose)

    #a few final cleaning steps
    #first, add back in some features 
    host_DF_north = host_DF[host_DF['decMean']>-30].reset_index(drop=True)
    host_DF_south = host_DF[host_DF['decMean']<=-30].reset_index(drop=True)
    if len(host_DF_north)>0: 
        host_DF_north = getColors(host_DF_north)
        host_DF_north = calc_7DCD(host_DF_north)
    host_DF = pd.concat([host_DF_north, host_DF_south], ignore_index=True)
    host_DF.drop_duplicates(subset=['TransientName'], inplace=True)
    host_DF = host_DF[host_DF['TransientName'] != ""]
    host_DF.reset_index(inplace=True, drop=True)
    host_DF['TransientName'] = [x.replace(" ", "") for x in host_DF['TransientName']]

    allTransients = pd.read_csv(path+fn_transients_preGLADE)

    matchFrac = len(host_DF)/len(allTransients)*100
    print("Found matches for %.1f%% of events."%matchFrac)
    if verbose:
        print("Saving table of hosts to %s."%(path+"tables/FinalAssociationTable.csv"))

    host_DF.to_csv(path+"FinalAssociationTable.csv", index=False)

    #sort things into the relevant folders
    cleanup(path)

    #remove if there's an extra index column
    try:
        del host_DF['index']
    except:
        pass
    return host_DF
