import numpy as np
import os
import pandas as pd
from astropy import units as u
import astropy.coordinates as coord

def clean_dict(dic, df, to_keep):
    """Re-writes the transient, host list dictionary such that
       no potential hosts remain that are not in our data frame
       of potential hosts (you can pass in an array,
       to_keep, of potential hosts to keep in the
       dictionary).

    :param dic: key,value pairs of transient name, list of candidate host PS1
        objIDs.
    :type dic: dictionary
    :param df: PS1 properties for candidate hosts.
    :type df: Pandas DataFrame
    :param to_keep: List of PS1 objIDs for candidates to keep in the dictionary
        (even if they're not in the dataframe).
    :type to_keep: array


    :return: key,value pairs of transient name, list of candidate host PS1
        objIDs (after cleaning).
    :rtype: dictionary
    """

    for name, host in dic.items():
            host = host.tolist()
            newHosts = host
            if len(host) > 0:
                for hostCandidate in np.array(host):
                    if hostCandidate not in np.array(df["objID"]) and hostCandidate not in np.array(to_keep):
                        newHosts.remove(hostCandidate)
                dic[name] = np.array(newHosts)
    return dic

def check_dict(dic, df):
    """Check to make sure that the list of all potential hosts
       in the dataframe matches all potential hosts in the
       dictionary - that the two describe the same hosts.

    :param dic: key,value pairs of transient name, list of candidate host PS1
        objIDs.
    :type dic: dictionary
    :param df: PS1 properties for candidate hosts.
    :type df: Pandas DataFrame
    """

    for name, host in dic.items():
            host = host.tolist()
            newHosts = host
            if len(host) > 0:
                for hostCandidate in np.array(host):
                    if hostCandidate not in np.array(df["objID"]):
                        print("Error: {}".format(hostCandidate))
                        print("objID not found!")

def clean_dict(dic, df, to_keep=[],bestDetectionCut=False):
    """ Clean the dictionary to match the dataset.
        note that if we're cutting by bestDetection, we find that
        we cut out many true hosts. So we only overwrite the
        list of potential hosts in this case if our new list isn't
        empty.

    :param dic: key,value pairs of transient name, list of candidate host PS1
        objIDs.
    :type dic: dictionary
    :param df: PS1 properties for candidate hosts.
    :type df: Pandas DataFrame
    :param to_keep: List of PS1 objIDs for candidates to keep in the dictionary
        (even if they're not in the dataframe).
    :type to_keep: array
    :param bestDetectionCut: If True, bestDetection==1 was a selection cut.
        Don't remove hosts from the dictionary if this would remove all
        of a transient's potential hosts.
    :type bestDetectionCut: bool

    :return: key,value pairs of transient name, list of candidate
        host PS1 objIDs, after removing PS1 objects not in the dataframe.
    :rtype: dictionary
    """

    for name, host in dic.items():
            host = host.tolist()
            newHosts = host
            if len(host) > 0:
                for hostCandidate in np.array(host):
                    if hostCandidate not in np.array(df["objID"]) and hostCandidate not in np.array(to_keep):
                        newHosts.remove(hostCandidate)
                if ((bestDetectionCut==0) | ((bestDetectionCut ==1) & (len(np.array(newHosts) != 0)))):
                    dic[name] = np.unique(newHosts)
                else:
                    dic[name] = np.unique(dic[name])
    return dic

def clean_df_from_dict(dic, df):
    """Remove sources from PS1 object DataFrame if not in the
       dictionary matching transients to their candidate hosts.


    :param dic: key,value pairs of transient name, list of candidate host PS1
        objIDs.
    :type dic: dictionary
    :param df: PS1 properties for candidate hosts.
    :type df: Pandas DataFrame
    :return: PS1 properties, after removing sources.
    :rtype: Pandas DataFrame
    """

    allHosts = np.array([])
    for name, host in dic.items():
        if host.size>0:
            try:
                allHosts = np.concatenate((allHosts, np.array(host)),axis=0)
            except:
                continue
    allHosts = np.unique(allHosts)
    for index, row in df.iterrows():
        if (row['objID'] not in allHosts):
            df.drop(index, inplace=True)
    return df

def removePS1Duplicates(df):
    """Because there are many duplicate entries in PS1 host table, follow this hierarchy for
        prioritizing which ones to keep:
        1. If duplicate, remove non-primary detections
        2. If still duplicate, remove NANs in yKronFlux, yskyErr, and yExtNSigma
        3. If still duplicate, take the value with the smallest yKronFluxErr

    :param df: PS1 properties for candidate hosts.
    :type df: Pandas DataFrame
    :return: PS1 properties, after removing duplicates.
    :rtype: Pandas DataFrame
    """

    df.replace(-999.0,np.nan, inplace=True)
    new_df = []
    for hostCandidate in np.unique(df["objID"]):
        hostFrame = df[df["objID"] == hostCandidate]
        if len(hostFrame) > 1:
            newhostFrame = hostFrame[hostFrame["primaryDetection"] == 1]
            if len(newhostFrame) > 1:
                newhostFrame = newhostFrame[~np.isnan(newhostFrame["rApMag"])]
            if len(newhostFrame) > 1:
                newhostFrame = newhostFrame[newhostFrame["rApMagErr"] == np.min(newhostFrame["rApMagErr"])]
            if len(newhostFrame) > 0:
                new_df.append(newhostFrame)
            else:
                new_df.append(hostFrame)
        else:
            new_df.append(hostFrame)
    if len(new_df) > 0:
        df = pd.concat(new_df)
    return df

def getColors(df):
    """Calulate observer-frame colors for PS1 sources, and make some cuts
       from bad photometry.

    :param df: PS1 properties for candidate hosts.
    :type df: Pandas DataFrame
    :return: PS1 properties, with color and photometry cuts.
    :rtype: Pandas DataFrame
    """

    df.replace(-999, np.nan, inplace=True)
    df.replace(999, np.nan, inplace=True)

    # create color attributes for all hosts
    df["g-r"]= df["gApMag"] - df["rApMag"]
    df["r-i"]= df["rApMag"] - df["iApMag"]
    df["i-z"] = df["iApMag"] - df["zApMag"]
    df["z-y"] = df["zApMag"] - df["yApMag"]

    df['g-rErr'] = np.sqrt(df['gApMagErr']**2 + df['rApMagErr']**2)
    df['r-iErr'] = np.sqrt(df['rApMagErr']**2 + df['iApMagErr']**2)
    df['i-zErr'] = np.sqrt(df['iApMagErr']**2 + df['zApMagErr']**2)
    df['z-yErr'] = np.sqrt(df['zApMagErr']**2 + df['yApMagErr']**2)

    # To be sure we're getting physical colors
    for col in ['g-r', 'r-i', 'i-z', 'z-y']:
        df.loc[np.abs(df[col]) > 100, col] = np.nan

    # and PSF - Kron mag "morphology" information
    for band in 'grizy':
        # to be sure we're getting physical mags
        col = '%sApMag_%sKronMag'%(band, band)
        df[col] = df["%sApMag"%band] - df["%sKronMag"%band]
        df.loc[np.abs(df[col]) > 100, col] = np.nan
        col = '%sApMag'%band
        df.loc[np.abs(df[col]) > 100, col] = np.nan
    return df

def makeCuts(df,cuts=[],dict=""):
    """Make a series of quality cuts on the candidate host galaxies in the dataframe.

    :param df: PS1 properties for candidate hosts.
    :type df: Pandas DataFrame
    :param cuts: List of cuts to apply. Options are:
        'n' - remove objects without at least 10 detections.
        'quality'- remove objects with PS1 qualityFlag > 128 (suggesting bad photometry).
        'coords' - remove objects with missing position information.
        'mag' - remove objects with missing photometry (aperture magnitudes).
        'primary' - remove objects with primaryDetection = 0.
        'best' - remove objects with bestDetection = 0.
        'duplicate' - remove all completely duplicated rows.
    :type cuts: array
    :param dic: key,value pairs of transient name, list of candidate host PS1
        objIDs.
    :type dic: dictionary

    :return: PS1 properties, with quality cuts applied.
    :rtype: Pandas DataFrame
    """

    for cut in cuts:
        if cut == "n":
            df = df[df['nDetections'] >= 10]
        elif cut == "quality":
            df = df[df["qualityFlag"] <= 165]
        elif cut == "coords":
            df = df.dropna(subset=['raMean', 'decMean'])
        elif cut == "mag":
            for band in 'grizy':
                df = df[pd.notnull(df['%sApMag'%band])]
        elif cut =="primary":
            if dict != "":
                df_mod = df[df["primaryDetection"] == 1]
                clean_dict(dict, df_mod, np.array([]),bestDetectionCut=1)
                clean_df_from_dict(dict, df)
            else:
                print("Can't make a primary detection cut - no dict provided!")
                return
        elif cut == "best":
            # subset by best detection, clean the dictionary, but include a flag that only
            # cleans the dict entry if there was a best detection in the field (else leave unchanged!)
            if dict != "":
                df_mod = df[df['bestDetection'] == 1]
                clean_dict(dict, df_mod, np.array([]),bestDetectionCut=1)
                clean_df_from_dict(dict, df)
            else:
                print("Can't make a best detection cut - no dict provided!")
                return
        elif cut=="duplicate":
            df = df.drop_duplicates(subset=['objID'])
        else:
            print("I didn't understand your cut!")
            return
    df.reset_index(drop=True, inplace=True)
    return df
