from __future__ import print_function
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
#from PS1QueryFunctions import *
import json
from astropy.coordinates import Angle
from astropy import units as u
import astropy.coordinates as coord

## between two dataframes, remove all the rows in the second one
def removeDuplicateSN_hostTable(host_DF, severalHosts):
    for host in severalHosts:
        host_DF_sub = host_DF[host_DF['NED_name'] == host]
        tempCoord = coord.SkyCoord(row.TransientRA*u.deg,row.TransientDEC*u.deg, frame='icrs')
        sep = coord.SkyCoord(host_DF_sub['TransientRA']*u.deg,host_DF_sub['TransientDEC']*u.deg, frame='icrs').separation(tempCoord)
        idx = np.argmin(sep)
        idx_DF = host_DF.index[idx]
        if (sep[idx].arcsec <= 1) and (row.TransientDiscoveryYear == host_DF_sub.loc[idx_DF, 'TransientDiscoveryYear']):
            #print('supernova to match is: ')
            #print(index)
            print("Matched:")
            print("%s" % row['TransientName'])
            print("%s" % row.TransientClass)
            print("%s "%host_DF_sub.loc[idx_DF, 'TransientName'])
            print("%s" %host_DF_sub.loc[idx_DF, 'TransientClass'])
            host_DF.drop([idx_DF], inplace=True)
            host_DF.reset_index(inplace=True, drop=True)
        else:
            print("No matches, smallest separation is:")
            print(sep[idx].arcsec)
    return host_DF

## between two dataframes, remove all the rows in the second one
def find_unique_transients(transients, OSC_df):
    for index, row in transients.iterrows():
        print(index)
        tempCoord = coord.SkyCoord(row.RA_deg*u.deg,row.DEC_deg*u.deg, frame='icrs')
        sep = coord.SkyCoord(OSC_df['RA_deg']*u.deg,OSC_df['DEC_deg']*u.deg, frame='icrs').separation(tempCoord)
        idx = np.argmin(sep)

        if sep[idx].arcsec < 2 and row.Year == OSC_df.loc[idx, 'Year']:
            #print('supernova to match is: ')
            #print(index)
            print("TNS SNe: %s" % row.Name)
            print("TNS Type: %s" % row.ObjType)
            #print("from")
            #print(row.Source)
            #print("supernova matched is:")
            #print(idx)
            print("OSC SNe: %s "%OSC_df.loc[idx, 'Name'])
            print("OSC Type: %s" %OSC_df.loc[idx, 'ObjType'])
            #print("from")
            #print(OSC_df.loc[idx, 'Source'])
            #print("distance between the two is:")
            #print(sep[idx])
            OSC_df.drop([idx], inplace=True)
            OSC_df.reset_index(inplace=True, drop=True)
        else:
            print("No matches, smallest separation is:")
            print(sep[idx].arcsec)

# re-write the dictionary such that
# no potential hosts exist that are not
# in our data frame of potential hosts
# (you can pass in an array, to_keep,
# of potential hosts to keep in the
# dictionary even if they're not in
# the dataframe)
def clean_dict(dic, df, to_keep):
    for name, host in dic.items():
            host = host.tolist()
            newHosts = host
            if len(host) > 0:
                for hostCandidate in np.array(host):
                    if hostCandidate not in np.array(df["objID"]) and hostCandidate not in np.array(to_keep):
                        newHosts.remove(hostCandidate)
                dic[name] = np.array(newHosts)
    return dic

# check to make sure that the list of all potential hosts
# in the dataframe matches all potential hosts in the
# dictionary - that the two describe the same hosts
def check_dict(dic, df):
    for name, host in dic.items():
            host = host.tolist()
            newHosts = host
            if len(host) > 0:
                for hostCandidate in np.array(host):
                    if hostCandidate not in np.array(df["objID"]):
                        print("Error: {}".format(hostCandidate))
                        print("objID not found!")

# clean the dictionary to match the datasetself
# note that if we're cutting by bestDetection, we find that
# we cut out many true hosts. So we only overwrite the
# list of potential hosts in this case if our new list isn't
# empty
def clean_dict(dic, df, to_keep=[],bestDetectionCut=0):
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

#Because there are many duplicate entries in PS1 host table, follow this hierarchy:
# 1. If duplicate, remove non-primary detections
# 2. If still duplicate, remove NANs in yKronFlux, yskyErr, and yExtNSigma
# 3. If still duplicate, take the value with the smallest yKronFluxErr
def removePS1Duplicates(df):
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

# TODO add description
def getColors(df):
    df.replace(-999, np.nan, inplace=True)
    df.replace(999, np.nan, inplace=True)
    # create color attributes for all hosts
    df["i-z"] = df["iApMag"] - df["zApMag"]
    df["g-r"]= df["gApMag"] - df["rApMag"]
    df["r-i"]= df["rApMag"] - df["iApMag"]
    df["g-i"] = df["gApMag"] - df["iApMag"]
    df["z-y"] = df["zApMag"] - df["yApMag"]

    df['g-rErr'] = np.sqrt(df['gApMagErr']**2 + df['rApMagErr']**2)
    df['r-iErr'] = np.sqrt(df['rApMagErr']**2 + df['iApMagErr']**2)
    df['i-zErr'] = np.sqrt(df['iApMagErr']**2 + df['zApMagErr']**2)
    df['z-yErr'] = np.sqrt(df['zApMagErr']**2 + df['yApMagErr']**2)

    # To be sure we're getting physical colors
    df.loc[df['i-z'] > 100, 'i-z'] = np.nan
    df.loc[df['i-z'] < -100, 'i-z'] = np.nan
    df.loc[df['g-r'] > 100, 'i-z'] = np.nan
    df.loc[df['g-r'] < -100, 'i-z'] = np.nan

    # and PSF - Kron mag "morphology" information
    df["gApMag_gKronMag"] = df["gApMag"] - df["gKronMag"]
    df["rApMag_rKronMag"] = df["rApMag"] - df["rKronMag"]
    df["iApMag_iKronMag"] = df["iApMag"] - df["iKronMag"]
    df["zApMag_zKronMag"] = df["zApMag"] - df["zKronMag"]
    df["yApMag_yKronMag"] = df["yApMag"] - df["yKronMag"]
    #df["gApMag_rApMag"] = df["gApMag"] - df["rApMag"]
    #df["iApMag_zApMag"] = df["iApMag"] - df["zApMag"]

    # to be sure we're getting physical mags
    df.loc[df['iApMag_iKronMag'] > 100, 'iApMag_iKronMag'] = np.nan
    df.loc[df['iApMag_iKronMag'] < -100, 'iApMag_iKronMag'] = np.nan
    df.loc[df['iApMag'] > 100, 'iApMag'] = np.nan
    df.loc[df['iApMag'] < -100, 'iApMag'] = np.nan
    return df

# TODO Add description
def makeCuts(df,cuts=[],dict=""):
    for cut in cuts:
        if cut == "n":
            df = df[df['nDetections'] >= 1]
            df = df[df['ng'] >= 1]
            df = df[df['nr'] >= 1]
            df = df[df['ni'] >= 1]
            #df = df[df['nz'] >= 1]
            #df = df[df['ny'] >= 1]
        elif cut == "quality":
            df = df[df["qualityFlag"] < 128]
        elif cut == "coords":
            df = df.dropna(subset=['raMean', 'decMean'])
        elif cut == "mag":
            df = df[pd.notnull(df['gApMag'])]
            df = df[pd.notnull(df['rApMag'])]
            df = df[pd.notnull(df['iApMag'])]
            df = df[pd.notnull(df['zApMag'])]
            df = df[pd.notnull(df['yApMag'])]
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
