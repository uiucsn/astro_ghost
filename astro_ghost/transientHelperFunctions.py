import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from SNghost import PS1QueryFunctions as ps1
from astropy.io import fits

# Returns a list of potential hosts for a named transient
# input:  transientName - name of transient as reported on TNS
# output: hostList      - potential hosts in PS1
def get_host(transientName):
    os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/dictionaries')
    PS1_SNe = pickle.load( open( "SNe_TNS_061019_PS1_hosts_lookup_NoStars.p", "rb" ) )
    hostList = PS1_SNe[str(transientName)]
    return hostList

# Returns a dataframe of info from NED, DES, or PS1 for a transient's potential hosts
# Note: if multiple hosts are given in the PS1 dictionary, then each potential host
# is a row in the returned dataframe
# input:  RA            - RA of the transient as reported on TNS (in deg)
#         DEC           - DEC of the transient as reported on TNS (in deg)
#         source        - the source of the data i.e. NED, DES, or PS1
# output: hostInfo      - the info for all potential hosts in PS1
def get_host_info(RA, DEC, source):
    tempRA = np.round(RA, 3)
    tempDEC = np.round(DEC, 3)

    if source == "NED":
        os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/')
        df = pd.read_csv('SNe_TNS_061019_PS1_hosts_NoStars_NEDInfo.csv')
        df = df[(np.round(df["raMean"],3) == tempRA) & (np.round(df["raMean"],3) == tempDEC)]
        hostInfo = df[["objID", "NED_Query"]]
    elif source == "PS1":
        os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/')
        df = pd.read_csv('SNe_TNS_061019_PS1_hosts_NoStars.csv')
        hostInfo = df[(np.round(df["raMean"],3) == tempRA) & (np.round(df["raMean"],3) == tempDEC)]
    elif source == "DES":
        os.chdir('/home/alexgagliano/Documents/Research/Transient_ML_Box/')
        df = pd.read_csv('SNe_TNS_061019_DES_hosts_20s.csv.gz')
        hostInfo = df[(np.round(df["ra"],3) == tempRA) & (np.round(df["dec"],3) == tempDEC)]
    else:
        print("I didn't understand the information source you provided!\n")
        return
    return hostInfo

# Returns a dataframe of info from NED, DES, or PS1 for a transient's potential hosts
# input:  transientName - name of transient as reported on TNS
# output: hostInfo      - potential hosts in PS1_hosts
def get_host_info_by_name(transientName, source):
    hostList = get_host(transientName)
    if source == "NED":
        os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/')
        df = pd.read_csv('SNe_TNS_061019_PS1_hosts_NoStars_NEDInfo.csv')
        df = df[df["objID"] in hostList]
        hostInfo = df[["objID", "NED_Query"]]
    elif source == "PS1":
        os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/')
        df = pd.read_csv('SNe_TNS_061019_PS1_hosts_NoStars.csv')
        hostInfo = df[df["objID"] in hostList]
    elif source == "DES":
        os.chdir('/home/alexgagliano/Documents/Research/Transient_ML_Box/')
        df = pd.read_csv('SNe_TNS_061019_DES_hosts_20s.csv.gz')
        hostInfo = df[df["objID"] in hostList]
    else:
        print("I didn't understand the information source you provided!\n")
        return
    return hostInfo

# Returns the lightcurve data from a given transient
# input:  transientName - name of transient as reported on TNS
# output: lightcurve    - A dataframe of time, flux, and fluxerr for the transient
def load_lightcurve(transientName):
    os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/dictionaries')
    a = find_all("{}_TNS_spectra.txt".format(transientName), ".")
    if not a:
        a = find_all('{}_sneSpace_spectra.json'.format(transientName), ".")
        if not a:
            print("Spectral data not found for {}.\n".format(transientName))
            return
    # READ AND PROCESS lightcurve
    # CONVERT TO CSV
    return lightcurve

# Returns a list of transients associated with a given host
# input:  hostID        - the ID of the host in PS1
# output: transientList - the list of potential transients in TNS
def get_transient_list(hostID):
    transientList = []
    os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/dictionaries')
    PS1_SNe = pickle.load( open( "SNe_TNS_061019_PS1_hosts_lookup_NoStars.p", "rb" ) )
    for name, host in PS1_SNe.items():
        if hostID in host:
            transientList.append(name)
    if not transientList:
        print("I couldn't find any transients associated with this host!\n")
    return transientList


# Returns a list of the names of transients matching a certain class
# input:  transientClass - name of the class of interest
# output: transientList      - list of names of transients matching that class
def get_transient_class(transientClass):
    os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/')
    df = pd.read_csv('SNe_TNS_061019.csv')
    transientList = df[[transientClass in x for x in df["ObjType"]]]
    return transientList["Name"]

# Returns the image of a host in the band specified
# input:  objID - the objID of the host
#         band - color band for the image
# output: postageStamp      - the 10''x10'' postage stamp of the host
def get_picture_by_objID(objID, band):
    os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/')
    host = pd.read_csv("SNe_TNS_061019_PS1_hosts_notNull_NoStars.csv")
    tempRA = host[host["objID"] == objID]["raMean"]
    tempDEC = host[host["objID"] == objID]["decMean"]
    os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/PS1_PostageStamps')
    postageName = find_all("PS1_ra={}_dec={}_{}arcsec_g.fits".format(tempRA, tempDEC, 10), ".")
    if not postageName:
        print("I couldn't find that postage stamp!")
        return
    else:
        postageStamp = fits.open(postageName)
        return postageStamp

# Returns the image of a host in the band specified
# input:  RA   - RA of the host in PS1, in degrees
#         DEC  - DEC of the host in PS1, in degrees
#         band - color band for the image
# output: postageStamp      - the 10''x10'' postage stamp of the host
def get_picture_by_coords(RA, DEC, band):
    os.chdir('/home/alexgagliano/Documents/Research/Transient_ML/PS1_PostageStamps')
    postageName = find_all("PS1_ra={}_dec={}_{}arcsec_g.fits".format(RA, DEC, 10), ".")
    if not postageName:
        print("I couldn't find that postage stamp!")
        return
    else:
        postageStamp = fits.open(postageName)
        return postageStamp

# Set the host of a transient in the PS1 dictionary
# input:  transientName - name of transient as reported on TNS
#         hostList      - potential hosts in PS1_hosts
# output: 0 if successful, -1 if failed
def set_host(transientName, hostName):
    try:
        PS1_SNe = pickle.load( open( "SNe_TNS_061019_PS1_hosts_lookup_NoStars.p", "rb" ) )
        PS1_SNe[transientName] = hostName
        with open('SNe_TNS_061019_PS1_hosts_lookup_NoStars.p', 'wb') as fp:
            pickle.dump(PS1_SNe, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return 0
    except:
        return -1
