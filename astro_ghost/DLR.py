import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from astro_ghost.sourceCleaning import clean_dict
from astro_ghost.NEDQueryFunctions import getNEDInfo
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, Distance
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astroquery.ipac.ned import Ned

def choose_band_SNR(host_df):
    """Gets the PS1 band (of grizy) with the highest SNR in PSF mag.
       From https://www.eso.org/~ohainaut/ccd/sn.html,
       Error on Mag  ~    1/  (S/N)
       So calculating S/N for each band as 1/PSFMagErr
       Estimate the S/N of each band and choose the bands
       with the highest S/N for the rest of our measurements.

    :param host_df: The dataframe containing the candidate host galaxy (should just be one galaxy).
    :type host_df: Pandas DataFrame
    :return: The PS1 band with the highest S/N.
    :rtype: str
    """

    host_df.reset_index(inplace=True, drop=True)
    bands = 'grizy'
    SNR = []
    try:
        for band in bands:
             if host_df.iloc[0, "%sKronMagErr"%band] == host_df.iloc[0, "%sKronMagErr"%band]:
                 SNR.append(float(1/host_df.iloc[0, "%sKronMagErr"%band]))
             else:
                 SNR.append(np.nan)
        i = np.nanargmax(np.array(SNR))
    except:
        #if we have issues getting the band with the highest SNR, just use r-band
        i = 1
    return bands[i]

def calc_DLR_glade(ra_SN, dec_SN, ra_host, dec_host, r_a, a_over_b, phi):
    """Calculates the DLR between transients and GLADE host galaxy candidates.

    (very similar to calc_DLR but the parameters are calculated slightly
    differently)

    :param ra_SN: The right ascension of the SN, in degrees.
    :type ra_SN: float
    :param dec_SN: The declination of the SN, in degrees.
    :type dec_SN: float
    :param ra_host: The right ascension of the host, in degrees.
    :type ra_host: float
    :param dec_host: The declination of the host, in degrees.
    :type dec_host: float
    :param r_a: The semi-major axis of the host in arcseconds.
    :type r_a: float
    :param a_over_b: The candidate host axis ratio.
    :type a_over_b: float
    :param phi: The galaxy position angle (in radians).
    :type phi: float
    :return: The angular separation between galaxy and transient, in arcseconds.
    :rtype: float
    :return: The normalized distance (angular separation divided by the DLR).
    :rtype: float

    """
    xr = (ra_SN- float(ra_host))*3600
    yr = (dec_SN - float(dec_host))*3600

    SNcoord = SkyCoord(ra_SN*u.deg, dec_SN*u.deg, frame='icrs')
    hostCoord = SkyCoord(ra_host*u.deg, dec_host*u.deg, frame='icrs')
    sep = SNcoord.separation(hostCoord)
    dist = sep.arcsecond
    badR = 10000000000.0

    gam = np.arctan2(xr, yr)

    theta = phi - gam

    DLR = r_a/np.sqrt(((a_over_b)*np.sin(theta))**2 + (np.cos(theta))**2)

    R = float(dist/DLR)

    if (R != R):
        return dist, badR

    return dist, R

def calc_DLR(ra_SN, dec_SN, ra_host, dec_host, r_a, r_b, source, best_band):
    """Calculate the directional light radius for a given galaxy and transient pair. Calculation is adapted from
       Gupta et al., 2013 found at
       https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1916&context=edissertations.

    :param ra_SN: The right ascension of the SN, in degrees.
    :type ra_SN: float
    :param dec_SN: The declination of the SN, in degrees.
    :type dec_SN: float
    :param ra_host: The right ascension of the host, in degrees.
    :type ra_host: float
    :param dec_host: The declination of the host, in degrees.
    :type dec_host: float
    :param r_a: The semi-major axis of the host in arcseconds.
    :type r_a: float
    :param r_b: The semi-minor axis of the host in arcseconds.
    :type r_b: float
    :param source: The Dataframe containing the PS1 information for the candidate host galaxy.
    :type source: Pandas DataFrame
    :param best_band: The PS1 passband with the highest S/N, from which second-order moments are estimated.
    :type best_band: str
    :return: The angular separation between galaxy and transient, in arcseconds.
    :rtype: float
    :return: The normalized distance (angular separation divided by the DLR).
    :rtype: float
    """

    source.reset_index(inplace=True, drop=True)

    xr = (ra_SN.deg - float(ra_host))*3600
    yr = (dec_SN.deg - float(dec_host))*3600

    SNcoord = SkyCoord(ra_SN, dec_SN, frame='icrs')
    hostCoord = SkyCoord(ra_host*u.deg, dec_host*u.deg, frame='icrs')
    sep = SNcoord.separation(hostCoord)
    dist = sep.arcsecond
    badR = 10000000000.0

    XX = float(source.loc[0, best_band + 'momentXX'])
    YY = float(source.loc[0, best_band + 'momentYY'])
    XY = float(source.loc[0, best_band + 'momentXY'])

    # if we don't have spatial information, get rid of it
    # this gets rid of lots of artifacts without radius information
    if (XX != XX) | (XY != XY) | \
        (YY != YY):
        return dist, badR

    U = XY
    Q = XX - YY
    if Q == 0:
        return dist, badR

    phi = 0.5*np.arctan(U/Q)
    kappa = Q**2 + U**2
    a_over_b = (1 + kappa + 2*np.sqrt(kappa))/(1 - kappa)

    gam = np.arctan2(xr, yr)

    theta = phi - gam

    DLR = r_a/np.sqrt(((a_over_b)*np.sin(theta))**2 + (np.cos(theta))**2)

    R = float(dist/DLR)

    if (R != R):
        return dist, badR

    return dist, R


def calc_DLR_SM(ra_SN, dec_SN, ra_host, dec_host, r_a, elong, phi, source, best_band):
    """ Calculate the DLR method but for Skymapper (southern-hemisphere) sources,
        which don't have xx and yy moments reported in the catalog.

    :param ra_SN: The right ascension of the SN, in degrees.
    :type ra_SN: float
    :param dec_SN: The declination of the SN, in degrees.
    :type dec_SN: float
    :param ra_host: The right ascension of the host, in degrees.
    :type ra_host: float
    :param dec_host: The declination of the host, in degrees.
    :type dec_host: float
    :param r_a: The semi-major axis of the host in arcseconds.
    :type r_a: float
    :param elong: The elongation parameter of the galaxy.
    :type elong: float
    :param phi: The rotation angle of the galaxy, in radians.
    :type phi: float
    :param source: The Dataframe containing the PS1 information for the candidate host galaxy.
    :type source: Pandas DataFrame
    :param best_band: The PS1 passband with the highest S/N, from which second-order moments are estimated.
    :type best_band: str
    :return: The angular separation between galaxy and transient, in arcseconds.
    :rtype: float
    :return: The normalized distance (angular separation divided by the DLR).
    :rtype: float
    """

    # EVERYTHING IS IN ARCSECONDS
    ## taken from "Understanding Type Ia Supernovae Through Their Host Galaxies..." by Gupta
    #https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1916&context=edissertations
    xr = (ra_SN.deg - float(ra_host))*3600
    yr = (dec_SN.deg - float(dec_host))*3600

    SNcoord = SkyCoord(ra_SN, dec_SN, frame='icrs')
    hostCoord = SkyCoord(ra_host*u.deg, dec_host*u.deg, frame='icrs')
    sep = SNcoord.separation(hostCoord)
    dist = sep.arcsecond
    badR = 10000000000.0

    # if we don't have shape information for a southern-hemisphere galaxy, return
    if (float(r_a) != float(r_a)) | (float(elong) != float(elong)):
        return dist, badR

    gam = np.arctan2(xr, yr)

    theta = phi - gam

    #elong == a/b, which allows us to substitute here
    DLR = r_a/np.sqrt(((elong)*np.sin(theta))**2 + (np.cos(theta))**2)

    R = float(dist/DLR)

    if (R != R):
        return dist, badR

    return dist, R

def chooseByDLR(path, hosts, transients, fn, orig_dict, todo="s"):
    """The wrapper function for selecting hosts by the directional light radius method
       introduced in Gupta et al., 2013.

    :param path: Filepath where to write out the results of the DLR algorithm.
    :type path: str
    :param hosts: DataFrame containing PS1 information for all candidate hosts.
    :type hosts: Pandas DataFrame
    :param transients: DataFrame containing TNS information for all transients.
    :type transients: Pandas DataFrame
    :param fn: Filename to write the results of the associations (useful for debugging).
    :type fn: str
    :param orig_dict: Dictionary consisting of key,val pairs of transient names, and lists of
        their candidate host galaxy objIDs in PS1.
    :type orig_dict: dictionary
    :param todo: If todo == \\'s\\', save the dictionary and the list of remaining sources.
        If todo == \\'r\\', return them.
    :type todo: str
    :return: The dataframe of PS1 properties for host galaxies found by DLR.
    :rtype: Pandas DataFrame
    :return: Dictionary of matches after DLR, with transient names as keys and a list of host galaxy pan-starrs objIDs as values.
    :rtype: dictionary
    :return: List of transients for which no reliable host galaxy was found.
    :rtype: array-like
    :return: List of transients for which an issue arose in DLR (most likely, a candidate
        host galaxy in the field didn't have radius information). This list is used
        to recommend candidates to associate via the gradient ascent method.
    :rtype: array-like
    """

    dict_mod = orig_dict.copy()
    if todo=="s":
        if not os.path.exists(path+'/dictionaries'):
             os.makedirs(path+'/dictionaries')
        if not os.path.exists(path+'/tables'):
             os.makedirs(path+'/tables')
    hosts['dist/DLR'] = np.nan
    hosts['dist'] = np.nan
    noHosts = []

    clean_dict(orig_dict, hosts, [])
    f = open(path+fn, 'w')
    GDflag = 0
    GA_SN = []
    for name, host in orig_dict.items():
        if (type(host) is not np.int64 and type(host) is not float):
             if (len(host.shape) > 0) and (host.shape[0] > 0):
                R_dict = {}
                ra_dict = {}
                dist_dict = {}
                host = np.array(host)
                if len(host)>0:
                    for tempHost in host:
                        theta = 0
                        host_df = hosts[hosts["objID"] == tempHost]
                        transient_df = transients[transients["Name"] == str(name)]

                        # If no good, we want to artificially inflate the distance to the SN so that
                        # we don't incorrectly pick this as our host
                        # (but the radius goes into plotting, so ra is artificially shrunk)
                        R =  dist = 1.e10
                        r_a = 0.05

                        #flag for running gradient ascent, if the DLR method fails.
                        GDflag = 1

                        if ":" in str(transient_df["RA"].values):
                            ra_SN = Angle(transient_df["RA"].values, unit=u.hourangle)
                        else:
                            ra_SN = Angle(transient_df["RA"].values, unit=u.deg)
                        dec_SN = Angle(transient_df["DEC"].values, unit=u.degree)
                        ra_host = host_df['raMean'].values[0]
                        dec_host = host_df['decMean'].values[0]
                        if len(np.array(ra_SN)) > 1:
                            ra_SN = ra_SN[0]
                            dec_SN = dec_SN[0]
                        if (dec_SN.deg > -30):
                            band = choose_band_SNR(host_df)
                            r_a = r_b = float(host_df[band + 'KronRad'].values[0])
                            dist, R = calc_DLR(ra_SN, dec_SN, ra_host, dec_host, r_a, r_b, host_df, band)
                        else:
                            band = choose_band_SNR(host_df)
                            r_a = float(host_df['%sKronRad'%band].values[0])
                            if r_a == r_a:
                                elong = host_df[band + "_elong"].values[0]
                                phi = np.radians(host_df[band + "_pa"].values[0])
                                r_a = host_df[band + 'KronRad'].values[0]*0.5 #plate scale

                                # in arcsec, the radius containing 90% of the galaxy light.
                                # This empirically has improved association performance
                                # for southern-hemisphere sources.
                                dist, R = calc_DLR_SM(ra_SN, dec_SN, ra_host, dec_host, r_a, elong, phi, host_df, band)

                        R_dict[tempHost] = R
                        ra_dict[tempHost] = r_a
                        dist_dict[tempHost] = dist

                        hosts.loc[hosts['objID'] == tempHost, 'dist/DLR'] = R
                        hosts.loc[hosts['objID'] == tempHost, 'dist'] = dist

                print("\n transient = \\", file=f)
                print(name, file=f)
                print("offset/DLR = \\", file=f)
                #round for printing purposes
                R_dict_print = {k:round(v,2) if isinstance(v,float) else v for k,v in R_dict.items()}
                print(R_dict_print, file=f)
                ra_dict_print = {k:round(v,2) if isinstance(v,float) else v for k,v in ra_dict.items()}
                print("candidate semi-major axis = \\", file=f)
                print(ra_dict_print, file=f)

                #Subset so that we're less than 5 in DLR units
                #Tentative selection of the host with lowest DLR
                chosenHost = min(R_dict, key=R_dict.get)

                if R_dict[chosenHost] > 5.0:
                    #If we can't find a host, say that this galaxy has no host
                    dict_mod[name] = np.nan
                    noHosts.append(name)
                    print("No host chosen! r/DLR > 5.0.", file=f)
                    continue
                else:
                    #Truncate candidates at <5 DLR.
                    R_dict_sub = dict((k, v) for k, v in R_dict.items() if v <= 5.0)
                    #Sort from lowest to highest DLR value
                    R_dict_sub = {k: v for k, v in sorted(R_dict_sub.items(), key=lambda item: item[1])}

                    #If there are multiple candidates remaining
                    if len(R_dict_sub.keys()) > 1:
                        gal_hosts = []
                        Simbad_hosts = []
                        for key in R_dict_sub:
                            tempType = hosts[hosts['objID'] == key]['NED_type'].values[0]
                            hasSimbad = hosts[hosts['objID'] == key]['hasSimbad'].values[0]
                            if (tempType == "G"):
                                gal_hosts.append(key)
                            if (hasSimbad) & (tempType != '*'):
                                Simbad_hosts.append(key)
                        if len(gal_hosts) > 0:
                            # only change if we're still within the light profile of the galaxy and previous host has DLR > 1
                            # (meaning we're outside of the light profile of the previous host)
                            if (gal_hosts[0] != chosenHost) and (R_dict[gal_hosts[0]] < 5.0) and (R_dict[chosenHost] > 1):
                                chosenHost = gal_hosts[0]
                                print("Choosing the galaxy with the smallest DLR - nearest source had DLR > 1!", file=f)
                        if len(Simbad_hosts) > 0:
                            print("Chosen SIMBAD host!", file=f)
                            if Simbad_hosts[0] != chosenHost and R_dict[Simbad_hosts[0]] < 1.0:
                                chosenHost = Simbad_hosts[0] #only change if we're within the light profile of the galaxy
                                print("Choosing the simbad source with the smallest DLR!", file=f)
                    dict_mod[name] = chosenHost
                    hosts.loc[hosts['objID'] == chosenHost, 'dist/DLR'] = R_dict[chosenHost]
                    hosts.loc[hosts['objID'] == chosenHost, 'dist'] = dist_dict[chosenHost]
                    if (GDflag):
                        GDflag = 0
                        print("Issue with DLR. Try Gradient Descent!", file=f)
                        GA_SN.append(name)
                    print(float(hosts[hosts['objID'] == chosenHost]['raMean'].values[0]), float(hosts[hosts['objID'] == chosenHost]['decMean'].values[0]), file=f)
    f.close()
    if todo == "s":
        with open('../dictionaries/DLR_rankOrdered_hosts.p', 'wb') as fp:
            pickle.dump(dict_mod, fp, protocol=pickle.HIGHEST_PROTOCOL)
        hosts.to_csv("../tables/DLR_rankOrdered_hosts.csv")
        return
    elif todo =="r":
        return hosts, dict_mod, noHosts, GA_SN

def chooseByGladeDLR(path, fn, snDF, verbose=False, todo='r', GWGC=None):
    """The wrapper function for selecting hosts by the DLR method (Gupta et al., 2013).

    Here, candidate hosts are taken from the GLADE (Dalya et al., 2021; arXiv:2110.06184) catalog.

    :param path: Filepath where to write out the results of the DLR algorithm.
    :type path: str
    :param fn: Filename to write the results of the associations (useful for debugging).
    :type fn: str
    :param transients: DataFrame containing TNS information for all transients.
    :type transients: Pandas DataFrame
    :param todo: If todo == \\'s\\', save the dictionary and the list of remaining sources.
        If todo == \\'r\\', return them.
    :type todo: str
    :param GWGC: DataFrame of local galaxies with shape information, from GWGC (White et al., 2011).
    :type GWGC: Pandas DataFrame
    :return: The dataframe of properties for GLADE host galaxies found by DLR.
    :rtype: Pandas DataFrame
    :return: List of transients for which no reliable GLADE host galaxy was found.
    :rtype: array-like
    """
    if todo=="s":
        if not os.path.exists(path+'/dictionaries'):
             os.makedirs(path+'/dictionaries')
        if not os.path.exists(path+'/tables'):
             os.makedirs(path+'/tables')

    f = open(path+fn, 'w')

    foundHostDF = []
    noGladeHosts = []

    if GWGC is not None:
        GWGC_coords = SkyCoord(GWGC['RAJ2000'].values, GWGC['DEJ2000'].values, unit=(u.deg, u.deg))

    #assume standard cosmology for distance estimates
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    for idx, row in snDF.iterrows():
        name = str(row['Name'])
        ra_SN = float(row['RA'])
        dec_SN = float(row['DEC'])
        class_SN = str(row['Obj. Type'])

        #broad cone search of GWGC galaxies:
        seps = SkyCoord(ra=ra_SN, dec=dec_SN,unit=(u.deg, u.deg),frame='icrs').separation(GWGC_coords).deg
        hosts = GWGC[seps < 0.2] #within 0.2 deg

        if len(hosts)<1:
            noGladeHosts.append(name)
            continue
        hosts['MajorRad'] = hosts['maj']*60./2 # diameter to radius in arcsec
        hosts['MinorRad'] = hosts['min']*60./2 # diameter to radius in arcsec

        #get names for the galaxies that match
        hosts.rename(columns={'RAJ2000':'raMean','DEJ2000':'decMean'}, inplace=True)
        hosts = getNEDInfo(hosts)

        R_dict = {}
        ra_dict = {}
        dist_dict = {}
        for idx, row in hosts.iterrows():
            tempHost = row['NED_name']
            phi = np.radians(row['PAHyp'])
            r_a = row['MajorRad']

            #if it's a mostly round galaxy, the position angle doesn't matter!
            if (phi != phi) & (row['a_b'] >= 0.9):
                phi = 0
            dist, R = calc_DLR_glade(ra_SN, dec_SN, row['raMean'], row['decMean'], r_a, row['a_b'], phi)

            R_dict[tempHost] = R
            ra_dict[tempHost] = r_a
            dist_dict[tempHost] = dist

            hosts.loc[hosts['NED_name'] == tempHost, 'dist/DLR'] = R
            hosts.loc[hosts['NED_name'] == tempHost, 'dist'] = dist

        print("\n transient = \\", file=f)
        print(name, file=f)
        print("offset/DLR = \\", file=f)
        #round for printing purposes
        R_dict_print = {k:round(v,2) if isinstance(v,float) else v for k,v in R_dict.items()}
        print(R_dict_print, file=f)
        ra_dict_print = {k:round(v,2) if isinstance(v,float) else v for k,v in ra_dict.items()}
        print("candidate semi-major axis = \\", file=f)
        print(ra_dict_print, file=f)

        #subset so that we're less than 5 in DLR units
        chosenHost = min(R_dict, key=R_dict.get)
        if R_dict[chosenHost] > 5.0:
            #If we can't find a host, say that this galaxy has no host
            noGladeHosts.append(name)
            print("No host chosen! r/DLR > 5.0.", file=f)
            continue
        else:
            print("Selecting GLADE host: %s"%chosenHost, file=f)
            foundHost = hosts.loc[hosts['NED_name'] == chosenHost]

            #add in the associated transient's information
            foundHost['TransientRA'] = ra_SN
            foundHost['TransientDEC'] = dec_SN
            foundHost['TransientName'] = name
            foundHost['TransientClass'] = class_SN
            foundHostDF.append(foundHost)

    if len(foundHostDF) > 0:
        foundHostDF = pd.concat(foundHostDF, ignore_index=True)
        # adding some relevant redshift information
        foundHostDF['GLADE_redshift'] = np.nan
        #foundHostDF['GLADE_redshift_flag'] = ''

        for idx, row in foundHostDF.iterrows():
            if row['Dist'] == row['Dist']:
                dist = Distance(value=row['Dist'], unit=u.Mpc)
                calc_z = z_at_value(cosmo.luminosity_distance, dist, zmin=1.e-5, zmax=1, method='Bounded').value
                foundHostDF['GLADE_redshift'] = calc_z

        #print some relevant information to terminal
        print("Found %i hosts in GLADE! See %s for details."%(len(foundHostDF), fn))
        if todo == "s":
            foundHostDF.to_csv("../tables/gladeDLR_hosts.csv")
            return
        elif todo == "r":
            return foundHostDF, noGladeHosts

    else:
        foundHostDF = pd.DataFrame({'TransientName':[], 'raMean':[], 'decMean':[]})
        print("Found no hosts in GLADE.")
        if todo == 'r':
            return foundHostDF, noGladeHosts
    f.close()
