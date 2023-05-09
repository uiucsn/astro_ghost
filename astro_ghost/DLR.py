import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from astro_ghost.sourceCleaning import clean_dict
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

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

    bands = 'grizy'
    try:
        gSNR = float(1/host_df["gPSFMagErr"])
        rSNR = float(1/host_df["rPSFMagErr"])
        iSNR = float(1/host_df["iPSFMagErr"])
        zSNR = float(1/host_df["zPSFMagErr"])
        ySNR = float(1/host_df["yPSFMagErr"])

        SNR = np.array([gSNR, rSNR, iSNR, zSNR, ySNR])
        i = np.nanargmax(SNR)
    except:
        #if we have issues getting the band with the highest SNR, just use r-band
        i = 1
    return bands[i]

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

    xr = np.abs(ra_SN.deg - float(ra_host))*3600
    yr = np.abs(dec_SN.deg - float(dec_host))*3600

    SNcoord = SkyCoord(ra_SN, dec_SN, frame='icrs')
    hostCoord = SkyCoord(ra_host*u.deg, dec_host*u.deg, frame='icrs')
    sep = SNcoord.separation(hostCoord)
    dist = sep.arcsecond
    badR = 10000000000.0

    XX = best_band + 'momentXX'
    YY = best_band + 'momentYY'
    XY = best_band + 'momentXY'

    # if we don't have spatial information, get rid of it
    # this gets rid of lots of artifacts without radius information
    if (float(source[XX]) != float(source[XX])) | (float(source[XY]) != float(source[XY])) | \
        (float(source[YY]) != float(source[YY])):
        return dist, badR

    U = float(source[XY])
    Q = float(source[XX]) - float(source[YY])
    if Q == 0:
        return dist, badR

    phi = 0.5*np.arctan(U/Q)
    kappa = Q**2 + U**2
    a_over_b = (1 + kappa + 2*np.sqrt(kappa))/(1 - kappa)

    gam = np.arctan(yr/xr)
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
    xr = np.abs(ra_SN.deg - float(ra_host))*3600
    yr = np.abs(dec_SN.deg - float(dec_host))*3600

    SNcoord = SkyCoord(ra_SN, dec_SN, frame='icrs')
    hostCoord = SkyCoord(ra_host*u.deg, dec_host*u.deg, frame='icrs')
    sep = SNcoord.separation(hostCoord)
    dist = sep.arcsecond
    badR = 10000000000.0

    # if we don't have shape information for a southern-hemisphere galaxy, return
    if (float(r_a) != float(r_a)) | (float(elong) != float(elong)):
        return dist, badR

    gam = np.arctan(yr/xr)
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
                        ra_host = host_df['raMean']
                        dec_host = host_df['decMean']
                        if len(np.array(ra_SN)) > 1:
                            ra_SN = ra_SN[0]
                            dec_SN = dec_SN[0]
                        if (dec_SN.deg > -30):
                            #switching from kron radius to petrosian radius (more robust!)
                            for band in 'gri':
                                #temp_r_a = float(host_df[band + 'HalfLightRad'].values[0])
                                try:
                                    temp_r_a = float(host_df[band + 'petR90'].values[0])
                                except:
                                    temp_r_a = np.nan
                                if (temp_r_a == temp_r_a) & (temp_r_a > 0):
                                    #r_a = r_b = float(host_df[band + 'HalfLightRad'].values[0])
                                    r_a = r_b = float(host_df[band + 'petR90'].values[0])
                                    dist, R = calc_DLR(ra_SN, dec_SN, ra_host, dec_host, r_a, r_b, host_df, band)
                                    break
                        else:
                            band = choose_band_SNR(host_df)
                            r_a = float(host_df['%sradius_frac90'%band].values[0])
                            if r_a == r_a:
                                elong = host_df[band + "_elong"].values[0]
                                phi = np.radians(host_df[band + "_pa"].values[0])
                                r_a = host_df[band + 'radius_frac90'].values[0]*0.5 #plate scale

                                # in arcsec, the radius containing 90% of the galaxy light.
                                # This empirically has improved association performance
                                # for southern-hemisphere sources.
                                dist, R = calc_DLR_SM(ra_SN, dec_SN, ra_host, dec_host, r_a, elong, phi, host_df, band)

                        R_dict[tempHost] = R
                        ra_dict[tempHost] = r_a
                        dist_dict[tempHost] = dist

                        hosts.loc[hosts['objID'] == tempHost, 'dist/DLR'] = R
                        hosts.loc[hosts['objID'] == tempHost, 'dist'] = dist

                print(name, file=f)
                print("transient = \\", file=f)
                print(name, file=f)
                print("R_dict = \\", file=f)
                print(R_dict, file=f)
                print("ra_dict = \\", file=f)
                print(ra_dict, file=f)

                #subset so that we're less than 5 in DLR units
                chosenHost = min(R_dict, key=R_dict.get)
                if R_dict[chosenHost] > 5.0:
                    #If we can't find a host, say that this galaxy has no host
                    dict_mod[name] = np.nan
                    noHosts.append(name)
                    print("No host chosen! r/DLR > 5.0.", file=f)
                    continue
                else:
                    R_dict_sub = dict((k, v) for k, v in R_dict.items() if v <= 5.0)
                    #Sort from lowest to highest DLR value
                    R_dict_sub = {k: v for k, v in sorted(R_dict_sub.items(), key=lambda item: item[1])}

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
                            if gal_hosts[0] != chosenHost and R_dict[gal_hosts[0]] < 5.0:
                                chosenHost = gal_hosts[0] #only change if we're within the light profile of the galaxy
                                print("Choosing the galaxy with the smallest DLR - nearest source had DLR > 1!", file=f)
                        if len(Simbad_hosts) > 0:
                            print("Chosen SIMBAD host!", file=f)
                            if Simbad_hosts[0] != chosenHost and R_dict[Simbad_hosts[0]] < 5.0:
                                chosenHost = Simbad_hosts[0] #only change if we're within the light profile of the galaxy
                                print("Choosing the simbad source with the smallest DLR!", file=f)
                    dict_mod[name] = chosenHost
                    hosts.loc[hosts['objID'] == chosenHost, 'dist/DLR'] = R_dict[chosenHost]
                    hosts.loc[hosts['objID'] == chosenHost, 'dist'] = dist_dict[chosenHost]
                    if (GDflag):
                        GDflag = 0
                        print("Issue with DLR. Try Gradient Descent!", file=f)
                        GA_SN.append(name)
                    print(float(hosts[hosts['objID'] == chosenHost]['raMean']), float(hosts[hosts['objID'] == chosenHost]['decMean']), file=f)
                f.flush()
    f.close()
    if todo == "s":
        with open('../dictionaries/DLR_rankOrdered_hosts.p', 'wb') as fp:
            pickle.dump(dict_mod, fp, protocol=pickle.HIGHEST_PROTOCOL)
        hosts.to_csv("../tables/DLR_rankOrdered_hosts.csv")
        return
    elif todo =="r":
        return hosts, dict_mod, noHosts, GA_SN
