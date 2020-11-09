from matplotlib import colors
from scipy import ndimage
from astropy.wcs import WCS
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
from astro_ghost.sourceCleaning import clean_dict
from astropy.io import ascii
from astropy.table import Table
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from collections import OrderedDict
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

#Taken from https://www.eso.org/~ohainaut/ccd/sn.html
#Error on Mag  ~    1/  (S/N)
# So calculating SNR for each band as 1/PSFMagErr
# Estimate the SNR of each band and choose the bands
# with the highest SNR for the rest of our measurements
# host_df - the dataframe of hosts we're considering
def choose_band_SNR(host_df):
    bands = ['g', 'r', 'i', 'z', 'y']
    try:
        gSNR = float(1/host_df["gPSFMagErr"])
        rSNR = float(1/host_df["rPSFMagErr"])
        iSNR = float(1/host_df["iPSFMagErr"])
        zSNR = float(1/host_df["zPSFMagErr"])
        ySNR = float(1/host_df["yPSFMagErr"])

        SNR = np.array([gSNR, rSNR, iSNR, zSNR, ySNR])
        i = np.argmax(SNR)
    except:
        #if we have issues getting the band with the highest SNR, just use 'r'-band
        i = 1
    return bands[i]
#Plot the DLR on each of these, as vectors in direction of SNe (SNe as star)
def calc_DLR(ra_SN, dec_SN, ra_host, dec_host, r_a, r_b, source, best_band):
    # EVERYTHING IS IN ARCSECONDS

    ## taken from "Understanding Type Ia Supernovae Through Their Host Galaxies..." by Gupta
    #https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1916&context=edissertations
    xr = np.abs(ra_SN.deg - float(ra_host))*3600
    yr = np.abs(dec_SN.deg - float(dec_host))*3600

    SNcoord = SkyCoord(ra_SN, dec_SN, frame='icrs')
    hostCoord = SkyCoord(ra_host*u.deg, dec_host*u.deg, frame='icrs')
    sep = SNcoord.separation(hostCoord)
    dist = sep.arcsecond
    badR = 10000000000.0 # if we don't have spatial information, get rid of it #this
    # is good in that it gets rid of lots of artifacts without radius information
    #dist = float(np.sqrt(xr**2 + yr**2))


    XX = best_band + 'momentXX'
    YY = best_band + 'momentYY'
    XY = best_band + 'momentXY'

    if (np.float(source[XX]) != np.float(source[XX])) | (np.float(source[XY]) != np.float(source[XY])) | \
        (np.float(source[YY]) != np.float(source[YY])):
        return dist, badR

    U = np.float(source[XY])
    Q = np.float(source[XX]) - np.float(source[YY])
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

def plot_ellipse(ax, px, s, ra, dec, color):
    i=0
    size = px  #PS cutout image size, 240*sidelength in arcmin
    x0, y0 = ((ra-s['raMean'])*4*3600*np.cos(s['decMean']/180*np.pi)+(size/2)), (s['decMean']-dec)*4*3600+(size/2)
    i=i+1

    y, x = np.mgrid[0:size, 0:size]# 4 pixel for 1 arcsec for PS1, here image size is set to be 20"*20", depend on your cutout image size
    #make fitted image
    n_radius=2
    theta1 = s['phi']#rot angle
    a1= s['r_a']
    b1= s['r_b']
    e1 = mpatches.Ellipse((x0, y0), 4*n_radius*a1, 4*n_radius*b1, theta1, lw=2, ls='--', edgecolor=color,
                          facecolor='none',  label='source 1')
    ax.add_patch(e1)

def plot_DLR_vectors(path, transient, transient_df, host_dict_candidates, host_dict_final, host_df, R_dict, ra_dict, df = "TNS", dual_axes=0, scale=1, postCut=0):
    hostList = host_dict_candidates[str(transient)]
    #os.chdir(path)
    if type(hostList) is np.ndarray:
        if len(hostList) > 1:
            chosen = host_dict_final[transient]
        else:
            chosen = hostList[0]
    else:
        chosen = hostList
        hostList = np.array(hostList)
    band = 'r'
    px = int(240*scale)
    row = transient_df[transient_df['Name'] == transient]

    tempRA = Angle(row.RA, unit=u.hourangle)
    tempDEC = Angle(row.DEC, unit=u.degree)
    transientRA = tempRA.degree[0]
    transientDEC = tempDEC.degree[0]

    try:
        truehostRA  = host_df.loc[host_df['objID'] == chosen, 'raMean'].values[0]
        truehostDEC = host_df.loc[host_df['objID'] == chosen, 'decMean'].values[0]
        searchRA = truehostRA
        searchDEC = truehostDEC
    except:
        searchRA = transientRA
        searchDEC = transientDEC

    a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(searchRA, searchDEC, int(px*0.25), band), '.')
    if not a:
        get_PS1_Pic(searchRA, searchDEC, px, band)
        a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(searchRA, searchDEC, int(px*0.25), band), '.')
    hdu = fits.open(a[0])[0]
    image_data = fits.open(a[0])[0].data
    wcs = WCS(hdu.header)
    figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')
    ax = plt.subplot(projection=wcs)
    ax.annotate(r'%s' % np.array(row.Name)[0], xy=(transientRA-150, transientDEC+25), xytext=(transientRA-150, transientDEC+5), fontsize=20, color='r')
    if (dual_axes):
        overlay = ax.get_coords_overlay('fk5')
        overlay['ra'].set_axislabel('RA', fontsize=16)
        overlay['dec'].set_axislabel('DEC', fontsize=16)
    else:
        ax.set_xlabel("RA", fontsize=16)
        ax.set_ylabel("DEC", fontsize=16)
    # plotting the transient
    ax.scatter(transientRA, transientDEC, transform=ax.get_transform('fk5'), marker='*', lw=4, s=200,
           color="#f3a712",zorder=100)
    for host in hostList:
        hostDF = host_df[host_df['objID'] == host]

        band = choose_band_SNR(hostDF)
        XX = hostDF[band + 'momentXX'].values[0]
        YY = hostDF[band + 'momentYY'].values[0]
        XY = hostDF[band + 'momentXY'].values[0]
        U = np.float(XY)
        Q = np.float(XX) - np.float(YY)
        if (Q == 0):
            r_a = 1.e-5
        else:
            phi = 0.5*np.arctan(U/Q)
        kappa = Q**2 + U**2
        a_over_b = (1 + kappa + 2*np.sqrt(kappa))/(1 - kappa)

        r_a = ra_dict[host]
        r_b = r_a/(a_over_b)

        hostDF['r_a'] = r_a
        hostDF['r_b'] = r_b
        hostDF['phi'] = phi
        hostRA = host_df.loc[host_df['objID'] == host,'raMean'].values[0]
        hostDEC = host_df.loc[host_df['objID'] == host,'decMean'].values[0]

        hostDLR = R_dict[host]
        c = '#666dc9'
        c2 = 'red'
        if (host == chosen):
            c = c2 = '#d308d0'
        plot_ellipse(ax, px, hostDF, searchRA, searchDEC, c)
        #plot_ellipse(ax, px, hostDF, transientRA, transientDEC, c)

        # in arcseconds
        dx = float(hostRA - transientRA)*3600
        dy = float(hostDEC - transientDEC)*3600

        dist = np.sqrt(dx**2 + dy**2)
        if hostDLR == 10000000000.0:
            hostDLR = 0.0
        else:
            hostDLR = dist/hostDLR
        #in arcseconds
        scale_factor = hostDLR/dist

        DLR_RA = float(hostRA) - dx*scale_factor/3600
        DLR_DEC = float(hostDEC) - dy*scale_factor/3600

        pointRA = [hostRA, DLR_RA]
        pointDEC = [hostDEC, DLR_DEC]

        ax.plot(pointRA, pointDEC, transform=ax.get_transform('fk5'), lw=2, color= c)
    ax.imshow(image_data, norm=colors.PowerNorm(gamma = 0.5, vmin=1, vmax=1.e4), cmap='gray_r')
    try:
        z_host = float(host_df[host_df["objID"] == chosen]['NED_redshift'].values[0])
        if(z_host == z_host):
            print(" z_host={}\n".format(z_host))
            #ax.annotate(r"$z_{Host}$ = {%.2f}".format(z_host),xy=(transientRA, transientDEC), xytext=(transientRA, transientDEC), fontsize=300)
            ax.annotate(r'$z_{Host}$ = %.3f' % z_host, xy=(transientRA-280, transientDEC+40), xytext=(transientRA-280, transientDEC+40), fontsize=20, color='k')
            #ax.text(transientRA+0.003, transientDEC+0.003, "$z_{Host}$ = {%.2f}".format(z_host), transform=ax.get_transform('fk5'), fontsize=10)
    except:
        host_redshift = 0
    if postCut:
        plt.savefig("PS1_ra={}_dec={}_{}arcsec_{}_postCut.pdf".format(transientRA, transientDEC, int(px*0.25), band))
    else:
        plt.savefig("PS1_ra={}_dec={}_{}arcsec_{}_DLR.pdf".format(transientRA, transientDEC, int(px*0.25), band))

# if todo == "s", save the dictionary and the list of
# remaining sources
# if todo == "r", return them
def chooseByDLR(path, hosts, transients, fn, orig_dict, dict_mod, todo="s"):
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
    GD_SN = []
    for name, host in orig_dict.items():
        if (type(host) is not np.int64 and type(host) is not float):
             if (len(host.shape) > 0) and (host.shape[0] > 0):
                R_dict = {}
                ra_dict = {}
                dist_dict = {}
                host = np.array(host)
                if len(host)>0:
                    for tempHost in host:
                        noGood = 0
                        theta = 0
                        host_df = hosts[hosts["objID"] == tempHost]
                        transient_df = transients[transients["Name"] == str(name)]
                        band = choose_band_SNR(host_df)
                        dist = 1.e10
                        R = 1.e10
                        if np.float64(host_df[band + 'KronRad']) > 0.0:
                            r_a = r_b = np.float64(host_df[band + 'KronRad'])
                        else:
                            noGood = 1
                        if noGood:
                            # If no good, we want to artificially inflate the distance to the SN so that
                            # we don't incorrectly pick this as our host
                            # (but the radius goes into plotting, so ra is artificially shrunk)
                            R = 1.e10
                            R_dict[tempHost] = R
                            noGood = 0
                            ra_dict[tempHost] = 0.05
                            GDflag = 1
                        else:
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
                            #print(tempHost)
                            #print(name)
                            dist, R = calc_DLR(ra_SN, dec_SN, ra_host, dec_host, r_a, r_b, host_df, band)
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
                    N = min(3, len(R_dict_sub.keys()))
                    R_dict_sub = dict(list(R_dict_sub.items())[:N])
                    if len(R_dict_sub.keys()) > 1:
                        gal_hosts = []
                        for key in R_dict_sub:
                            #print(hosts[hosts['objID'] == key]['NED_type'])
                            tempType = hosts[hosts['objID'] == key]['NED_type'].values[0]
                            if (tempType == "G"):
                                gal_hosts.append(key)
                        if len(gal_hosts) > 0:
                            if gal_hosts[0] != chosenHost and R_dict[gal_hosts[0]] < 1.0:
                                chosenHost = gal_hosts[0] #only change if we're within the light profile of the galaxy
                                print("Choosing the galaxy with the smallest DLR - nearest source had DLR > 1!", file=f)
                    dict_mod[name] = chosenHost
                    hosts.loc[hosts['objID'] == chosenHost, 'dist/DLR'] = R_dict[chosenHost]
                    hosts.loc[hosts['objID'] == chosenHost, 'dist'] = dist_dict[chosenHost]
                    if (GDflag):
                        GDflag = 0
                        print("Issue with DLR. Try Gradient Descent!", file=f)
                        GD_SN.append(name)
                    print(float(hosts[hosts['objID'] == chosenHost]['raMean']), float(hosts[hosts['objID'] == chosenHost]['decMean']), file=f)
                f.flush()
    f.close()
    if todo == "s":
        with open('../dictionaries/DLR_rankOrdered_hosts.p', 'wb') as fp:
            pickle.dump(dict_mod, fp, protocol=pickle.HIGHEST_PROTOCOL)
        hosts.to_csv("../tables/DLR_rankOrdered_hosts.csv")
        return
    elif todo =="r":
        return hosts, dict_mod, noHosts, GD_SN
