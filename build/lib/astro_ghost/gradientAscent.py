import os
from astro_ghost.PS1QueryFunctions import find_all, get_PS1_Pic, get_PS1_type, get_PS1_mask, query_ps1_noname
from astro_ghost.NEDQueryFunctions import getNEDInfo
from datetime import datetime
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
import pickle
from astropy.io import ascii
from collections import Counter
import scipy
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from matplotlib.colors import LogNorm
from astropy.utils.data import get_pkg_data_filename
#from astro_ghost import DLR as dlr
from photutils import Background2D
import numpy.ma as ma
from astropy.io import fits
import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.exceptions import AstropyWarning
from matplotlib import colors
from scipy import interpolate
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from astropy.visualization import SqrtStretch
from photutils import DAOStarFinder
from photutils import MedianBackground, MeanBackground
from astropy.stats import SigmaClip

############# functions ####################################
def updateStep(px, gradx, grady, step, point, size):
    max_x = px
    max_y =  px
    grad = np.array([gradx[point[0], point[1]], grady[point[0], point[1]]])
    #make sure we move at least one unit in grid spacing - so the grad must have len 1
#    if grad[0] + grad[1] > 0:
    ds = step/np.sqrt(grad[0]**2 + grad[1]**2)
    ds = np.nanmin([ds, step])
#    else:
#        ds = step

    newPoint = [point[0] + ds*grad[0], point[1] + ds*grad[1]]
    newPoint = [int(newPoint[0]), int(newPoint[1])] #round to nearest index
    if (newPoint[0] >= max_x) or (newPoint[1] >= max_y) or (newPoint[0] < 0) or (newPoint[1] < 0):
        #if we're going to go out of bounds, don't move
        return point
    elif ((newPoint == point) and (size == 'large')): #if we're stuck, perturb one pixel in a random direction:
        a = np.random.choice([-1, 0, 1], 2)#
        newPoint = [newPoint[0] + a[0], newPoint[1] + a[1]]
    return newPoint


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def plot_DLR_vectors_GD(size, path, transient, transient_df, host_dict_candidates, host_dict_final, host_df, R_dict, ra_dict, df = "TNS", dual_axes=0, scale=1, postCut=0):
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
    px = int(size*scale)
    row = transient_df[transient_df['Name'] == transient]

    tempRA = Angle(row.RA, unit=u.degree)
    tempDEC = Angle(row.DEC, unit=u.degree)
    transientRA = tempRA.degree[0]
    transientDEC = tempDEC.degree[0]
    print(transientRA)
    print(transientDEC)
    searchRA = transientRA
    searchDEC = transientDEC

    a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(searchRA, searchDEC, int(px*0.25), band), '.')
    if not a:
        get_PS1_Pic(searchRA, searchDEC, px, band)
        a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(searchRA, searchDEC, int(px*0.25), band), '.')
    #a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(transientRA, transientDEC, int(px*0.25), band), '.')
    #if not a:
    #    get_PS1_Pic(transientRA, transientDEC, px, band)
    #    a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(transientRA, transientDEC, int(px*0.25), band), '.')
    hdu = fits.open(a[0])[0]
    image_file = get_pkg_data_filename(a[0])
    image_data = fits.getdata(image_file, ext=0)
    wcs = WCS(hdu.header)
    fig = figure(num=None, figsize=(20,20), facecolor='w', edgecolor='k')
    #ax = plt.subplot(projection=wcs)
    fig.add_axes(projection=wcs)
    axes_coords = [0, 0, 1, 1] # plotting full width and height
    ax = fig.add_axes(axes_coords, projection=wcs)
    axes_coords2 = [-0.045, -0.03, 1.06, 1.08]
    ax_grads = fig.add_axes(axes_coords2, projection=None)
    plt.axis('off')
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
        hostDF['raMean'], hostDF['decMean']
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

        ax.plot(pointRA, pointDEC, transform=ax.get_transform('fk5'), lw=6, color= c)
#    ax.imshow(image_data, norm=colors.LogNorm(), cmap='gray_r')
    ax.imshow(image_data, norm=colors.PowerNorm(gamma = 0.5, vmin=1, vmax=1.e4), cmap='gray')
    plt.axis('off')
    return ax_grads


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
    e1 = mpatches.Ellipse((x0, y0), 4*n_radius*a1, 4*n_radius*b1, theta1, lw=6, ls='--', edgecolor=color,
                          facecolor='none',  label='source 1')
    ax.add_patch(e1)


def denoise(img, weight=0.1, eps=1e-3, num_iter_max=200):
    """Perform total-variation denoising on a grayscale image.

    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more de-noising (at
        the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : array
        De-noised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """
    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)

    nm = np.prod(img.shape[:2])
    tau = 0.125

    i = 0
    while i < num_iter_max:
        u_old = u

        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy
        norm_new = np.maximum(1, np.sqrt(px_new **2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # update image
        u = img + weight * div_p

        # calculate error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)

        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error

        # don't forget to update iterator
        i += 1

    return u

def get_clean_img(ra, dec, px, band):
    #first, mask the data
    a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(ra, dec, int(px*0.25), band), '.')
    if not a:
        get_PS1_Pic(0, ra, dec, px, band)
        a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(ra, dec, int(px*0.25), band), '.')
    b = find_all("PS1_ra={}_dec={}_{}arcsec_{}_mask.fits".format(ra, dec, int(px*0.25), band), '.')
    if not b:
        get_PS1_mask(ra, dec, px, band)
        b = find_all("PS1_ra={}_dec={}_{}arcsec_{}_mask.fits".format(ra, dec, int(px*0.25), band), '.')
    c = find_all("PS1_ra={}_dec={}_{}arcsec_{}_stack.num.fits".format(ra, dec, int(px*0.25), band), '.')
    if not c:
        get_PS1_type(ra, dec, px, band, 'stack.num')
        c = find_all("PS1_ra={}_dec={}_{}arcsec_{}_stack.num.fits".format(ra, dec, int(px*0.25), band), '.')
    #d = find_all("PS1_ra={}_dec={}_{}arcsec_{}_wt.fits".format(ra, dec, int(px*0.25), band), '.')
    #if not d:
    #    get_PS1_wt(ra, dec, px, band)
    #    d = find_all("PS1_ra={}_dec={}_{}arcsec_{}_wt.fits".format(ra, dec, int(px*0.25), band), '.')
    image_data_mask = fits.open(b[0])[0].data
    image_data_num = fits.open(c[0])[0].data
    #image_data_wt = fits.open(d[0])[0].data
    image_data = fits.open(a[0])[0].data

    hdu = fits.open(a[0])[0]
    wcs = WCS(hdu.header)

    bit = image_data_mask
    mask = image_data_mask
    for i in np.arange(np.shape(bit)[0]):
        for j in np.arange(np.shape(bit)[1]):
            if image_data_mask[i][j] == image_data_mask[i][j]:
                bit[i][j] = "{0:016b}".format(int(image_data_mask[i][j]))
                tempBit = str(bit[i][j])[:-2]
                if len(str(int(bit[i][j]))) > 12:
                    if (tempBit[-6] == 1) or (tempBit[-13] == 1):
                        mask[i][j] = np.nan
                elif len(str(int(bit[i][j]))) > 5:
                    if (tempBit[-6] == 1):
                        mask[i][j] = np.nan

    mask = ~np.isnan(image_data_mask)
    mask_num = image_data_num
    #weighted
    #image_data *= image_data_wt
    image_masked = ma.masked_array(image_data, mask=mask)
    image_masked_num = ma.masked_array(image_masked, mask=mask_num)

    #edited to PASS BACK THE MASKED ARRAY!!
    #then return the data
    return np.array(image_masked_num), wcs, hdu

def getSteps(SN_dict, SN_names, hostDF):
    steps = []
    hostDF.replace(-999, np.nan, inplace=True)
    hostDF.replace(-999, np.nan, inplace=True)
    for name in SN_names:
        hostList = SN_dict[name]
        if (type(hostList) is np.int64 or type(hostList) is float):
            hostList = [hostList]
        checkNan = [x == x for x in hostList]
        if np.sum(checkNan) > 0:
            hostRadii = hostDF.loc[hostDF['objID'].isin(hostList), 'rKronRad'].values
            mean = np.nanmean(hostRadii)
            if mean == mean:
#                print(mean)
#                mean /= 2
                mean = np.max([mean,2])
                step = np.min([mean, 50])
                steps.append(step) #find some proper scaling factor between the mean and the step size
            else:
                steps.append(5)
        else:
            steps.append(5)
    return steps

############# end functions ####################################

def gradientAscent(path, SN_dict, SN_dict_postDLR, SN_names, hostDF, transientDF, fn, plot=1):
    #os.chdir(path)
    warnings.filterwarnings('ignore', category=AstropyUserWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning)
    #warnings.filterwarnings("ignore", category=AstropyUserWarning)
    #debugging purposes
    step_sizes = getSteps(SN_dict, SN_names, hostDF)
    unchanged = []
    #r = 0 #counter for occasionally saving to file
    N_associated = 0
    f = open(fn, 'w')
    print("Starting size of data frame: %i" % len(hostDF), file=f)
    try:
        os.makedirs('quiverMaps')
    except:
        print("Already have the folder quiverMaps!")
    for i in np.arange(len(step_sizes)):
        try:
    #    if True:
            transient_name = SN_names[i]
            print("Transient: %s"% transient_name, file=f)

            ra = transientDF.loc[transientDF['Name'] == transient_name, 'RA'].values[0]
            dec = transientDF.loc[transientDF['Name'] == transient_name, 'DEC'].values[0]
            px = 800
            g_img, wcs, g_hdu  = get_clean_img(ra, dec, px, 'g')
            g_mask = np.ma.masked_invalid(g_img).mask
            r_img, wcs, r_hdu  = get_clean_img(ra, dec, px, 'r')
            r_mask = np.ma.masked_invalid(r_img).mask
            i_img, wcs, i_hdu  = get_clean_img(ra, dec, px, 'i')
            i_mask = np.ma.masked_invalid(i_img).mask

            #cleanup - remove the fits files when we're done using them
            for band in ['g', 'r', 'i']:
                os.remove("PS1_ra={}_dec={}_{}arcsec_{}_stack.num.fits".format(ra, dec, int(px*0.25), band))
                os.remove("PS1_ra={}_dec={}_{}arcsec_{}_mask.fits".format(ra, dec, int(px*0.25), band))
                os.remove("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(ra, dec, int(px*0.25), band))

            #os.chdir(path)
            #    if e.errno != errno.EEXIST:
            #        raise
            #os.chdir("./quiverMaps")
            nancount = 0
            obj_interp = []
            for obj in [g_img, r_img, i_img]:
                data = obj
                mean, median, std = sigma_clipped_stats(data, sigma=20.0)
                daofind = DAOStarFinder(fwhm=3.0, threshold=20.*std)
                sources = daofind(data - median)
                try:
                    xvals = np.array(sources['xcentroid'])
                    yvals = np.array(sources['ycentroid'])
#                    for col in sources.colnames:
#                        sources[col].info.format = '%.8g'  # for consistent table output
                    for k in np.arange(len(xvals)):
                        tempx = xvals[k]
                        tempy = yvals[k]
                        yleft = np.max([int(tempy) - 7, 0])
                        yright = np.min([int(tempy) + 7, np.shape(data)[1]-1])
                        xleft = np.max([int(tempx) - 7, 0])
                        xright = np.min([int(tempx) + 7, np.shape(data)[1]-1])

                        for r in np.arange(yleft,yright+1):
                            for j in np.arange(xleft, xright+1):
                                if dist([xvals[k], yvals[k]], [j, r]) < 5:
                                    data[r, j] = np.nan
                    nancount += np.sum(np.isnan(data))
                    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                    apertures = CircularAperture(positions, r=5.)
                    norm = ImageNormalize(stretch=SqrtStretch())

                    if plot:
                        fig = plt.figure(figsize=(10,10))
                        ax = fig.gca()
                        ax.imshow(data)
                        apertures.plot(color='blue', lw=1.5, alpha=0.5)
                        plt.axis('off')
                        plt.savefig("quiverMaps/detectedStars_%s.png"%transient_name, bbox_inches='tight')
                        plt.close()
                except:
                    print("No stars here!", file=f)

                backx = np.arange(0,data.shape[1])
                backy = np.arange(0, data.shape[0])
                backxx, backyy = np.meshgrid(backx, backy)
                #mask invalid values
                array = np.ma.masked_invalid(data)
                x1 = backxx[~array.mask]
                y1 = backyy[~array.mask]
                newarr = array[~array.mask]
                data = interpolate.griddata((x1, y1), newarr.ravel(), (backxx, backyy), method='cubic')
                obj_interp.append(data)

            #gvar = np.var(obj_interp[0])
            #gmean = np.nanmedian(obj_interp[0])
            gMax = np.nanmax(obj_interp[0])

            g_ZP = g_hdu.header['ZPT_0001']
            r_ZP = r_hdu.header['ZPT_0001']
            i_ZP = i_hdu.header['ZPT_0001']

            #combining into a mean img -
            # m = -2.5*log10(F) + ZP

            gmag = -2.5*np.log10(obj_interp[0]) + g_ZP
            rmag = -2.5*np.log10(obj_interp[1]) + r_ZP
            imag = -2.5*np.log10(obj_interp[2]) + i_ZP

            #now the mean can be taken
            mean_zp = (g_ZP + r_ZP + i_ZP)/3
            meanMag = (gmag + rmag + imag)/3
            meanImg = 10**((mean_zp-meanMag)/2.5) #convert back to flux

            #meanImg = (obj_interp[0] + obj_interp[0] + obj_interp[0])/3
            print("NanCount = %i"%nancount,file=f)
            #mean_center = np.nanmean([g_img[int(px/2),int(px/2)], i_img[int(px/2),int(px/2)], i_img[int(px/2),int(px/2)]])
            #if mean_center != mean_center:
            #    mean_center = 1.e-30
            mean_center = meanImg[int(px/2),int(px/2)]
            print("Mean_center = %f" % mean_center,file=f)
            #mean, median, std = sigma_clipped_stats(meanImg, sigma=10.0)
            meanImg[meanImg != meanImg] = 1.e-30
            mean, median, std = sigma_clipped_stats(meanImg, sigma=10.0)
            print("mean image = %e"% mean, file=f)
            aboveCount = np.sum(meanImg > 1.)
            aboveCount2 = np.sum(meanImg[int(px/2)-100:int(px/2)+100, int(px/2)-100:int(px/2)+100] > 1.)
            aboveFrac2= aboveCount2/40000
            print("aboveCount = %f"% aboveCount,file=f)
            print("aboveCount2 = %f "% aboveCount2, file=f)
            totalPx = px**2
            aboveFrac = aboveCount/totalPx
            print("aboveFrac= %f" % aboveFrac, file=f)
            print("aboveFrac2 = %f "% aboveFrac2, file=f)
            #meanImg[meanImg < 1.e-5] = 0
            if ((median <15) and (np.round(aboveFrac2, 2) < 0.70)) or ((mean_center > 1.e3) and (np.round(aboveFrac,2) < 0.60) and (np.round(aboveFrac2,2) < 0.75)):
                bs = 15
                fs = 1
                if aboveFrac2 < 0.7:
                    step_sizes[int(i)] = 2.
                else:
                    step_sizes[int(i)] = 10.
                print("Small filter", file=f)
                size = 'small'
            elif ((mean_center > 40) and (median > 500) and (aboveFrac > 0.60)) or ((mean_center > 300) and (aboveFrac2 > 0.7)):
                bs = 75 #the big sources
                fs = 3
                print("Large filter", file=f)
                step_sizes[int(i)] = np.max([step_sizes[int(i)], 50])
                size = 'large'
                #if step_sizes[int(i)] == 5:
                #    step_sizes[int(i)] *= 5
                #    step_sizes[int(i)] = np.min([step_sizes[int(i)], 50])
                #if mean_center < 200: #far from the center with a large host
                #    fs = 5
                #elif mean_center < 5000:
                #    step_sizes[int(i)] = np.max([step_sizes[int(i)], 50])
                #size = 'large'
            else:
                bs = 40 #everything in between
                fs = 3
                print("Medium filter", file=f)
                #if step_sizes[int(i)] == 5:
                #    step_sizes[int(i)] *= 3
        #        step_sizes[int(i)] = np.max([step_sizes[int(i)], 25])
                step_sizes[int(i)] = np.max([step_sizes[int(i)], 15])
                size = 'medium'
            #    step_sizes[int(i)] *= 3

            #if (median)
            sigma_clip = SigmaClip(sigma=15.)
            bkg_estimator = MeanBackground()
            #bkg_estimator = BiweightLocationBackground()
            bkg3_g = Background2D(g_img, box_size=bs, filter_size=fs,
             sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            bkg3_r = Background2D(r_img, box_size=bs, filter_size=fs,
             sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            bkg3_i = Background2D(i_img, box_size=bs, filter_size=fs,
             sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

            #pretend the background is in counts too (I think it is, right?) and average in mags
            bkg3_g.background[bkg3_g.background < 0] = 1.e-30
            bkg3_r.background[bkg3_r.background < 0] = 1.e-30
            bkg3_i.background[bkg3_i.background < 0] = 1.e-30

            backmag_g = -2.5*np.log10(bkg3_g.background) + g_ZP
            backmag_r = -2.5*np.log10(bkg3_r.background) + r_ZP
            backmag_i = -2.5*np.log10(bkg3_i.background) + i_ZP

            mean_zp = (g_ZP + r_ZP + i_ZP)/3.
            backmag = 0.333*backmag_g + 0.333*backmag_r + 0.333*backmag_i
            background = 10**(mean_zp-backmag/2.5)

            if plot:
                fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(20,10))
                axs[0].imshow(bkg3_g.background)
                axs[0].axis('off')
                axs[1].imshow(bkg3_r.background)
                axs[1].axis('off')
                axs[2].imshow(bkg3_i.background)
                axs[2].axis('off')
                plt.savefig("quiverMaps/backgrounds_%s.png" % transient_name, bbox_inches='tight')
                plt.close()

            mean, median, std = sigma_clipped_stats(meanImg, sigma=1.0)
            meanImg[meanImg <= (mean)] = 1.e-30
            meanImg[meanImg < 0] = 1.e-30

            if plot:
                fig = plt.figure(figsize=(10,10))
                ax = fig.gca()
                ax.imshow((meanImg)/np.nanmax(meanImg))
                plt.axis('off')
                plt.savefig("quiverMaps/normalizedMeanImage_%s.png" % transient_name, bbox_inches='tight')
                plt.close()

                fig = plt.figure(figsize=(10,10))
                ax = fig.gca()
                ax.imshow(background/np.nanmax(background))
                plt.axis('off')
                plt.savefig("quiverMaps/normalizedMeanBackground_%s.png" % transient_name, bbox_inches='tight')
                plt.close()

            if nancount > 1.e5:
                imgWeight = 0
            elif (mean_center > 1.e4): #and (size is not 'large'):
                imgWeight = 0.75
            elif size == 'medium':
                imgWeight = 0.33
            else:
                imgWeight = 0.10
            print("imgWeight= %f"%imgWeight, file=f)
            fullbackground = ((1-imgWeight)*background/np.nanmax(background) + imgWeight*meanImg/np.nanmax(meanImg))*np.nanmax(background)
#            background  = (0.66*background/np.max(background) +  imgWeight*meanImg/np.nanmax(meanImg))*np.max(background)

            n = px
            X, Y = np.mgrid[0:n, 0:n]
            dx, dy = np.gradient(fullbackground.T)

            n_plot = 10

            dx_small = dx[::n_plot, ::n_plot]
            dy_small = dy[::n_plot, ::n_plot]
            print("step = %f"%  step_sizes[int(i)], file=f)

            start = [[int(px/2),int(px/2)]] #the center of the grid

            if True:
            #if background[int(px/2),int(px/2)] > 0: #if we have some background flux (greater than 3 stdevs away from the median background), follow the gradient
                start.append(updateStep(px, dx, dy, step_sizes[int(i)], start[-1], size))
                for j in np.arange(1.e3):
                    start.append(updateStep(px, dx, dy, step_sizes[int(i)], start[-1], size))
                it_array = np.array(start)
                endPoint = start[-1]

                if plot:
                    fig  = plt.figure(figsize=(10,10))
                    ax = fig.gca()
                    ax.imshow(fullbackground)
                    plt.axis("off")
                    plt.savefig("quiverMaps/fullBackground_%s.png"%transient_name, bbox_inches='tight')
                    plt.close()

                coords = wcs.wcs_pix2world(endPoint[0], endPoint[1], 0., ra_dec_order = True) # Note the third argument, set to 0, which indicates whether the pixel coordinates should be treated as starting from (1, 1) (as FITS files do) or from (0, 0)
                print("Final ra, dec after GD : %f %f"% (coords[0], coords[1]), file=f)
                col = '#D34E24'
                col2 = '#B54A24'
                #lookup by ra, dec
                try:
                    if size == 'large':
                        a = query_ps1_noname(float(coords[0]), float(coords[1]), 20)
                    else:
                        a = query_ps1_noname(float(coords[0]), float(coords[1]), 5)
                except TypeError:
                     continue
                if a:
                    print("Found a host here!", file=f)
                    a = ascii.read(a)
                    a = a.to_pandas()

                    a = a[a['nDetections'] > 1]
                    #a = a[a['ng'] > 1]
                    #a = a[a['primaryDetection'] == 1]
                    smallType = ['AbLS', 'EmLS' , 'EmObj', 'G', 'GammaS', 'GClstr', 'GGroup', 'GPair', 'GTrpl', 'G_Lens', 'IrS', 'PofG', 'RadioS', 'UvES', 'UvS', 'XrayS', '', 'QSO', 'QGroup', 'Q_Lens']
                    medType = ['G', 'IrS', 'PofG', 'RadioS', 'GPair', 'GGroup', 'GClstr', 'EmLS', 'RadioS', 'UvS', 'UvES', '']
                    largeType = ['G', 'PofG', 'GPair', 'GGroup', 'GClstr']
                    if len(a) > 0:
                        a = getNEDInfo(a)
                        if (size == 'large'):# and (np.nanmax(a['rKronRad'].values) > 5)):
                        #    print("L: picking the largest >5 kronRad host within 10 arcsec", file=f)
                            print("L: picking the closest NED galaxy within 20 arcsec", file=f)
                            #a = a[a['rKronRad'] == np.nanmax(a['rKronRad'].values)]
                            tempA = a[a['NED_type'].isin(largeType)]
                            if len(tempA) > 0:
                                a = tempA
                            tempA = a[a['NED_type'] == 'G']
                            if len(tempA) > 0:
                                a = tempA
                            #tempA = a[a['NED_mag'] == np.nanmin(a['NED_mag'])]
                            #if len(tempA) > 0:
                            #    a = tempA
                            if len(a) > 1:
                                a = a.iloc[[0]]
                        elif (size == 'medium'):
                            #print("M: Picking the largest host within 5 arcsec", file=f)
                            print("M: Picking the closest NED galaxy within 5 arcsec", file=f)
                            #a = a[a['rKronRad'] == np.nanmax(a['rKronRad'].values)]
                            tempA = a[a['NED_type'].isin(medType)]
                            if len(tempA) > 0:
                                a = tempA
                            if len(a) > 1:
                                a = a.iloc[[0]]
                        else:
                            tempA = a[a['NED_type'].isin(smallType)]
                            if len(tempA) > 0:
                                a = tempA
                            a = a.iloc[[0]]
                            print("S: Picking the closest non-stellar source within 5 arcsec", file=f)
                        #else:
                        #    f.flush()
                        #    continue
                        #threshold = [1, 1, 0, 0, 0, 0]
                        #flag = ['nDetections', 'nr', 'rPlateScale', 'primaryDetection', 'rKronRad', 'rKronFlux']
                        #j = 0
                        #while len(a) > 1:
                        #    if np.sum(a[flag[int(j)]] > threshold[int(j)]) > 0:
                        #        tempA = a[a[flag[int(j)]] > threshold[int(j)]]
                        #        j += 1
                        #        a = tempA
                        #        if (j == 6):
                        #            break
                        #    else:
                        #        break
                        #if len(a) > 1:
                        #    if len(~a['rKronRad'].isnull()) > 0:
                        #        a = a[a['rKronRad'] == np.nanmax(a['rKronRad'].values)]
                        #    else:
                        #        a = a.iloc[0]
                        print("Nice! Host association chosen.", file=f)
                        print("NED type: %s" % a['NED_type'].values[0], file=f)
                        print(a['objID'].values[0], file=f)
                        print("Chosen Host RA and DEC: %f %f"% (a['raMean'], a['decMean']), file=f)
                        SN_dict_postDLR[transient_name] = a['objID'].values[0]
                        print("Dict value: %i"%SN_dict_postDLR[transient_name],file=f)
                        N = len(hostDF)
                        hostDF = pd.concat([hostDF, a], ignore_index=True)
                        N2 = len(hostDF)
                        if N2 != (N+1):
                            print("ERROR! Value not concatenated!!", file=f)
                            return
                        finalRA = np.array(a['raMean'])
                        finalDEC = np.array(a['decMean'])
                        col = 'tab:green'
                        col2 = '#078840'
                    else:
                        unchanged.append(transient_name)
                else:
                    unchanged.append(transient_name)
                if plot:
                    fig = plt.figure(figsize=(20,20))
                    ax = fig.gca()
                    ax.imshow(i_img, norm=colors.PowerNorm(gamma = 0.5, vmin=1, vmax=1.e4), cmap='gray')#, cmap='gray', norm=LogNorm())
                    it_array = np.array(start)
                    ax.plot(it_array.T[0], it_array.T[1], "--", lw=5, c=col, zorder=20)
                    ax.scatter([int(px/2)], [int(px/2)], marker='*', s=1000, color='#f3a712', zorder=50)
                    ax.scatter(endPoint[0], endPoint[1], marker='*', lw=4, s=1000, facecolor='#f3a712', edgecolor=col2, zorder=200)
                    ax.quiver(X[::n_plot,::n_plot], Y[::n_plot,::n_plot], dx[::n_plot,::n_plot], dy[::n_plot,::n_plot], color='#845C9B', angles='xy', scale_units = 'xy')
                    plt.axis('off')
                    plt.savefig("quiverMaps/quiverMap_%s.png"%transient_name, bbox_inches='tight')
                    plt.close()
                N_associated += 1
            f.flush()
            if N_associated%10 == 0:
                print("N_associated = %i"%N_associated,file=f)
                print("Size of table = %i"%len(hostDF),file=f)
                with open(path+"/dictionaries/gals_postGD.p", 'wb') as fp:
                    pickle.dump(SN_dict_postDLR, fp, protocol=pickle.HIGHEST_PROTOCOL)
                hostDF.to_csv(path+"/hostDF_postGD.tar.gz", index=False)
        except:
             continue

    with open(path+"/dictionaries/gals_postGD.p", 'wb') as fp:
        pickle.dump(SN_dict_postDLR, fp, protocol=pickle.HIGHEST_PROTOCOL)
    hostDF.to_csv(path+"/hostDF_postGD.tar.gz", index=False)
    f.close()
    return  SN_dict_postDLR, hostDF, unchanged
