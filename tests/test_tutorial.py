import os
import pytest
import sys
from astro_ghost.PS1QueryFunctions import *
from astro_ghost.TNSQueryFunctions import *
from astro_ghost.NEDQueryFunctions import *
from astro_ghost.ghostHelperFunctions import *
from astro_ghost.starSeparation import *
from astro_ghost.stellarLocus import *
from astro_ghost.photoz_helper import calc_photoz
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from datetime import datetime


#we want to include print statements so we know what the algorithm is doing
verbose = 1

def test_getGHOST():
    #Download the GHOST database.
    #note: real=False creates an empty database, which
    #allows you to use the association methods without
    #needing to download the full database first

    getGHOST(real=True, verbose=verbose, clobber=True)
    #test that it got the ghost database
    df = fullData()
    # GHOST has the correct match for at least NGC 2997
    assert df.loc[df['TransientName'] == 'SN2003jg', 'NED_name'].values[0] == 'NGC 2997'

def test_NED():
    # test our ability to snag a galaxy name from NED, using the coordinates of NGC 4321
    df = pd.DataFrame({'objID':23412341234, 'raMean':[185.7288750], 'decMean':[15.82230]})
    df = getNEDInfo(df)
    assert df['NED_name'].values[0] == 'NGC 4321'

def test_associate():
    #create a list of the supernova names, their skycoords, and their classes (these three are from TNS)
    transientName = ['SN 2012dt', 'SN 1998bn', 'SN 1957B']

    transientCoord = [SkyCoord(14.162*u.deg, -9.90253*u.deg, frame='icrs'), \
            SkyCoord(187.32867*u.deg, -23.16367*u.deg, frame='icrs'), \
            SkyCoord(186.26125*u.deg, +12.899444*u.deg, frame='icrs')]

    transientClass = ['SN IIP', 'SN', 'SN Ia']

    # run the association algorithm with the DLR method!
    hosts = getTransientHosts(transientName, transientCoord, transientClass, verbose=verbose, starcut='gentle', ascentMatch=False)

    correctHosts = [SkyCoord(14.1777425*u.deg, -9.9138756*u.deg, frame='icrs'),
                    SkyCoord(187.3380517*u.deg, -23.1666716*u.deg, frame='icrs'),
                    SkyCoord(186.2655971*u.deg, 12.8869831*u.deg, frame='icrs')]

    sep = []
    for i in np.arange(len(correctHosts)):
       c1 = correctHosts[i]
       c2 = SkyCoord(hosts['TransientRA'].values[i]*u.deg, hosts['TransientDEC'].values[i]*u.deg, frame='icrs')
       sep.append(c2.separation(c2).arcsec)

    #consider a success if the three hosts were found to a 1'' precision
    assert np.nanmax(sep) < 1


def test_starSeparation():
    #classify a few galaxies, classify a few stars
    sourceType = ['galaxy',  'galaxy', 'star', 'star']
    ra = [186.7154417, 31.0672500, 191.0597417, 17.8279833]
    dec = [9.1342306, 20.8474806, 12.3629917, 33.2604472]
    sourceSet = []
    for i in np.arange(len(ra)):
        a = ps1cone(ra[i], dec[i], 1/3600)
        a = ascii.read(a)
        sourceSet.append(a.to_pandas().iloc[[0]])
    sourceDF = pd.concat(sourceSet, ignore_index=True)
    sourceDF['trueSourceClass'] = sourceType
    sourceDF = getColors(sourceDF)
    sourceDF = calc_7DCD(sourceDF)
    sourceDF = getNEDInfo(sourceDF)
    gals, stars = separateStars_STRM(sourceDF)

    #True is stars, False is gals. Passes if all gals are False and all stars are True!
    assert (np.nansum(gals['sourceClass'] == False) + np.nansum(stars['sourceClass'])) == len(sourceDF)

def test_plotLocus():
    #plot a subset of GHOST data compared to the tonry stellar locus
    GHOST = fullData()
    plotLocus(GHOST.iloc[0:500], color=False, save=True, type="Gals", timestamp="")
    assert len(glob.glob(os.path.join(os.getcwd(), 'PS1_Gals_StellarLocus_.pdf'))) > 0

def test_skymapper_associate():
    transientName = ['AT2023hri']
    transientCoord = [SkyCoord(288.5964917*u.deg,  -54.5636583*u.deg, frame='icrs')]
    transientClass = ['AT']

    # run the association algorithm with the DLR method!
    hosts = getTransientHosts(transientName, transientCoord, transientClass, verbose=verbose, starcut='gentle', ascentMatch=False)
    assert hosts['NED_name'].values[0] == 'ESO 184- G 042'

def test_resolve():
    coords = resolve('M100')
    c1 = SkyCoord(coords[0]*u.deg, coords[1]*u.deg, frame='icrs')
    c2 = SkyCoord(185.7288750*u.deg, 15.82230*u.deg, frame='icrs')
    assert c2.separation(c2).arcsec < 0.1
    
def test_photoz():
    DF_test = pd.DataFrame({'raMean':[258.026087],'decMean':[40.344349,],'objID':[156412580262833857],})
    DF_test = calc_photoz(DF_test)
    assert np.abs(DF_test['photo_z'].iloc[0] - 0.260683) < 0.01
