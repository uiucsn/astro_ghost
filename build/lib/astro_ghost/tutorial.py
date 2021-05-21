import os
import sys
from astro_ghost.PS1QueryFunctions import getAllPostageStamps
from astro_ghost.TNSQueryFunctions import getTNSSpectra
from astro_ghost.NEDQueryFunctions import getNEDSpectra
from astro_ghost.ghostHelperFunctions import *
from astropy.coordinates import SkyCoord
from astro_ghost.classifier import classify
from astropy import units as u
import pandas as pd
from datetime import datetime

#we want to include print statements so we know what the algorithm is doing
verbose = 1

#download the database from ghost.ncsa.illinois.edu
#note: real=False creates an empty database, which
#allows you to use the association methods without
#needing to download the full database first
getGHOST(real=True, verbose=verbose)

#create a list of the supernova names and their skycoords (these three are from TNS)
snName = ['SN 2012dt', 'SN 1998bn', 'SN 1957B']

snCoord = [SkyCoord(14.162*u.deg, -9.90253*u.deg, frame='icrs'), \
           SkyCoord(187.32867*u.deg, -23.16367*u.deg, frame='icrs'), \
           SkyCoord(186.26125*u.deg, +12.899444*u.deg, frame='icrs')]

# run the association algorithm!
# this first checks the GHOST database for a SN by name, then by coordinates, and
# if we have no match then it manually associates them.
hosts = getTransientHosts(snName, snCoord, verbose=verbose, starcut='normal', ascentMatch=True)

# classify transients
predictions = classify(hosts)

#create directories to store the host spectra, the transient spectra, and the postage stamps
hSpecPath = "./hostSpectra/"
tSpecPath = "./SNspectra/"
psPath = "./hostPostageStamps/"
paths = [hSpecPath, tSpecPath, psPath]
for tempPath in paths:
    if not os.path.exists(tempPath):
        os.makedirs(tempPath)

transients = pd.DataFrame({'Name':snName, 'RA':[x.ra.deg for x in snCoord], 'DEC':[x.dec.deg for x in snCoord]})

#get postage stamps and spectra
getAllPostageStamps(hosts, 120, psPath, verbose) #get postage stamps of hosts
getNEDSpectra(hosts, hSpecPath, verbose) #get spectra of hosts
getTNSSpectra(transients, tSpecPath, verbose) #get spectra of transients (if on TNS)

# Helper functions for querying the database
supernovaCoord = [SkyCoord(344.5011708333333*u.deg, 6.0634388888888875*u.deg, frame='icrs')]
galaxyCoord = [SkyCoord(344.50184181*u.deg, 6.06983149*u.deg, frame='icrs')]
snName = ["PTF10llv"]
table = fullData()

# 1. Get the entry corresponding to a specific transient by its name (or coordinates)
#    note: The coordinate/name is passed as a list, so multiple entries can be
#          queried simultaneously
#    This function returns the matches as a pandas dataframe (df) along with
#    a list of the sources not found (by name or coordinate)
df, notFound = getDBHostFromTransientCoords(supernovaCoord)
df, notFound = getDBHostFromTransientName(snName)

# 2. Print summary statistics about a particular host galaxy system or set of systems from a supernova
getHostStatsFromTransientName(snName)
getHostStatsFromTransientCoords(supernovaCoord)

# 3. Get stats about the supernovae associated with a host galaxy
galaxyName = ['UGC 12266']
getTransientStatsFromHostName(galaxyName)
getTransientStatsFromHostCoords(galaxyCoord)

# 4. get an image of the field by coordinates
tempSize = 400 #size in pixels
band = ['grizy']
getcolorim(galaxyCoord[0].ra.deg, galaxyCoord[0].dec.deg, size=tempSize, filters=band, format="png")

# 5. get an image of the host galaxy system associated with a supernova (by supernova name)
getHostImage(snName, save=0)

# 6. Find all supernova-host galaxy matches within a certain search radius (in arcseconds)
coneSearchPairs(supernovaCoord[0], 1.e3)

#7. Beta: find photometric redshift of host galaxies matches:
from astro_ghost.photoz_helper import calc_photoz
hosts = calc_photoz(hosts)
print(hosts['photo_z'].values)
