import numpy as np
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import pylab
import os
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import re
from astro_ghost.PS1QueryFunctions import *
from astro_ghost.ghostHelperFunctions import *
import glob

# Get cooordinates associated with NGC 4568
supernovaCoord = SkyCoord(344.5011708333333*u.deg, 6.0634388888888875*u.deg, frame='icrs')
table = fullData()

galaxyCoord = SkyCoord(344.50184181*u.deg, 6.06983149*u.deg, frame='icrs')

getHostFromTransientCoords(supernovaCoord)
getHostFromTransientName("PTF10llv")

getHostStatsFromTransientName("SN2017hpi")
getHostStatsFromTransientCoords(supernovaCoord)

getTransientStatsFromHostName('UGC 12266')
getTransientStatsFromHostCoords(galaxyCoord)

getcolorim(ra, dec, size=tempSize, filters=band, format="png")
getHostImage('PTF10llv', save=0)

path = "/home/alexgagliano/Documents/Research/Transient_ML/tables/SNspectra/"
hostPath = '/home/alexgagliano/Documents/Research/Transient_ML_Box/hostSpectra/'

getTransientSpectra('SN2019ur', path)
getHostSpectra('2019dfi', hostPath)

coneSearchPairs(supernovaCoord, 1.e5)
