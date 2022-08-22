import matplotlib
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ned import Ned
from astroquery.simbad import Simbad
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astro_ghost.PS1QueryFunctions import find_all

def getSimbadInfo(df):
    df.reset_index(inplace=True, drop=True)

    df['hasSimbad'] = 0

    missingCounter = 0

    for index, row in df.iterrows():
        tempRA = row['raMean']
        tempDEC = row['decMean']
        # create a sky coordinate to query NED
        c = SkyCoord(ra=tempRA*u.degree, dec=tempDEC*u.degree, frame='icrs')
        # execute query

        result_table = Simbad.query_region(c, radius=(0.00055555)*u.deg)
        if result_table:
            df.at[index, 'hasSimbad'] = 1
        if missingCounter > 5000:
            print("Locked out of Simbad, will have to try again later...")
            return df
    return df
