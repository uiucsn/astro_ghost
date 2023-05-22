import os
import pytest
import sys
from astro_ghost.PS1QueryFunctions import *
from astro_ghost.TNSQueryFunctions import getTNSSpectra
from astro_ghost.NEDQueryFunctions import *
from astro_ghost.ghostHelperFunctions import *
from astro_ghost.stellarLocus import *
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from datetime import datetime
import astro_ghost

#we want to include print statements so we know what the algorithm is doing
verbose = 1


#need to generate tests for photoz_helper and gradientAscent!!

t = np.linspace(0, 70)
y = 7.*2.71828**(-(t)/50)/(1+2.71828**(-(t)/11))
plt.plot(t, y)
