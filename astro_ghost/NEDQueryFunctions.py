from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ned import Ned
import re
import numpy as np
from astro_ghost.PS1QueryFunctions import find_all

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def getNEDSpectra(df, path, verbose=False):
    """Downloads NED spectra for the host galaxy, if it exists.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe containing the associated transients and their host galaxies in PS1.
    path : str
        Filepath where NED spectra should be saved.
    verbose : boolean
        Whether to print relevant debugging information.

    """
    hostNames = np.array(df.dropna(subset=['NED_name'])['NED_name'])
    transientNames = np.array(df.dropna(subset=['NED_name'])['TransientName'])
    for j in np.arange(len(hostNames)):
        try:
            spectra = Ned.get_spectra(hostNames[j])
        except:
            continue
        if spectra:
            for i in np.arange(len(spectra)):
                fn = re.sub(r"^\s+", "", hostNames[j])
                a = find_all("%s_%.02i.fits"%(fn, i), path)
                sp = spectra[i]
                if not a:
                    if verbose:
                        print("Saving spectrum %i for %s, host of %s."%(i, hostNames[j], transientNames[j]))
                    try:
                        sp.writeto(path+"/%s_%.02i.fits"%(fn, i), output_verify='ignore')
                    except:
                        if verbose:
                            print("Error in saving host spectrum for %s." % transientNames[j])
                elif verbose:
                    print("host spectrum for %s already downloaded!" % transientNames[j])

def getNEDInfo(df):
    """Gets galaxy information from NED, if it exists.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe of the PS1 objects for which to query NED.

    Returns
    -------
    df : Pandas DataFrame
        The same dataframe as input, with NED info added.

    """
    df.reset_index(inplace=True, drop=True)

    df['NED_name'] = ""
    df['NED_type'] = ""
    df["NED_vel"] = np.nan
    df["NED_redshift"] = np.nan
    df["NED_mag"] = np.nan
    df["NED_redshift_flag"] = ""

    ra = df["raMean"]
    dec = df["decMean"]

    # setup lists for ra and dec in hr format, names of NED-identified object, and
    # separation between host in PS1 and host in NED
    ra_hms = []
    dec_dms = []
    names = []
    sep = []

    missingCounter = 0

    for index, row in df.iterrows():
        tempRA = ra[index]
        tempDEC = dec[index]

        # create a sky coordinate to query NED
        c = SkyCoord(ra=tempRA*u.degree, dec=tempDEC*u.degree, frame='icrs')

        # execute query
        result_table = []
        tempName = ""
        tempType = ""
        tempRed = np.nan
        tempVel = np.nan
        tempMag = np.nan

        try:
            #query NED with a 2'' radius.
            result_table = Ned.query_region(c, radius=(0.00055555)*u.deg, equinox='J2000.0')

            if len(result_table) > 0:
                missingCounter = 0
        except:
            missingCounter += 1
        if len(result_table) > 0:
            result_table = result_table[result_table['Separation'] == np.min(result_table['Separation'])]
            result_table = result_table[result_table['Type'] != b'SN']
            result_table = result_table[result_table['Type'] != b'MCld']
            result_gal = result_table[result_table['Type'] == b'G']
            if len(result_gal) > 0:
                result_table = result_gal
            if len(result_table) > 0:

                # Subset for the objects with listed references and photometry, if more than one option.
                result_table = result_table[result_table['Photometry Points'] == np.nanmax(result_table['Photometry Points'])]
                result_table = result_table[result_table['References'] == np.nanmax(result_table['References'])]

                # NED Info is presented as:
                # No. ObjectName	RA	DEC	Type	Velocity	Redshift	Redshift Flag	Magnitude and Filter	Separation	References	Notes	Photometry Points	Positions	Redshift Points	Diameter Points	Associations
                #Split NED info up - specifically, we want to pull the type, velocity, redshift, mag
                tempNED = str(np.array(result_table)[0]).split(",")
                if len(tempNED) > 2:
                    tempName = tempNED[1].strip().strip("b").strip("'")
                    if len(tempNED) > 20:
                        seps = [float(tempNED[9].strip()), float(tempNED[25].strip())]
                        if np.argmin(seps):
                            tempNED = tempNED[16:]
                    tempType =  tempNED[4].strip().strip("b").strip("''")
                    tempVel = tempNED[5].strip()
                    tempRed = tempNED[6].strip()
                    tempRedFlag = tempNED[7].strip().replace("'", "")
                    tempMag = tempNED[8].strip().strip("b").strip("''").strip(">").strip("<")
                    if tempName:
                        df.loc[index, 'NED_name'] = tempName
                    if tempType:
                        df.loc[index, 'NED_type'] = tempType
                    if tempVel:
                        df.loc[index, 'NED_vel'] = float(tempVel)
                    if tempRed:
                        df.loc[index, 'NED_redshift'] = float(tempRed)
                    if tempMag:
                        tempMag = re.findall(r"[-+]?\d*\.\d+|\d+", tempMag)[0]
                        df.loc[index, 'NED_mag'] = float(tempMag)
                    if tempRedFlag:
                        df.loc[index, 'NED_redshift_flag'] = str(tempRedFlag)
                        
        # if the method fails for many in a row, it's likely that too many queries have been made.
        if missingCounter > 5000:
            print("Locked out of NED, will have to try again later...")
            return df
    return df
