from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.ned import Ned
from time import sleep
from datetime import datetime, timezone, timedelta
import re, os
import numpy as np
from astro_ghost.PS1QueryFunctions import find_all

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

NED_TIME_SLEEP = 2


def ned_rate_limited():
    def ned_update_request_time(ned_time_file):
        with open(ned_time_file, 'w') as fp:
            fp.write(datetime.now(timezone.utc).isoformat())
    ned_time_file = '/tmp/ned_last_api_call'
    delay = False
    if os.path.exists(ned_time_file):
        with open(ned_time_file) as fp:
            ned_last_api_call = fp.read()
        if ned_last_api_call:
            last_query = datetime.fromisoformat(ned_last_api_call)
            current_time = datetime.now(timezone.utc)
            delay = current_time - last_query < timedelta(seconds=NED_TIME_SLEEP)
        if not delay or not ned_last_api_call:
            ned_update_request_time(ned_time_file)
    else:
        ned_update_request_time(ned_time_file)
    return delay


def getNEDSpectra(df, path, verbose=False):
    """Downloads NED spectra for the host galaxy, if it exists.

    :param df: Dataframe containing the associated transients and their host galaxies in PS1.
    :type df: Pandas DataFrame
    :param path: Filepath where NED spectra should be saved.
    :type path: str
    :param verbose: Whether to print relevant debugging information.
    :type verbose: boolean
    """

    hostNames = np.array(df.dropna(subset=['NED_name'])['NED_name'])
    transientNames = np.array(df.dropna(subset=['NED_name'])['TransientName'])
    for j in np.arange(len(hostNames)):
        try:
            while ned_rate_limited():
                print(f'Avoiding NED rate limit. Sleeping for {NED_TIME_SLEEP} seconds...')
                sleep(NED_TIME_SLEEP)
            else:
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

    :param df: Dataframe of the PS1 objects for which to query NED.
    :type df: Pandas DataFrame
    :return: The same dataframe as input, with NED info added.
    :rtype: Pandas DataFrame
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
            while ned_rate_limited():
                print(f'Avoiding NED rate limit. Sleeping for {NED_TIME_SLEEP} seconds...')
                sleep(NED_TIME_SLEEP)
            else:
                result_table = Ned.query_region(c, radius=(2/3600)*u.deg, equinox='J2000.0')
                if len(result_table) > 0:
                    missingCounter = 0
        except:
            missingCounter += 1

        if len(result_table) > 0:
            #if Messier or NGC object, take that, otherwise take the closest object
            result_df = result_table.to_pandas()
            if len(result_df) > 1:
                result_NGC = result_df[[x.startswith("NGC") for x in result_df['Object Name']]]
                if len(result_NGC) > 0:
                    result_df = result_NGC.copy()
                else:
                    result_M = result_df[[x.startswith("NGC") for x in result_df['Object Name']]]
                    if len(result_M) > 0:
                        result_df = result_M.copy()
            result_gal = result_df[result_df['Type'] == b'G']
            if len(result_gal) > 0:
                result_df = result_gal.copy()
            #get closest object
            result_df = result_df[result_df['Separation'] == np.min(result_df['Separation'])]

            #remove supernovae, etc
            result_df = result_df[result_df['Type'] != b'SN']

            # Note - sometimes the colon appears in names when a sub-structure of a galaxy. Get rid of it below!
            df.loc[index, 'NED_name'] = result_df['Object Name'].values[0].split(":")[0]
            df.loc[index, 'NED_type'] = result_df['Type'].values[0]
            df.loc[index, 'NED_vel'] = result_df['Velocity'].values[0]
            df.loc[index, 'NED_redshift'] = result_df['Redshift'].values[0]
            df.loc[index, 'NED_redshift_flag'] = result_df['Redshift Flag'].values[0]

        # if the method fails for many in a row, it's likely that too many queries have been made.
        if missingCounter > 5000:
            print("Locked out of NED, will have to try again later...")
            return df
    return df
