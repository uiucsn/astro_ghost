import numpy as np
from PIL import Image
import os
import pandas as pd
import sys
import re
import json
import mastcasjobs
import requests
from datetime import datetime
try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
    import http.client as httplib
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    import httplib
from os import listdir
from os.path import join
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits, ascii
from astropy.table import Table
import pickle
from io import BytesIO
from astropy.coordinates import SkyCoord
from warnings import simplefilter

# could absolutely be more efficient, but filter out warnings for now
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# set a few environmental variables that we'll need to query PS2 for more accurate galaxy sizes.
if "CASJOBS_USERID" not in os.environ:
    os.environ['CASJOBS_USERID'] = 'ghostbot'
    os.environ['CASJOBS_PW'] = 'ghostbot'

def create_dummy_df(nRows=1, fullDF=False):
    """Creates a dummy PS1 dataframe (useful for running scripts when no sources found).

    :param fullDF: If True, the dummy DataFrame replaces the full GHOST database.
    :dtype fullDF: Boolean
    :return: Empty Pandas DataFrame with PS1 columns.
    :rtype: Pandas DataFrame
    """
    colnames = ['objName', 'objAltName1', 'objAltName2', 'objAltName3', 'objID',
        'uniquePspsOBid', 'ippObjID', 'surveyID', 'htmID', 'zoneID',
        'tessID', 'projectionID', 'skyCellID', 'randomID', 'batchID',
        'dvoRegionID', 'processingVersion', 'objInfoFlag', 'qualityFlag',
        'raStack', 'decStack', 'raStackErr', 'decStackErr', 'raMean',
        'decMean', 'raMeanErr', 'decMeanErr', 'epochMean', 'posMeanChisq',
        'cx', 'cy', 'cz', 'lambda', 'beta', 'l', 'b', 'nStackObjectRows',
        'nStackDetections', 'nDetections', 'ng', 'nr', 'ni', 'nz', 'ny',
        'uniquePspsSTid', 'primaryDetection', 'bestDetection',
        'gippDetectID', 'gstackDetectID', 'gstackImageID', 'gra', 'gdec',
        'graErr', 'gdecErr', 'gEpoch', 'gPSFMag', 'gPSFMagErr', 'gApMag',
        'gApMagErr', 'gKronMag', 'gKronMagErr', 'ginfoFlag', 'ginfoFlag2',
        'ginfoFlag3', 'gnFrames', 'gxPos', 'gyPos', 'gxPosErr', 'gyPosErr',
        'gpsfMajorFWHM', 'gpsfMinorFWHM', 'gpsfTheta', 'gpsfCore',
        'gpsfLikelihood', 'gpsfQf', 'gpsfQfPerfect', 'gpsfChiSq',
        'gmomentXX', 'gmomentXY', 'gmomentYY', 'gmomentR1', 'gmomentRH',
        'gPSFFlux', 'gPSFFluxErr', 'gApFlux', 'gApFluxErr', 'gApFillFac',
        'gApRadius', 'gKronFlux', 'gKronFluxErr', 'gKronRad', 'gexpTime',
        'gExtNSigma', 'gsky', 'gskyErr', 'gzp', 'gPlateScale',
        'rippDetectID', 'rstackDetectID', 'rstackImageID', 'rra', 'rdec',
        'rraErr', 'rdecErr', 'rEpoch', 'rPSFMag', 'rPSFMagErr', 'rApMag',
        'rApMagErr', 'rKronMag', 'rKronMagErr', 'rinfoFlag', 'rinfoFlag2',
        'rinfoFlag3', 'rnFrames', 'rxPos', 'ryPos', 'rxPosErr', 'ryPosErr',
        'rpsfMajorFWHM', 'rpsfMinorFWHM', 'rpsfTheta', 'rpsfCore',
        'rpsfLikelihood', 'rpsfQf', 'rpsfQfPerfect', 'rpsfChiSq',
        'rmomentXX', 'rmomentXY', 'rmomentYY', 'rmomentR1', 'rmomentRH',
        'rPSFFlux', 'rPSFFluxErr', 'rApFlux', 'rApFluxErr', 'rApFillFac',
        'rApRadius', 'rKronFlux', 'rKronFluxErr', 'rKronRad', 'rexpTime',
        'rExtNSigma', 'rsky', 'rskyErr', 'rzp', 'rPlateScale',
        'iippDetectID', 'istackDetectID', 'istackImageID', 'ira', 'idec',
        'iraErr', 'idecErr', 'iEpoch', 'iPSFMag', 'iPSFMagErr', 'iApMag',
        'iApMagErr', 'iKronMag', 'iKronMagErr', 'iinfoFlag', 'iinfoFlag2',
        'iinfoFlag3', 'inFrames', 'ixPos', 'iyPos', 'ixPosErr', 'iyPosErr',
        'ipsfMajorFWHM', 'ipsfMinorFWHM', 'ipsfTheta', 'ipsfCore',
        'ipsfLikelihood', 'ipsfQf', 'ipsfQfPerfect', 'ipsfChiSq',
        'imomentXX', 'imomentXY', 'imomentYY', 'imomentR1', 'imomentRH',
        'iPSFFlux', 'iPSFFluxErr', 'iApFlux', 'iApFluxErr', 'iApFillFac',
        'iApRadius', 'iKronFlux', 'iKronFluxErr', 'iKronRad', 'iexpTime',
        'iExtNSigma', 'isky', 'iskyErr', 'izp', 'iPlateScale',
        'zippDetectID', 'zstackDetectID', 'zstackImageID', 'zra', 'zdec',
        'zraErr', 'zdecErr', 'zEpoch', 'zPSFMag', 'zPSFMagErr', 'zApMag',
        'zApMagErr', 'zKronMag', 'zKronMagErr', 'zinfoFlag', 'zinfoFlag2',
        'zinfoFlag3', 'znFrames', 'zxPos', 'zyPos', 'zxPosErr', 'zyPosErr',
        'zpsfMajorFWHM', 'zpsfMinorFWHM', 'zpsfTheta', 'zpsfCore',
        'zpsfLikelihood', 'zpsfQf', 'zpsfQfPerfect', 'zpsfChiSq',
        'zmomentXX', 'zmomentXY', 'zmomentYY', 'zmomentR1', 'zmomentRH',
        'zPSFFlux', 'zPSFFluxErr', 'zApFlux', 'zApFluxErr', 'zApFillFac',
        'zApRadius', 'zKronFlux', 'zKronFluxErr', 'zKronRad', 'zexpTime',
        'zExtNSigma', 'zsky', 'zskyErr', 'zzp', 'zPlateScale',
        'yippDetectID', 'ystackDetectID', 'ystackImageID', 'yra', 'ydec',
        'yraErr', 'ydecErr', 'yEpoch', 'yPSFMag', 'yPSFMagErr', 'yApMag',
        'yApMagErr', 'yKronMag', 'yKronMagErr', 'yinfoFlag', 'yinfoFlag2',
        'yinfoFlag3', 'ynFrames', 'yxPos', 'yyPos', 'yxPosErr', 'yyPosErr',
        'ypsfMajorFWHM', 'ypsfMinorFWHM', 'ypsfTheta', 'ypsfCore',
        'ypsfLikelihood', 'ypsfQf', 'ypsfQfPerfect', 'ypsfChiSq',
        'ymomentXX', 'ymomentXY', 'ymomentYY', 'ymomentR1', 'ymomentRH',
        'yPSFFlux', 'yPSFFluxErr', 'yApFlux', 'yApFluxErr', 'yApFillFac',
        'yApRadius', 'yKronFlux', 'yKronFluxErr', 'yKronRad', 'yexpTime',
        'yExtNSigma', 'ysky', 'yskyErr', 'yzp', 'yPlateScale', 'distance']

    if fullDF:
        colnames = np.concatenate([colnames, ['NED_name', 'NED_type', 'NED_vel', 'NED_redshift', 'NED_mag',
    'i-z', 'g-r', 'r-i', 'g-i', 'z-y', 'g-rErr', 'r-iErr', 'i-zErr',
    'z-yErr', 'gApMag_gKronMag', 'rApMag_rKronMag', 'iApMag_iKronMag',
    'zApMag_zKronMag', 'yApMag_yKronMag', '7DCD', 'class', 'dist/DLR',
    'dist', 'TransientClass', 'TransientRA', 'TransientDEC',
    'TransientRedshift', 'TransientName']])
    df = pd.DataFrame(columns = colnames)
    return df

def getAllPostageStamps(df, tempSize, path=".", verbose=False):
    """Loops through a pandas dataframe and saves PS1 stacked color images of all
       host galaxies.

    :param df: Dataframe of PS1 sources.
    :type df: Pandas DataFrame
    :param tempSize: The downloaded image will be tempSize x tempSize pixels.
    :type tempSize: int
    :param path: Filepath where images should be saved.
    :type path: str
    :param verbose: If true, The progress of the image downloads is printed.
    :type verbose: bool
    """

    for i in np.arange(len(df["raMean"])):
            tempRA = df.loc[i, 'raMean']
            tempDEC = df.loc[i, 'decMean']
            tempName = df.loc[i, 'TransientName']
            a = find_all(path+"/%s.png" % tempName, path)
            if not a:
                try:
                    img = getcolorim(tempRA, tempDEC, size=tempSize, filters="grizy", format="png")
                    img.save(path+"/%s.png" % tempName)
                    if verbose:
                        print("Saving postage stamp for the host of %s."% tempName)
                except:
                    continue

def ps1crossmatch_GLADE(foundGladeHosts):
    """Gets PS1 photometry for GLADE sources by crossmatching.

    :param foundGladeHosts: DataFrame of Glade sources to cross-match in ps1
    :type foundGladeHosts: Pandas DataFrame
    """
    ps1matches = []
 
    # Create dummy column of objID - we'll fill these in the loop below.
    foundGladeHosts['objID'] = foundGladeHosts.index
    foundGladeHosts = foundGladeHosts.astype({'objID': 'int64'})
    for idx, row in foundGladeHosts.iterrows():
        a = ps1cone(row.raMean, row.decMean, 10./3600)
        if a:
            a = ascii.read(a)
            a = a.to_pandas()
            ps1match = a.iloc[[0]]
            #get rid of coord info - GLADE properties are better!
            ps1match.drop(['raMean', 'decMean'], axis=1, inplace=True)
            foundGladeHosts.loc[foundGladeHosts.index == idx, 'objID'] = ps1match['objID'].values[0]
            ps1matches.append(ps1match)
    if len(ps1matches) > 0:
        ps1matches = pd.concat(ps1matches, ignore_index=True)
    else:
        print("Warning! Found no ps1 sources for GLADE galaxies.")
        ps1matches = create_dummy_df(nRows=len(foundGladeHosts))
    foundGladeHosts = foundGladeHosts.merge(ps1matches, how = 'outer')
    return foundGladeHosts

def get_hosts(path, transient_fn, fn_Host, rad):
    """Wrapper function for getting candidate host galaxies in PS1 for dec>-30 deg and
       in Skymapper for dec<-30 deg.

    :param path: Filepath where csv of candidate hosts should be saved.
    :type path: str
    :param transient_fn: Filename of csv containing the transients to associate (and their coordinates).
    :type transient_fn: str
    :param fn_Host: Filename of csv containing candidate host galaxy properties.
    :type fn_Host: str
    :param rad: Search radius of the algorithm, in arcseconds.
    :type rad: float
    :return: Dataframe of all candidate host galaxies.
    :rtype: Pandas DataFrame
    """

    transient_df = pd.read_csv(path+"/"+transient_fn)
    now = datetime.now()
    dict_fn = fn_Host.replace(".csv", "") + ".p"

    tempDEC = Angle(transient_df['DEC'], unit=u.deg)
    tempDEC = tempDEC.deg

    # distinguish between PS1 sources and Skymapper sources.
    df_North = transient_df[(tempDEC > -30)].reset_index()
    df_South = transient_df[(tempDEC <= -30)].reset_index()

    # get southern-hemisphere sources
    append=0
    if len(df_South) > 0:
        print("Finding southern sources with SkyMapper...")
        find_host_info_SH(df_South, fn_Host, dict_fn, path, rad)
        append=1

    # get northern-hemisphere sources
    if len(df_North) > 0:
        print("Finding northern sources with Pan-starrs...")
        find_host_info_PS1(df_North, fn_Host, dict_fn, path, rad, append=append)

    # load the saved csv and remove duplicates, then return
    host_df = pd.read_csv(path+"/"+fn_Host)
    host_df = host_df.drop_duplicates()
    host_df.to_csv(path+"/"+fn_Host[:-4]+"_cleaned.csv", index=False)
    return host_df

def find_all(name, path):
    """Crawls through a directory and all its sub-directories looking for a file matching
       \\'name\\'. If found, it is returned.

    :param name: The filename for which to search.
    :type name: str
    :param path: The directory to search.
    :type path: str
    :return: The list of absolute paths to all files called \\'name\\' in \\'path\\'.
    :rtype: list
    """

    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def getimages(ra,dec,size=240,filters="grizy", type='stack'):
    """Query ps1filenames.py service to get a list of images.

    :param ra: Right ascension of position, in degrees.
    :type ra: float
    :param dec: Declination of position, in degrees.
    :type dec: float
    :param size: The image size in pixels (0.25 arcsec/pixel)
    :type size: int
    :param filters: A string with the filters to include
    :type filters: str
    :return: The results of the search for relevant images.
    :rtype: Astropy Table
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}&type={type}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False, type='stack'):
    """Get the URL for images in the table.

    :param ra: Right ascension of position, in degrees.
    :type ra: float
    :param dec: Declination of position, in degrees.
    :type dec: float
    :param size: The extracted image size in pixels (0.25 arcsec/pixel)
    :type size: int
    :param output_size: output (display) image size in pixels (default = size).
        The output_size has no effect for fits format images.
    :type output_size: int
    :param filters: The string with filters to include.
    :type filters: str
    :param format: The data format (options are \\"jpg\\", \\"png" or \\"fits\\").
    :type format: str
    :param color: If True, creates a color image (only for jpg or png format).
        If False, return a list of URLs for single-filter grayscale images.
    :type color: bool, optional
    :return: The url for the image to download.
    :rtype: str
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,size=size,filters=filters, type=type)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)

    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    """Get a PS1 color image at a sky position.

    :param ra: Right ascension of position, in degrees.
    :type ra: float
    :param dec: Declination of position, in degrees.
    :type dec: float
    :param size: The extracted image size in pixels (0.25 arcsec/pixel)
    :type size: int
    :param output_size: output (display) image size in pixels (default = size).
        The output_size has no effect for fits format images.
    :type output_size: int
    :param filters: The string with filters to include.
    :type filters: str
    :param format: The data format (options are \\'jpg\\', \\'png\\' or \\'fits\\').
    :type format: str
    :return: The image.
    :rtype: PIL Image
    """

    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True, type='stack')
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im

def get_PS1_type(path, ra, dec, size, band, type):
    """Download and save PS1 imaging data in a given band of a given type.

    :param path: filepath where the image will be saved.
    :type path: str
    :param ra: Right ascension of position, in degrees.
    :type ra: float
    :param dec: Declination of position, in degrees.
    :type dec: float
    :param size: The extracted image size in pixels (0.25 arcsec/pixel)
    :type size: int
    :param band: The PS1 band.
    :type band: str
    :param type: The type of imaging data to obtain. Options are given below for the PS1 stack.
        \\'stack.mask\\' - indicate which pixels in the stack are good and which are bad
        \\'stack.wt\\' - stack variance images
        \\'stack.num\\' - contain the number of warps with valid data which contributed to each pixel
        \\'stack.exp\\' - contain the exposure time in seconds which contributed to each pixel
        \\'stack.expw\\' - weighted exposure time maps
        See more information at https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images.
    :type type: str
    """

    fitsurl = geturl(ra, dec, size=size, filters="{}".format(band), format="fits", type=type)
    fh = fits.open(fitsurl[0])
    fh.writeto(path + '/PS1_ra={}_dec={}_{}arcsec_{}_{}.fits'.format(ra, dec, int(size*0.25), band, type))

def get_PS1_Pic(path, objID, ra, dec, size, band, safe=False):
    """Downloads PS1 picture (in fits) centered at a given location.

    :param path: The filepath where the fits file will be saved.
    :type path: str
    :param objID: The PS1 objID of the object of interest (to save as filename).
    :type objID: int
    :param ra: Right ascension of position, in degrees.
    :type ra: float
    :param dec: Declination of position, in degrees.
    :type dec: float
    :param size: The extracted image size in pixels (0.25 arcsec/pixel)
    :type size: int
    :param band: The PS1 band.
    :type band: str
    :param safe: If True, include the objID of the object of interest in the filename
        (useful when saving multiple files at comparable positions).
    :type safe: bool, optional
    """

    fitsurl = geturl(ra, dec, size=size, filters="{}".format(band), format="fits")
    fh = fits.open(fitsurl[0])
    if safe==True:
        fh.writeto(path + '/PS1_{}_{}arcsec_{}.fits'.format(objID, int(size*0.25), band))
    else:
        fh.writeto(path + '/PS1_ra={}_dec={}_{}arcsec_{}.fits'.format(ra, dec, int(size*0.25), band))

def ps1metadata(table="mean",release="dr1",baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table.

    :param table: Table type. Can be \\'mean\\', \\'stack\\', or \\'detection\\'
    :type table: str
    :param release: The Pan-STARRS data release. Can be \\'dr1' or \\'dr2\\'.
    :type release: str
    :param baseurl: The base URL for the request
    :type baseurl: str
    :return: The table containing the metadata, with columns name, type, and description.
    :rtype: Astropy Table
    """

    checklegal(table,release)
    url = "{baseurl}/{release}/{table}/metadata".format(**locals())
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()

    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab


def mastQuery(request):
    """Perform a MAST query.

    :param request: The MAST request json object.
    :type request: dictionary
    :return: The response HTTP headers.
    :rtype: HTTP header
    :return: The data obtained.
    :rtype: str
    """

    server='mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content

def resolve(name):
    """Get the RA and Dec for an object using the MAST name resolver.

    :param name: Name of the object of interest.
    :type name: str
    :return: Position of resolved object, in degrees.
    :rtype: tuple
    """

    resolverRequest = {'service':'Mast.Name.Lookup',
                       'params':{'input':name,
                                 'format':'json'
                                },
                      }
    headers,resolvedObjectString = mastQuery(resolverRequest)
    resolvedObject = json.loads(resolvedObjectString)

    # The resolver returns a variety of information about the resolved object,
    # however for our purposes all we need are the RA and Dec
    try:
        objRa = resolvedObject['resolvedCoordinate'][0]['ra']
        objDec = resolvedObject['resolvedCoordinate'][0]['decl']
    except IndexError as e:
        raise ValueError("Unknown object '{}'".format(name))
    return (objRa, objDec)

def checklegal(table,release):
    """Checks if this combination of table and release is acceptable.
       Raises a ValueError exception if there is problem.

    :param table: Table type. Can be \\'mean\\', \\'stack\\', or \\'detection\\'
    :type table: str
    :param release: The Pan-STARRS data release. Can be \\'dr1\\' or \\'dr2\\'.
    :type release: str
    :raises ValueError: Raises error if table and release combination are invalid.
    """

    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection", "forced_mean")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))

def ps1search(table="mean",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,**kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius).

    :param table: Table type. Can be \\'mean\\', \\'stack\\', or \\'detection\\'
    :type table: str
    :param release: The Pan-STARRS data release. Can be \\'dr1\\' or 'dr2\\'.
    :type release: str
    :param format: Can be \\'csv\\', \\'votable\\', or \\'json\\'.
    :type format: str
    :param columns: Column names to include (None means use defaults).
    :type columns: str
    :param baseurl: Base URL for the request.
    :type baseurl: str
    :param verbose: If true, print info about request.
    :type verbose: bool
    :param \\*\\*kw: Other parameters (e.g., \\'nDetections.min\\':2).  Note that this is required!
    :type \\*\\*kw: dictionary
    :return: Result of PS1 query, in \\'csv\\', \\'votable\\', or \\'json\\' format.
    :rtype: Same as \\'format\\'
    """

    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    checklegal(table,release)
    if format not in ("csv","votable","json"):
        raise ValueError("Bad value for format")
    url = "{baseurl}/{release}/{table}.{format}".format(**locals())
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        data['columns'] = '[{}]'.format(','.join(columns))

    # either get or post works
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text


def ps1cone(ra,dec,radius,table="stack",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,**kw):
    """Do a cone search of the PS1 catalog. Note that this is just a thin wrapper for the function \\'ps1search\\'.

    :param ra: Right ascension of central coordinate, in J200 degrees.
    :type ra: float
    :param dec: Declination of central coordinate, in J2000 degrees.
    :type dec: float
    :param radius: Search radius, in degrees (<= 0.5 degrees)
    :type radius: float
    :param table: Table type. Can be \\'mean\\', \\'stack\\', or \\'detection\\'
    :type table: str
    :param release: The Pan-STARRS data release. Can be \\'dr1\\' or \\'dr2\\'.
    :type release: str
    :param format: Can be \\'csv\\', \\'votable\\', or \\'json\\'.
    :type format: str
    :param columns: Column names to include (None means use defaults).
    :type columns: str
    :param baseurl: Base URL for the request.
    :type baseurl: str
    :param verbose: If true, print info about request.
    :type verbose: bool
    :param \\*\\*kw: Other parameters (e.g., \\'nDetections.min\\':2).  Note that this is required!
    :type \\*\\*kw: dictionary
    :return: Result of PS1 query, in \\'csv\\', \\'votable\\', or \\'json\\' format.
    :rtype: Same as \\'format\\'
    """

    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data)

def ps1metadata(table="mean", release="dr1", baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table. Snagged from the
       wonderful API at https://ps1images.stsci.edu/ps1_dr2_api.html.

    :param table: Table type. Can be \\'mean\\', \\'stack\\', or \\'detection\\'
    :type table: str
    :param release: The Pan-STARRS data release. Can be \\'dr1\\' or \\'dr2\\'.
    :type release: str
    :param baseurl: Base URL for the request.
    :type baseurl: str
    :return: Table with columns name, type, description.
    :rtype: Astropy Table
    """

    checklegal(table,release)
    url = f"{baseurl}/{release}/{table}/metadata"
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()

    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab

def find_host_info_PS1(df, fn, dict_fn, path, rad, append=0):
    """Querying PS1 for all objects within rad arcsec of each SN.

    :param df: Dataframe of transient information (name and coordinates).
    :type df: Pandas DataFrame
    :param fn: Filename of PS1 candidate host dataframe.
    :type fn: str
    :param dict_fn: The filename of the dictionary to keep track of transient-candidate host matches.
        Keys are transient names, values are lists containing objIDs of all host candidates.
    :type dict_fn: dictionary
    :param path: The filepath where df will be saved.
    :type path: str
    :param rad: The search radius, in arcsec.
    :type rad: float
    :param append: If True, append results to fn. If False, create a new file.
    :type append: bool, optional
    """

    i = 0

    # The dictionary to map SN names to nearby obj IDs in PS1
    SN_Host_PS1 = {}

    #a running list of all PS1 results
    PS1_queries = []
    for j, row in enumerate(df.itertuples(), 1):
            if ":" in str(row.RA):
                tempRA = Angle(row.RA, unit=u.hourangle)
            else:
                tempRA = Angle(row.RA, unit=u.deg)
            tempDEC = Angle(row.DEC, unit=u.deg)
            a = ps1cone(tempRA.degree,tempDEC.degree,rad/3600,table="stack",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False)
            if a:
                a = ascii.read(a)
                a = a.to_pandas()
                PS1_queries.append(a)
                SN_Host_PS1[row.Name] = np.array(a['objID'])
            else:
                SN_Host_PS1[row.Name] = np.array([])

            # Print status messages every 10 lines
            if j%10 == 0:
                print("Processed {} of {} lines!".format(j, len(df.Name)))

            # Print every query to a file. This was done in order
            # to prevent the code crashing after processing 99% of the data
            # frame and losing everything. This allows for duplicates, though,
            # so they should be removed before the file is used again.
            if (len(PS1_queries) > 0):
                PS1_hosts = pd.concat(PS1_queries)
                PS1_hosts = PS1_hosts.drop_duplicates()

                #match up rows to skymapper cols to join into a single dataframe
                newCols = np.array(['SkyMapper_StarClass', 'gelong','g_a','g_b','g_pa',
                'relong','r_a','r_b','r_pa',
                'ielong','i_a','i_b','i_pa',
                'zelong','z_a','z_b','z_pa'])
                for col in newCols:
                    PS1_hosts[col] = np.nan
                PS1_queries = []
                if not append:
                    PS1_hosts.to_csv(path+fn, header=True, index=False)
                    i = 1
                    append = True
                else:
                    PS1_hosts.to_csv(path+"/"+fn, mode='a+', header=False, index=False)
            else:
                print("No potential hosts found for this object...")

            # Save host info
            if not os.path.exists(path+ '/dictionaries/'):
            	os.makedirs(path+'/dictionaries/')
            option = "wb"
            if append:
                option = "ab"
            with open(path+"/dictionaries/" + dict_fn, option) as fp:
                pickle.dump(SN_Host_PS1, fp, protocol=pickle.HIGHEST_PROTOCOL)

def find_host_info_SH(df, fn, dict_fn, path, rad):
    """VO Cone Search for all objects within rad arcsec of SNe (for Southern-Hemisphere (SH) objects).

    :param df: Dataframe of transient information (name and coordinates).
    :type df: Pandas DataFrame
    :param fn: Filename of PS1 candidate host dataframe.
    :type fn: str
    :param dict_fn: The filename of the dictionary to keep track of transient-candidate host matches.
        Keys are transient names, values are lists containing objIDs of all host candidates.
    :type dict_fn: dictionary
    :param path: The filepath where df will be saved.
    :type path: str
    :param rad: The search radius, in arcsec.
    :type rad: float
    :param append: If True, append results to fn. If False, create a new file.
    :type append: bool, optional
    """

    i = 0
    SN_Host_SH = {}
    SH_queries = []
    pd.options.mode.chained_assignment = None
    for j, row in df.iterrows():
            if ":" in str(row.RA):
                tempRA = Angle(row.RA, unit=u.hourangle)
            else:
                tempRA = Angle(row.RA, unit=u.deg)
            tempDEC = Angle(row.DEC, unit=u.deg)
            a = pd.DataFrame({})
            a = southernSearch(tempRA.degree,tempDEC.degree, rad)
            if len(a)>0:
                SH_queries.append(a)
                SN_Host_SH[row.Name] = np.array(a['objID'])
            else:
                SN_Host_SH[row.Name] = np.array([])

            # Print status messages every 10 lines
            if j%10 == 0:
                print("Processed {} of {} lines!".format(j, len(df.Name)))

            # Print every query to a file Note: this was done in order
            # to prevent the code crashing after processing 99% of the data
            # frame and losing everything. This allows for duplicates though,
            # so they should be removed before the file is used again
            if (len(SH_queries) > 0):
                SH_hosts = pd.concat(SH_queries)
                SH_hosts = SH_hosts.drop_duplicates()
                SH_queries = []
                if i == 0:
                    SH_hosts.to_csv(path+"/"+fn, header=True, index=False)
                    i = 1
                else:
                    SH_hosts.to_csv(path+"/"+fn, mode='a+', header=False, index=False)
            else:
                print("No potential hosts found for this object...")

            # Save host info
            if not os.path.exists(path+ '/dictionaries/'):
            	os.makedirs(path+'/dictionaries/')
            with open(path+"/dictionaries/" + dict_fn, 'wb') as fp:
                pickle.dump(SN_Host_SH, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pd.options.mode.chained_assignment = 'warn'

def southernSearch(ra, dec, rad):
    """Conducts a cone search for Skymapper objects at a given position.

    :param ra: Right ascension of central coordinate, in J200 degrees.
    :type ra: float
    :param dec: Declination of central coordinate, in J2000 degrees.
    :type dec: float
    :param radius: Search radius, in degrees (<= 0.5 degrees)
    :type radius: float
    :return: Dataframe of Skymapper objects formatted to be joined with PS1 sources.
    :rtype: Pandas DataFrame.
    """

    searchCoord = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    responseMain = requests.get("http://skymapper.anu.edu.au/sm-cone/public/query?CATALOG=dr2.master&RA=%.5f&DEC=%.5f&SR=%.5f&RESPONSEFORMAT=CSV&VERB=3" %(ra, dec, (rad/3600)))
    responsePhot = requests.get("http://skymapper.anu.edu.au/sm-cone/public/query?CATALOG=dr2.photometry&RA=%.5f&DEC=%.5f&SR=%.5f&RESPONSEFORMAT=CSV&VERB=3" %(ra, dec, (rad/3600)))

    dfMain = pd.read_csv(BytesIO(responseMain.content))
    dfPhot = pd.read_csv(BytesIO(responsePhot.content))

    filt_dfs = []
    for filter in 'griz':

        #add the photometry columns
        tempDF = dfPhot[dfPhot['filter']==filter]
        if len(tempDF) < 1:
            pd.concat([tempDF, pd.Series()], ignore_index=True) #add dummy row for the sake of not crashing
        for col in tempDF.columns.values:
            if col != 'object_id':
                tempDF[filter + col] = tempDF[col]
                del tempDF[col]

        # take the column with the smallest uncertainty in the measured semi-major axis -
        # this is what we'll use to
        # calculate DLR later!
        tempDF = tempDF.loc[tempDF.groupby("object_id")[filter + 'e_a'].idxmin()]
        filt_dfs.append(tempDF)

    test = filt_dfs[0]

    for i in np.arange(1, len(filt_dfs)):
        test = test.merge(filt_dfs[i], on='object_id', how='outer')

    test['object_id'] =  np.nan_to_num(test['object_id'])
    test['object_id'] = test['object_id'].astype(np.int64)

    fullDF = test.merge(dfMain, on='object_id', how='outer')

    flag_mapping = {'objID':'object_id', 'raMean':'raj2000',
        'decMean':'dej2000','gKronRad':'gradius_kron',
        'rKronRad':'rradius_kron', 'iKronRad':'iradius_kron',
        'zKronRad':'iradius_kron', 'yKronRad':np.nan,
        'gPSFMag':'g_psf', 'rPSFMag':'r_psf','iPSFMag':'i_psf',
        'zPSFMag':'z_psf','yPSFMag':np.nan,'gPSFMagErr':'e_g_psf',
        'rPSFMagErr':'e_r_psf','iPSFMagErr':'e_i_psf','zPSFMagErr':'e_z_psf',
        'yPSFMagErr':np.nan,'gKronMag':'g_petro', 'rKronMag':'r_petro',
        'iKronMag':'i_petro', 'zKronMag':'z_petro', 'yKronMag':np.nan,
        'gKronMagErr':'e_g_petro', 'rKronMagErr':'e_r_petro',
        'iKronMagErr':'e_i_petro', 'zKronMagErr':'e_z_petro',
        'yKronMagErr':np.nan, 'ng':'g_ngood', 'nr':'r_ngood', 'ni':'i_ngood',
        'nz':'z_ngood', 'graErr':'e_raj2000', 'rraErr':'e_raj2000', 'iraErr':'e_raj2000',
        'zraErr':'e_raj2000', 'gdecErr':'e_dej2000', 'rdecErr':'e_dej2000','idecErr':'e_dej2000',
        'zdecErr':'e_dej2000','l':'glon', 'b':'glat', 'gra':'gra_img', 'rra':'rra_img', 'ira':'ira_img',
        'zra':'zra_img', 'yra':'raj2000', 'gdec':'gdecl_img', 'rdec':'rdecl_img',
        'idec':'idecl_img', 'zdec':'zdecl_img', 'ydec':'dej2000', 'gKronFlux':'gflux_kron',
        'rKronFlux':'rflux_kron', 'iKronFlux':'iflux_kron', 'zKronFlux':'zflux_kron', 'yKronFlux':np.nan,
        'gKronFluxErr':'ge_flux_kron', 'rKronFluxErr':'re_flux_kron', 'iKronFluxErr':'ie_flux_kron',
        'zKronFluxErr':'ze_flux_kron', 'yKronFluxErr':np.nan,
        'gPSFFlux':'gflux_psf',
        'rPSFFlux':'rflux_psf', 'iKronFlux':'iflux_psf', 'zKronFlux':'zflux_psf', 'yKronFlux':np.nan,
        'gPSFFluxErr':'ge_flux_psf', 'rKronFluxErr':'re_flux_psf', 'iKronFluxErr':'ie_flux_psf',
        'zPSFFluxErr':'ze_flux_psf', 'yKronFluxErr':np.nan, 'gpsfChiSq':'gchi2_psf',
        'rpsfChiSq':'rchi2_psf', 'ipsfChiSq':'ichi2_psf', 'zpsfChiSq':'zchi2_psf',
        'ypsfChiSq':np.nan, 'nDetections':'ngood', 'SkyMapper_StarClass':'rclass_star',
        'distance':'r_cntr', 'objName':'object_id',
        'g_elong':'gelong', 'g_a':'ga', 'g_b':'gb', 'g_pa':'gpa',
        'r_elong':'relong', 'r_a':'ra', 'r_b':'rb', 'r_pa':'rpa',
        'i_elong':'ielong', 'i_a':'ia', 'i_b':'ib', 'i_pa':'ipa',
        'z_elong':'zelong', 'z_a':'za', 'z_b':'zb', 'z_pa':'zpa'} #'class_star' should be added

    keepCols = []
    for band in 'griz':
        for rad in ['radius_petro', 'radius_frac20', 'radius_frac50', 'radius_frac90']:
            flag_mapping[band + rad] = band + rad
            keepCols.append(band + rad)

    df_cols = np.array(list(flag_mapping.values()))
    mapped_cols = np.array(list(flag_mapping.keys()))

    PS1_cols = np.array(['objName', 'objAltName1', 'objAltName2', 'objAltName3', 'objID',
           'uniquePspsOBid', 'ippObjID', 'surveyID', 'htmID', 'zoneID',
           'tessID', 'projectionID', 'skyCellID', 'randomID', 'batchID',
           'dvoRegionID', 'processingVersion', 'objInfoFlag', 'qualityFlag',
           'raStack', 'decStack', 'raStackErr', 'decStackErr', 'raMean',
           'decMean', 'raMeanErr', 'decMeanErr', 'epochMean', 'posMeanChisq',
           'cx', 'cy', 'cz', 'lambda', 'beta', 'l', 'b', 'nStackObjectRows',
           'nStackDetections', 'nDetections', 'ng', 'nr', 'ni', 'nz', 'ny',
           'uniquePspsSTid', 'primaryDetection', 'bestDetection',
           'gippDetectID', 'gstackDetectID', 'gstackImageID', 'gra', 'gdec',
           'graErr', 'gdecErr', 'gEpoch', 'gPSFMag', 'gPSFMagErr', 'gApMag',
           'gApMagErr', 'gKronMag', 'gKronMagErr', 'ginfoFlag', 'ginfoFlag2',
           'ginfoFlag3', 'gnFrames', 'gxPos', 'gyPos', 'gxPosErr', 'gyPosErr',
           'gpsfMajorFWHM', 'gpsfMinorFWHM', 'gpsfTheta', 'gpsfCore',
           'gpsfLikelihood', 'gpsfQf', 'gpsfQfPerfect', 'gpsfChiSq',
           'gmomentXX', 'gmomentXY', 'gmomentYY', 'gmomentR1', 'gmomentRH',
           'gPSFFlux', 'gPSFFluxErr', 'gApFlux', 'gApFluxErr', 'gApFillFac',
           'gApRadius', 'gKronFlux', 'gKronFluxErr', 'gKronRad', 'gexpTime',
           'gExtNSigma', 'gsky', 'gskyErr', 'gzp', 'gPlateScale',
           'rippDetectID', 'rstackDetectID', 'rstackImageID', 'rra', 'rdec',
           'rraErr', 'rdecErr', 'rEpoch', 'rPSFMag', 'rPSFMagErr', 'rApMag',
           'rApMagErr', 'rKronMag', 'rKronMagErr', 'rinfoFlag', 'rinfoFlag2',
           'rinfoFlag3', 'rnFrames', 'rxPos', 'ryPos', 'rxPosErr', 'ryPosErr',
           'rpsfMajorFWHM', 'rpsfMinorFWHM', 'rpsfTheta', 'rpsfCore',
           'rpsfLikelihood', 'rpsfQf', 'rpsfQfPerfect', 'rpsfChiSq',
           'rmomentXX', 'rmomentXY', 'rmomentYY', 'rmomentR1', 'rmomentRH',
           'rPSFFlux', 'rPSFFluxErr', 'rApFlux', 'rApFluxErr', 'rApFillFac',
           'rApRadius', 'rKronFlux', 'rKronFluxErr', 'rKronRad', 'rexpTime',
           'rExtNSigma', 'rsky', 'rskyErr', 'rzp', 'rPlateScale',
           'iippDetectID', 'istackDetectID', 'istackImageID', 'ira', 'idec',
           'iraErr', 'idecErr', 'iEpoch', 'iPSFMag', 'iPSFMagErr', 'iApMag',
           'iApMagErr', 'iKronMag', 'iKronMagErr', 'iinfoFlag', 'iinfoFlag2',
           'iinfoFlag3', 'inFrames', 'ixPos', 'iyPos', 'ixPosErr', 'iyPosErr',
           'ipsfMajorFWHM', 'ipsfMinorFWHM', 'ipsfTheta', 'ipsfCore',
           'ipsfLikelihood', 'ipsfQf', 'ipsfQfPerfect', 'ipsfChiSq',
           'imomentXX', 'imomentXY', 'imomentYY', 'imomentR1', 'imomentRH',
           'iPSFFlux', 'iPSFFluxErr', 'iApFlux', 'iApFluxErr', 'iApFillFac',
           'iApRadius', 'iKronFlux', 'iKronFluxErr', 'iKronRad', 'iexpTime',
           'iExtNSigma', 'isky', 'iskyErr', 'izp', 'iPlateScale',
           'zippDetectID', 'zstackDetectID', 'zstackImageID', 'zra', 'zdec',
           'zraErr', 'zdecErr', 'zEpoch', 'zPSFMag', 'zPSFMagErr', 'zApMag',
           'zApMagErr', 'zKronMag', 'zKronMagErr', 'zinfoFlag', 'zinfoFlag2',
           'zinfoFlag3', 'znFrames', 'zxPos', 'zyPos', 'zxPosErr', 'zyPosErr',
           'zpsfMajorFWHM', 'zpsfMinorFWHM', 'zpsfTheta', 'zpsfCore',
           'zpsfLikelihood', 'zpsfQf', 'zpsfQfPerfect', 'zpsfChiSq',
           'zmomentXX', 'zmomentXY', 'zmomentYY', 'zmomentR1', 'zmomentRH',
           'zPSFFlux', 'zPSFFluxErr', 'zApFlux', 'zApFluxErr', 'zApFillFac',
           'zApRadius', 'zKronFlux', 'zKronFluxErr', 'zKronRad', 'zexpTime',
           'zExtNSigma', 'zsky', 'zskyErr', 'zzp', 'zPlateScale',
           'yippDetectID', 'ystackDetectID', 'ystackImageID', 'yra', 'ydec',
           'yraErr', 'ydecErr', 'yEpoch', 'yPSFMag', 'yPSFMagErr', 'yApMag',
           'yApMagErr', 'yKronMag', 'yKronMagErr', 'yinfoFlag', 'yinfoFlag2',
           'yinfoFlag3', 'ynFrames', 'yxPos', 'yyPos', 'yxPosErr', 'yyPosErr',
           'ypsfMajorFWHM', 'ypsfMinorFWHM', 'ypsfTheta', 'ypsfCore',
           'ypsfLikelihood', 'ypsfQf', 'ypsfQfPerfect', 'ypsfChiSq',
           'ymomentXX', 'ymomentXY', 'ymomentYY', 'ymomentR1', 'ymomentRH',
           'yPSFFlux', 'yPSFFluxErr', 'yApFlux', 'yApFluxErr', 'yApFillFac',
           'yApRadius', 'yKronFlux', 'yKronFluxErr', 'yKronRad', 'yexpTime',
           'yExtNSigma', 'ysky', 'yskyErr', 'yzp', 'yPlateScale', 'distance', 'SkyMapper_StarClass',
           'g_elong','g_a','g_b','g_pa',
           'r_elong','r_a','r_b','r_pa',
           'i_elong','i_a','i_b','i_pa',
           'z_elong','z_a','z_b','z_pa'])

    for i in np.arange(len(df_cols)):
        if df_cols[i] == 'nan':
            fullDF[mapped_cols[i]] = np.nan
        else:
            fullDF[mapped_cols[i]] = fullDF[df_cols[i]]

    # save the plate scale of skymapper in each band
    for band in 'grizy':
        fullDF['%sPlateScale'%band] = 0.5
    fullDF['primaryDetection'] = 1
    fullDF['bestDetection'] = 1

    #dummy variable set so that no sources get cut by qualityFlag in PS1 (which isn't in SkyMapper)
    fullDF['qualityFlag'] = 0

    #dummy variable
    fullDF['ny'] = 1
    colSet = np.concatenate([list(flag_mapping.keys()), ['gPlateScale', 'rPlateScale',
        'iPlateScale', 'zPlateScale', 'yPlateScale', 'primaryDetection', 'bestDetection', 'qualityFlag', 'ny']])

    fullDF = fullDF[colSet]

    leftover = set(PS1_cols) - set(fullDF.columns.values)
    for col in leftover:
        fullDF[col] = np.nan

    # arrange in the correct order for combining with northern-hemisphere PS1 sources
    fullDF = fullDF[PS1_cols]
    fullDF.drop_duplicates(subset=['objID'], inplace=True)
    return fullDF

def getDR2_petrosianSizes(ra_arr, dec_arr, rad):
    """Retrieves petrosian radius information from DR2 for panstarrs sources.

    :param ra_arr: List of right ascension values, in degrees.
    :type ra_arr: list
    :param dec_arr: List of declination values, in degrees.
    :type dec_arr: list
    :param rad: The search radius, in arcseconds.
    :type rad: float
    :return: Dataframe containing dr2 petrosian radiuses.
    :rtype: Pandas DataFrame
    """

    if len(ra_arr) < 1:
        return

    petroList = []
    for i in np.arange(len(ra_arr)):
        query = """select st.objID, st.primaryDetection, st.gpetR90, st.rpetR90, st.ipetR90, st.zpetR90, st.ypetR90
        from fGetNearbyObjEq(%.3f,%.3f,%.1f/60.0) nb
        inner join StackPetrosian st on st.objID=nb.objid where st.primaryDetection = 1""" %(ra_arr[i], dec_arr[i], rad)

        jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
        tab = jobs.quick(query, task_name="PetrosianRadiusSearch")
        df_petro = tab.to_pandas()
        petroList.append(df_petro)

    df_petro_full = pd.concat(petroList)
    return df_petro_full

def getDR2_halfLightSizes(ra_arr, dec_arr, rad):
    """Retrieves half-light radius information from DR2 for panstarrs sources.

    :param ra_arr: List of right ascension values, in degrees.
    :type ra_arr: list
    :param dec_arr: List of declination values, in degrees.
    :type dec_arr: list
    :param rad: The search radius, in arcseconds.
    :type rad: float
    :return: Dataframe containing dr2 half-light radiuses.
    :rtype: Pandas DataFrame
    """

    if len(ra_arr) < 1:
        return

    halfLightList = []
    for i in np.arange(len(ra_arr)):
        query = """select st.objID, st.primaryDetection, st.gHalfLightRad, st.rHalfLightRad, st.iHalfLightRad,
        st.zHalfLightRad, st.yHalfLightRad from fGetNearbyObjEq(%.3f,%.3f,%.1f/60.0) nb
        inner join StackModelFitExtra st on st.objID=nb.objid where st.primaryDetection = 1""" %(ra_arr[i], dec_arr[i], rad)

        jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
        tab = jobs.quick(query, task_name="halfLightSearch")
        df_halfLight = tab.to_pandas()
        halfLightList.append(df_halfLight)

    df_halfLight_full = pd.concat(halfLightList)
    return df_halfLight_full
