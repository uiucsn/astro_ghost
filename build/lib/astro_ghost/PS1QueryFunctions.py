from __future__ import print_function
import numpy as np
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import os
import pandas as pd
from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
from astropy.io import ascii
from astropy.table import Table
import sys
import re
import json
import requests
from datetime import datetime
from astropy import units as u
from astropy.coordinates import Angle
try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib
# std lib
from collections import OrderedDict
from os import listdir
from os.path import isfile, join
# 3rd party
from astropy import utils, io, convolution, wcs
from astropy.visualization import make_lupton_rgb
from astropy.coordinates import name_resolve
from pyvo.dal import sia
import pickle

def getAllPostageStamps(df, tempSize, path=".", verbose=0):
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

def preview_image(i, ra, dec, rad, band, save=1):
    a = find_all("PS1_ra={}_dec={}_{}arcsec_{}.fits".format(ra, dec, rad, band), ".")
    hdul = fits.open(a[0])
    image_file = get_pkg_data_filename(a[0])
    image_data = fits.getdata(image_file, ext=0)
    plt.figure()
    plt.imshow(image_data,cmap='viridis')
    plt.axis('off')
    #plt.colorbar()
    if save:
        plt.savefig("PS1_%i_%s.png" % (i, band))

def get_hosts(path, transient_fn, fn_Host, rad):
    transient_df = pd.read_csv(path+transient_fn)
    now = datetime.now()
    dict_fn = fn_Host.replace(".csv", "") + ".p"
    find_host_info_PS1(transient_df, fn_Host, dict_fn, path, rad)
    host_df = pd.read_csv(path+fn_Host)
    host_df = host_df.drop_duplicates()
    host_df.to_csv(path+fn_Host[:-4]+"_cleaned.csv", index=False)
    return host_df

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def getimages(ra,dec,size=240,filters="grizy", type='stack'):

    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}&type={type}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False, type='stack'):

    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
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

    """Get color image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True, type='stack')
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):

    """Get grayscale image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra,dec,size=size,filters=filter,output_size=output_size,format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im

def get_PS1_type(ra, dec, rad, band, type):
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits", type=type)
    fh = fits.open(fitsurl[0])
    fh.writeto('PS1_ra={}_dec={}_{}arcsec_{}_{}.fits'.format(ra, dec, int(rad*0.25), band, type))

def get_PS1_wt(ra, dec, rad, band):
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits", type='stack.wt')
    fh = fits.open(fitsurl[0])
    fh.writeto('PS1_ra={}_dec={}_{}arcsec_{}_wt.fits'.format(ra, dec, int(rad*0.25), band))

def get_PS1_mask(ra, dec, rad, band):
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits", type='stack.mask')
    fh = fits.open(fitsurl[0])
    fh.writeto('PS1_ra={}_dec={}_{}arcsec_{}_mask.fits'.format(ra, dec, int(rad*0.25), band))

def get_PS1_Pic(objID, ra, dec, rad, band, safe=False):
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits")
    fh = fits.open(fitsurl[0])
    if safe==True:
        fh.writeto('PS1_{}_{}arcsec_{}.fits'.format(objID, int(rad*0.25), band))
    else:
        fh.writeto('PS1_ra={}_dec={}_{}arcsec_{}.fits'.format(ra, dec, int(rad*0.25), band))

# Data Lab
#from dl import queryClient as qc
#from dl.helpers.utils import convert

# set up Simple Image Access (SIA) service
DEF_ACCESS_URL = "http://datalab.noao.edu/sia/des_dr1"
svc = sia.SIAService(DEF_ACCESS_URL)

##################### PS1 HELPER FUNCTIONS ############################################
def ps1metadata(table="mean",release="dr1",baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table

    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    baseurl: base URL for the request

    Returns an astropy table with columns name, type, description
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

        Parameters
        ----------
        request (dictionary): The MAST request json object

        Returns head,content where head is the response HTTP headers, and content is the returned data"""

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
    """Get the RA and Dec for an object using the MAST name resolver

    Parameters
    ----------
    name (str): Name of object

    Returns RA, Dec tuple with position"""

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
    """Checks if this combination of table and release is acceptable

    Raises a VelueError exception if there is problem
    """

    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))

def ps1search(table="mean",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,**kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius)

    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2).  Note this is required!
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
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

# either get or post works
#    r = requests.post(url, data=data)
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text


def ps1cone(ra,dec,radius,table="stack",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,**kw):
    """Do a cone search of the PS1 catalog

    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)
    """

    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data)

#########################END PS1 HELPER FUNCTIONS##############################################

def create_df(tns_loc):
    """Combine all supernovae data into dataframe"""
    files = [f for f in listdir(tns_loc) if isfile(join(tns_loc, f))]
    arr = []
    for file in files:
        tempPD = pd.read_csv(tns_loc+file)
        arr.append(tempPD)
    df = pd.concat(arr)
    df = df.loc[df['RA'] != '00:00:00.000']
    df = df.drop_duplicates()
    df = df.replace({'Anon.': ''})
    df = df.replace({'2019-02-13.49': ''})
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df
    #df.to_csv('SNe_TNS_061019.csv')

def query_ps1_noname(RA, DEC, rad):
    #print("Querying PS1 for nearest host...")
    return ps1cone(RA,DEC,rad/3600,table="stack",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False)

def query_ps1_name(name, rad):
    #print("Querying PS1 with host name!")
    [ra, dec] = resolve(name)
    return ps1cone(ra,dec,rad/3600,table="stack",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False)

# Queries PS1 to find host info for each transient
# Input: df - a dataframe of all spectroscopically classified transients in TNS
#        fn - the output data frame of all PS1 potential hosts
#        dict_fn - the dictionary matching candidate hosts in PS1 and transients
# Output: N/A
def find_host_info_PS1(df, fn, dict_fn, path, rad):
    i = 0
    """Querying PS1 for all objects within rad arcsec of SNe"""
    #os.chdir()
    # SN_Host_PS1 - the dictionary to map SN IDs to nearby obj IDs in PS1
    # EDIT - We now know there are MANY SNe without IDs!! This is pretty problematic, so we're going to switch to
    # keeping a dictionary between names and objIDs
    SN_Host_PS1 = {}
    # PS1_queries - an array of relevant PS1 obj info
    PS1_queries = []
    for j, row in enumerate(df.itertuples(), 1):
            if ":" in str(row.RA):
                tempRA = Angle(row.RA, unit=u.hourangle)
            else:
                tempRA = Angle(row.RA, unit=u.deg)
            tempDEC = Angle(row.DEC, unit=u.deg)
            tempName = row.HostName
            a = ''
            if(row.HostName == ''): # and (int(row.DEC[0:3]) > -30)):
                a = query_ps1_noname(tempRA.degree,tempDEC.degree, rad)
            else:
            #    try:
            #        a = query_ps1_name(tempName, rad)
            #    except:
                a = query_ps1_noname(tempRA.degree,tempDEC.degree, rad)
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
                #print(SN_Host_PS1)

            # Print every query to a file Note: this was done in order
            # to prevent the code crashing after processing 99% of the data
            # frame and losing everything. This allows for duplicates though,
            # so they should be removed before the file is used again
            if (len(PS1_queries) > 0):
                PS1_hosts = pd.concat(PS1_queries)
                PS1_hosts = PS1_hosts.drop_duplicates()
                PS1_queries = []
                if i == 0:
                    PS1_hosts.to_csv(path+fn, header=True, index=False)
                    i = 1
                else:
                    PS1_hosts.to_csv(path+fn, mode='a+', header=False, index=False)
            else:
                print("No potential hosts found for this object...")
            # Save host info
            if not os.path.exists(path+ '/dictionaries/'):
            	os.makedirs(path+'/dictionaries/')
            with open(path+"/dictionaries/" + dict_fn, 'wb') as fp:
                pickle.dump(SN_Host_PS1, fp, protocol=pickle.HIGHEST_PROTOCOL)
