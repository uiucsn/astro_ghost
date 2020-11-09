from bs4 import BeautifulSoup
import pandas as pd
import requests
import os
from astro_ghost.PS1QueryFunctions import create_df
import urllib.request
from os import listdir
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from astro_ghost.PS1QueryFunctions import find_all

def remove_comments(line,sep="#"):
    for s in sep:
        i = line.find(s)#find the position of
        if i == 0 :
            line = None
    return line

def clean_spectra(dir):
    for filename in os.listdir(dir):
        try:
            data = pd.read_csv(dir+'/' +filename, delim_whitespace=True, header=None)
            plt.plot(data[0], data[1])
            plt.clf()
        except:
            print(filename)
            with open(dir+filename, "r+") as f:
                lines = f.readlines()
                f.seek(0)
                for line in lines:
                    if not line.startswith("#"):
                        f.write(line)
                    else:
                        print(line)
                f.truncate()

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

# combine OSC and TNS transients!
# scrape new TNS objects and
#
def get_new_transients(period, period_unit):
    url = "https://wis-tns.weizmann.ac.il/search?&discovered_period_value=%i&discovered_period_units=%s&unclassified_at=0&classified_sne=0&name=&name_like=0&isTNS_AT=all&public=all&ra=&decl=&radius=&coords_unit=arcsec&reporting_groupid%5B%5D=null&groupid%5B%5D=null&classifier_groupid%5B%5D=null&objtype%5B%5D=18%2C19%2C20%2C1%2C2%2C15%2C16%2C3%2C103%2C104%2C106%2C100%2C102%2C105%2C4%2C8%2C107%2C6%2C9%2C5%2C7%2C108%2C10%2C110%2C14%2C12%2C13%2C112%2C11&at_type%5B%5D=null&date_start%5Bdate%5D=&date_end%5Bdate%5D=&discovery_mag_min=&discovery_mag_max=&internal_name=&discoverer=&classifier=&spectra_count=&redshift_min=&redshift_max=&hostname=&ext_catid=&ra_range_min=&ra_range_max=&decl_range_min=&decl_range_max=&discovery_instrument%5B%5D=null&classification_instrument%5B%5D=null&associated_groups%5B%5D=null&at_rep_remarks=&class_rep_remarks=&num_page=1000" %(period, period_unit)
    return df

##will have the website scraping
def get_transients(table_dir, fn_SN, tns_loc):
    # current date and time
    #but down the line - write a script to Scrape all new SNe
    #create a transient dataframe from TNS queries
    transient_df = create_df(tns_loc)
    transient_df = transient_df.drop_duplicates()
    transient_df.to_csv(table_dir + fn_SN)

#AND TNS spectrum scraping
def getTNSSpectra(transients, path, verbose=0):
    names = [remove_prefix(x, "SN") for x in transients['Name']]
    for name in names:
        URL = 'https://wis-tns.weizmann.ac.il/object/' + name
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        files = soup.findAll("td", {"class": "cell-asciifile"})

        if (len(files) > 0):
            if verbose:
                print("%s found in TNS. Downloading spectra:"%name)
            for file in files:
                link = file.a['href']
                fn = link.split("/")[-1]
                a = find_all(fn, path)
                if not a:
                    urllib.request.urlretrieve(link, fn)
        else:
            if verbose:
                print("No spectra for %s found on TNS."%name)
            else:
                continue
