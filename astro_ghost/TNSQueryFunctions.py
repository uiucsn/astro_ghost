from bs4 import BeautifulSoup
import requests
import urllib.request
from astro_ghost.PS1QueryFunctions import find_all
from astro_ghost.ghostHelperFunctions import remove_prefix

def getTNSSpectra(transients, path, verbose=False):
    """Scrapes the Transient Name Server for all public spectra of a transient.

    :param transients: Dataframe of transient information (must contain the column 'Name').
    :type transients: Pandas DataFrame
    :param path: Filepath where spectra will be saved.
    :type path: str
    :param verbose: If True, print progress.
    :type verbose: bool, optional
    """

    names = [remove_prefix(x, "SN") for x in transients['Name']]

    # loop through transients
    for name in names:
        # set up URL for TNS object
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
