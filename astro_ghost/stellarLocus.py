import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors import LogNorm
from astropy.table import Table
import scipy.interpolate as scinterp
import pkg_resources

def convert_to_SDSS(conversions, g_iPS1, band, mPS1):
    """Convert PS1 to SDSS photometry. Recalibration done via the coefficients in
       Hypercalibration: A Pan-starrs1-Based recalibration of the sloan digital sky survey
       photometry (Finkbeiner et al., 2016). Publication can be found at
       https://iopscience.iop.org/article/10.3847/0004-637X/822/2/66.

    Parameters
    ----------
    conversions : Pandas DataFrame
        Coefficients from Finkbeiner et al., 2016.
    g_iPS1 : float
        Apparent g - i color of the source in PS1.
    band : str
        Band in which to calculate the conversion.
    mPS1 : float
        AB apparent magnitude of the source in PS1.

    Returns
    -------
    mSDSS, float
        Estimated AB apparent magnitude of the source in SDSS.

    """
    a0 = conversions.loc[conversions['Band'] == band, 'a_0'].values[0]
    a1 = conversions.loc[conversions['Band'] == band, 'a_1'].values[0]
    a2 = conversions.loc[conversions['Band'] == band, 'a_2'].values[0]
    a3 = conversions.loc[conversions['Band'] == band, 'a_3'].values[0]
    mPS1_mSDSS = a0 + a1*g_iPS1 + a2*g_iPS1**2 + a3*g_iPS1**3
    mSDSS = mPS1 - (mPS1_mSDSS)
    return mSDSS

def calc_7DCD(df):
    """Calculates the color distance (7DCD) of objects in df from the
       stellar locus from Tonry et al., 2012.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe of PS1 objects.

    Returns
    -------
    df : Pandas DataFrame
        The same dataframe as input, with new column 7DCD.

    """
    df.replace(999.00, np.nan)
    df.replace(-999.00, np.nan)

    #read the stellar locus table from SDSS
    stream = pkg_resources.resource_stream(__name__, 'tonry_ps1_locus.txt')
    skt = Table.read(stream, format='ascii')

    gr = scinterp.interp1d(skt['ri'], skt['gr'], kind='cubic', fill_value='extrapolate')
    iz = scinterp.interp1d(skt['ri'], skt['iz'], kind='cubic', fill_value='extrapolate')
    zy = scinterp.interp1d(skt['ri'], skt['zy'], kind='cubic', fill_value='extrapolate')
    ri = np.arange(-0.4, 2.01, 0.001)

    gr_new = gr(ri)
    iz_new = iz(ri)
    zy_new = zy(ri)

    bands = ['g', 'r', 'i', 'z']

    #adding the errors in quadrature
    df["g-rErr"] =  np.sqrt(df["gApMagErr"].astype('float')**2 + df["rApMagErr"].astype('float')**2)
    df["r-iErr"] =  np.sqrt(df["rApMagErr"].astype('float')**2 + df["iApMagErr"].astype('float')**2)
    df["i-zErr"] =  np.sqrt(df["iApMagErr"].astype('float')**2 + df["zApMagErr"].astype('float')**2)
    df['z-yErr'] =  np.sqrt(df['zApMagErr'].astype('float')**2 + df['yApMagErr'].astype('float')**2)

    df["7DCD"] = np.nan
    df.reset_index(drop=True, inplace=True)
    for i in np.arange(len(df["i-z"])):
        temp_7DCD = []

        temp_7DCD_1val_gr = (df.loc[i,"g-r"] - gr_new)**2/df.loc[i, "g-rErr"]
        temp_7DCD_1val_ri = (df.loc[i,"r-i"] - ri)**2 /df.loc[i, "r-iErr"]
        temp_7DCD_1val_iz = (df.loc[i,"i-z"] - iz_new)**2/df.loc[i, "i-zErr"]
        temp_7DCD_1val_zy = (df.loc[i,"z-y"] - zy_new)**2/df.loc[i, "z-yErr"]

        temp_7DCD_1val = temp_7DCD_1val_gr + temp_7DCD_1val_ri + temp_7DCD_1val_iz + temp_7DCD_1val_zy

        df.loc[i,"7DCD"] = np.nanmin(np.array(temp_7DCD_1val))
    return df

def plotLocus(df, color=False, save=False, type="", timestamp=""):
    """Plots the color-color distribution of objects in df along with the Tonry et al., 2012
       PS1 stellar locus.

    Parameters
    ----------
    df : Pandas DataFrame
        Description of parameter `df`.
    color : bool
        If True, color objects by their distance from the stellar locus.
    save : bool
        If True, saves the image.
    type : str
        Can be "Gals" for galaxies or "Stars" for stars. Only relevant for
        coloring the distributions.
    timestamp : str
        Timestamp to append to the saved plot filename.

    """
    if color:
        plt.figure(figsize=(8,8))
        plt.scatter(df["i-z"], df["g-r"], c=df["7DCD"], s=2, alpha=0.8, norm=LogNorm())
        plt.xlim(-0.75, 1)
        plt.clim(0.1, 1000)
        plt.ylim(-0.5, 2)
        plt.xlabel("i-z")
        plt.ylabel("g-r")
        cbar = plt.colorbar()
        cbar.set_label("4D Color Distance")
        if save:
            plt.savefig("PS1_%s_StellarLocus_%s_Colored.pdf"%(type, timestamp))
        else:
            plt.show()
    else:
        #read the stellar locus table from PS1
        skt = Table.read('./tonry_ps1_locus.txt', format='ascii')

        gr = scinterp.interp1d(skt['ri'], skt['gr'], kind='cubic', fill_value='extrapolate')
        iz = scinterp.interp1d(skt['ri'], skt['iz'], kind='cubic', fill_value='extrapolate')
        zy = scinterp.interp1d(skt['ri'], skt['zy'], kind='cubic', fill_value='extrapolate')
        ri = np.arange(-0.4, 2.01, 0.001)

        gr_new = gr(ri)
        iz_new = iz(ri)
        if type == 'Gals':
            c = '#087e8b'
        elif type == "Stars":
            c = '#ffbf00'
        else:
            c = 'violet'
        g = sns.JointGrid(x="i-z", y="g-r", data=df, height=9, ratio=5, xlim=(-0.4, 0.8), ylim=(-0.2, 2.0), space=0)
        g = g.plot_joint(sns.kdeplot, color=c,gridsize=200)
        plt.plot(iz_new, gr_new, '--', c='#8c837c', lw=2)
        plt.xlabel(r"$i-z$",fontsize=18)
        plt.ylabel(r"$g-r$",fontsize=18)
        g = g.plot_marginals(sns.kdeplot, color=c, shade=True)
        if save:
            g.savefig("PS1_%s_StellarLocus_%s.pdf"%(type, timestamp))
