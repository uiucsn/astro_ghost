from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from astro_ghost.stellarLocus import *
import pickle
import importlib_resources

def separateStars_STRM(df, model_path='.', plot=False, verbose=False, starcut='gentle'):
    """Star-galaxy separation, using a random forest trained on the PS1-STRM-classified star
       and galaxy labels given in Beck et al., 2021.

    :param df: Dataframe of PS1 sources.
    :type df: Pandas DataFrame
    :param model_path: Filepath to the saved random forest model.
    :type model_path: str, optional
    :param plot: If True, shows the separated stars and galaxies in Ap - Kron vs Ap Mag space.
    :type plot: bool, optional
    :param verbose: If true, print details of the classification routine.
    :type verbose: bool, optional
    :param starcut: Labels corresponding to the classification thresholds required to classify a star as such.
        Options are 'gentle' (P>0.8), normal (P>0.5), and aggressive (P>0.3).
    :type starcut: str

    :return: A dataframe of sources classified as galaxies.
    :rtype: Pandas DataFrame
    :return: A dataframe of sources classified as stars.
    :rtype: Pandas DataFrame
    """

    #remove known galaxies and stars
    NED_stars = df[df['NED_type'].isin(['*', '**', 'WD*', '!*', '!V*', 'V*', '!Nova'])]
    NED_gals = df[df['NED_type'] == 'G']

    #label the sources we know to be stars and galaxies
    NED_stars['sourceClass'] = 1
    NED_gals['sourceClass'] = 0

    #get the remaining objects
    unsure = df[~df.index.isin(NED_stars.index)]
    unsure = unsure[~unsure.index.isin(NED_gals.index)]

    #also drop HII regions, QSOs, etc
    unsure = unsure[~unsure['NED_type'].isin(['HII', 'QSO', 'SNR', 'PN'])]

    #remove all sources with bad values for any required PS1 properties
    unsure_dropped = unsure.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    #keep track of the unknown sources with bad values - we don't want to lose them entirely
    only_na = unsure[~unsure.index.isin(unsure_dropped.index)]

    # load random forest model
    modelName = "Star_Galaxy_RealisticModel_GHOST_PS1ClassLabels.sav"
    stream = importlib_resources.files(__name__).joinpath(modelName).open("rb")
    if verbose:
        print("Loading model %s."%modelName)
    model = pickle.load(stream)

    # plot the distribution of objects before classifying
    if plot:
        sns.set_style("dark")
        sns.set_context("talk")

        plt.figure(figsize=(10,8))
        plt.plot(unsure['iApMag'], unsure['iApMag_iKronMag'], 'o', alpha=0.1)
        plt.xlabel(r"Ap Mag, $i$")
        plt.ylabel(r"Ap - Kron Mag, $i$")
        plt.savefig("TNS_NoClassificationInfo.pdf")

    if len(unsure_dropped) <1:
        if verbose:
            print("No sources in field with feature values, skipping star/galaxy separation...")
        return df, unsure_dropped
    test_X = np.asarray(unsure_dropped[['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag']])

    # define probability threshold for classification
    cutdict = {'normal':0.5, 'aggressive':0.3, 'gentle':0.8}

    try:
        test_y = model.predict_proba(test_X)[:,1] > cutdict[starcut]
    except:
        print("Error! I didn't understand your starcut option.")

    unsure_dropped['sourceClass'] = test_y
    test_stars = unsure_dropped[unsure_dropped['sourceClass'] == 1]
    test_gals = unsure_dropped[unsure_dropped['sourceClass'] == 0]

    # plot the distribution of classified stars and galaxies.
    if plot:
        c_gals = '#087e8b'
        c_stars = '#ffbf00'

        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        sns.kdeplot(test_gals['iApMag'], test_gals['iApMag_iKronMag'], n_levels=20, fill=False,label='Galaxies', color=c_gals, alpha=0.5,legend=True);
        if len(test_stars) > 1:
            sns.kdeplot(test_stars['iApMag'], test_stars['iApMag_iKronMag'], n_levels=20, fill=False,label='Stars', color=c_stars, alpha=0.5);
        plt.legend()
        plt.xlim(xmin=13,xmax=23)
        plt.ylim(ymin=-1,ymax=5)
        plt.xlabel("Ap Magnitude, $i$",fontsize=16)
        plt.ylabel("Ap - Kron Magnitude, $i$",fontsize=16)

        plt.savefig("TNS_STRM_Stars_vs_Gals_Contours.pdf")

        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = sns.distplot(test_gals['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Galaxies',kde=False,hist_kws={"color": c_gals});
        if len(test_stars) > 1:
            sns.distplot(test_stars['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Stars',kde=False,hist_kws={"color": c_stars});
        sns.distplot(unsure['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Total',kde=False,hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": "black"});

        plt.xlim(-1, 3)
        plt.xlabel("Ap - Kron Mag, $i$",fontsize=16)
        plt.xlim()
        plt.ylabel("N",fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig("TNS_iAp_iKronMag_Histogram.pdf")

    # Adding back in the ones we identified as galaxies from our clustering above, and the NED sources.

    # Also adding back in sources with photometry missing in any band. 
    #df_gals = pd.concat([test_gals, NED_gals])
    df_gals = pd.concat([test_gals, NED_gals, only_na])
    df_gals.reset_index(inplace=True, drop=True)

    # Adding back in the ones we identified as stars from our clustering above, and the NED sources.
    df_stars = pd.concat([test_stars, NED_stars])
    df_stars.reset_index(inplace=True, drop=True)
    return df_gals, df_stars

def separateStars_South(df, plot=0, verbose=0, starcut='gentle'):
    """Star-galaxy separation in the southern hemisphere, using a simple
       SkyMapper_StarClass threshold cut.

    :param df: Dataframe of PS1 sources.
    :type df: Pandas DataFrame
    :param plot: If True, shows the separated stars and galaxies in Ap - Kron vs Ap Mag space.
    :type plot: bool, optional
    :param verbose: If true, print details of the classification routine.
    :type verbose: bool, optional
    :param starcut: Labels corresponding to the classification thresholds required to classify a star as such.
        Options are 'gentle' (P>0.8), normal (P>0.5), and aggressive (P>0.3).
    :type starcut: str

    :return: A dataframe of sources classified as galaxies.
    :rtype: Pandas DataFrame
    :return: A dataframe of sources classified as stars.
    :rtype: Pandas DataFrame
    """

    #remove known galaxies and stars
    NED_stars = df[df['NED_type'].isin(['*', '**', 'WD*', '!*', '!V*', 'V*', '!Nova'])]
    NED_gals = df[df['NED_type'] == 'G']

    #get the remaining objects
    unsure = df[~df.index.isin(NED_stars.index)]
    unsure = unsure[~unsure.index.isin(NED_gals.index)]

    unsure_dropped = unsure[unsure['SkyMapper_StarClass']>0]
    only_na = unsure[~unsure.index.isin(unsure_dropped.index)]

    # define probability threshold for classification
    cutdict = {'normal':0.5, 'aggressive':0.3, 'gentle':0.8}
    try:
        test_stars = unsure_dropped[unsure_dropped['SkyMapper_StarClass']> cutdict[starcut]]
    except:
        print("Error! I didn't understand your starcut option.")

    test_gals = unsure_dropped[~unsure_dropped.index.isin(test_stars.index)]

    # Adding back in the ones we identified as galaxies from our clustering above, and the NED sources.
    df_gals = pd.concat([test_gals, NED_gals])
    df_gals['sourceClass'] = 0
    df_gals.reset_index(inplace=True, drop=True)

    # Adding back in the ones we identified as stars from our clustering above, and the NED sources.
    df_stars = pd.concat([test_stars, NED_stars])
    df_stars['sourceClass'] = 1
    df_stars.reset_index(inplace=True, drop=True)
    return df_gals, df_stars
