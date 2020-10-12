from sklearn import svm
import seaborn as sns
#from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from astro_ghost.stellarLocus import *
import os
import pickle
import pkg_resources

#10/11/2020: a new star/galaxy separation model, this time using a random forest classifier saved with hyperparameters chosen with randomGridSearch
def separateStars_STRM(df, model_path='.', plot=0, verbose=0, starcut='gentle'):
    df_dropped = df.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    only_na = df[~df.index.isin(df_dropped.index)]
    unsure = df_dropped.reset_index(drop=True)

    modelName = "Star_Galaxy_RealisticModel_GHOST_PS1ClassLabels.sav"
    stream = pkg_resources.resource_stream(__name__, modelName)
    if verbose:
        print("Loading model %s."%modelName)
    model = pickle.load(stream)

    if plot:
        sns.set_style("dark")
        sns.set_context("talk")

        plt.figure(figsize=(10,8))
        plt.plot(unsure['iApMag'], unsure['iApMag_iKronMag'], 'o', alpha=0.1)
        plt.xlabel(r"Ap Mag, $i$")
        plt.ylabel(r"Ap - Kron Mag, $i$")
        plt.savefig("TNS_NoClassificationInfo.pdf")

    #NED_stars_nona = NED_stars.dropna(subset=['iApMag','iApMag_iKronMag'])
    #NED_gals_nona = NED_gals.dropna(subset=['iApMag','iApMag_iKronMag'])

    unsure = unsure.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    if len(unsure) <1:
        if verbose:
            print("No sources in field with feature values, skipping star/galaxy separation...")
        return df, unsure
    test_X = np.matrix(unsure[['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag']])
    if starcut is 'normal':
        test_y = model.predict(test_X)
    elif starcut is 'aggressive':
        test_y = model.predict_proba(test_X)[:,1] > 0.3
    elif starcut is 'gentle':
        test_y = model.predict_proba(test_X)[:,1] > 0.8
    else:
        print("Error! I didn't understand your starcut option.")
    unsure['class'] = test_y
    test_stars = unsure[unsure['class'] == 1]
    test_gals = unsure[unsure['class'] == 0]

    if plot:
        c_gals = '#087e8b'
        c_stars = '#ffbf00'

        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        sns.kdeplot(test_gals['iApMag'], test_gals['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Galaxies', color=c_gals, alpha=0.5,legend=True);
        if len(test_stars) > 1:
            sns.kdeplot(test_stars['iApMag'], test_stars['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Stars', color=c_stars, alpha=0.5);
        plt.legend()
        plt.xlim(xmin=13,xmax=23)
        plt.ylim(ymin=-1,ymax=5)
        plt.xlabel("Ap Magnitude, $i$",fontsize=16)
        plt.ylabel("Ap - Kron Magnitude, $i$",fontsize=16)
        #plt.legend(fontsize=16)
        plt.savefig("TNS_STRM_Stars_vs_Gals_RF_Contours.pdf")

        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = sns.distplot(test_gals['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Galaxies',kde=False,hist_kws={"color": c_gals});
        if len(test_stars) > 1:
            sns.distplot(test_stars['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Stars',kde=False,hist_kws={"color": c_stars});
        sns.distplot(unsure['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Total',kde=False,hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": "black"});
        #ax.set_yscale('log')
        plt.xlim(-1, 3)
        plt.xlabel("Ap - Kron Mag, $i$",fontsize=16)
        plt.xlim()
        plt.ylabel("N",fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig("TNS_iAp_iKronMag_Histogram_RF.pdf")

    # removing the ones where we have no NED identification
    df = df[~df['NED_type'].isnull()]

    # now we want to add back in the ones we identified as galaxies from our clustering above:
    df_gals = pd.concat([test_gals, only_na])
    df_gals.reset_index(inplace=True, drop=True)
    df_stars = test_stars.drop_duplicates()
    #df_stars.to_csv('OSC_061019_PS1_stars_109_NEDCuts.tar.gz')
    #df_gals.to_csv('TNS_PS1_gals_109_NEDCuts.tar.gz')
    return df_gals, df_stars#

#10/06/2020: a new star/galaxy separation model, this time using a random forest classifier saved with hyperparameters chosen with randomGridSearch
def separateStars_RF(df, model_path='.', plot=0, verbose=0):
    df.replace('', np.nan, inplace=True)
    df_dropped = df.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    only_na = df[~df.index.isin(df_dropped.index)]
    df = df_dropped.reset_index(drop=True)

    NED_stars = df[df['NED_type'] == '*']
    if verbose:
        print("Found %s NED-identified stars in the candidate list!"%len(NED_stars))
    NED_gals = df[df['NED_type'] == 'G']
    if verbose:
        print("Found %s NED-identified galaxies in the candidate list!"%len(NED_gals))
    NED_unsure= df[df['NED_type'].isnull()]

    modelName = "Star_Galaxy_RealisticModel.sav"
    stream = pkg_resources.resource_stream(__name__, modelName)
    model = pickle.load(stream)

    if plot:
        sns.set_style("dark")
        sns.set_context("talk")

        plt.figure(figsize=(10,8))
        plt.plot(NED_unsure['iApMag'], NED_unsure['iApMag_iKronMag'], 'o', alpha=0.1)
        plt.xlabel(r"Ap Mag, $i$")
        plt.ylabel(r"Ap - Kron Mag, $i$")
        plt.savefig("TNS_NoNedInfo.pdf")

    NED_stars_nona = NED_stars.dropna(subset=['iApMag','iApMag_iKronMag'])
    NED_gals_nona = NED_gals.dropna(subset=['iApMag','iApMag_iKronMag'])

    if plot:
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        sns.kdeplot(NED_gals_nona['iApMag'], NED_gals_nona['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Galaxies', alpha=0.5,legend=True);
        sns.kdeplot(NED_stars_nona['iApMag'], NED_stars_nona['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Stars', alpha=0.5);
        plt.legend()
        plt.xlim(xmin=12,xmax=24)
        plt.ylim(ymin=-1,ymax=4)
        plt.xlabel("Ap Magnitude, $i$",fontsize=16)
        plt.ylabel("Ap - Kron Magnitude, $i$",fontsize=16)
        plt.savefig("TNS_NED_Stars_vs_Gals_Contours.pdf")

        plt.figure(figsize=(10,8))
        plt.plot(NED_gals['7DCD'], NED_gals['iApMag_iKronMag'],'o', alpha=0.1, color='k',label='NED Galaxies')
        plt.plot(NED_stars['7DCD'], NED_stars['iApMag_iKronMag'],'o', alpha=0.1, color='r', label='NED Stars')
        plt.xlabel(r"7DCD, $i$")
        plt.xscale("log")
        plt.ylabel(r"Ap - Kron Mag, $i$")
        plt.legend(fontsize=20)
        plt.savefig("TNS_NED_Stars_vs_Gals_v7DCD.pdf")

    NED_unsure = NED_unsure.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    NED_test_X = np.matrix(NED_unsure[['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag']])

    if verbose:
        print("Running star-galaxy separator on %s sources...."%len(NED_unsure))

    NED_test_y = model.predict(NED_test_X)
    NED_unsure['class'] = NED_test_y
    NED_test_stars = NED_unsure[NED_unsure['class'] == 1]
    NED_test_gals = NED_unsure[NED_unsure['class'] == 0]

    if plot:
        c_gals = '#087e8b'
        c_stars = '#ffbf00'

        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        sns.kdeplot(NED_test_gals['iApMag'], NED_test_gals['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Galaxies', color=c_gals, alpha=0.5,legend=True);
        if len(NED_test_stars) > 1:
            sns.kdeplot(NED_test_stars['iApMag'], NED_test_stars['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Stars', color=c_stars, alpha=0.5);
        plt.legend()
        plt.xlim(xmin=13,xmax=23)
        plt.ylim(ymin=-1,ymax=5)
        plt.xlabel("Ap Magnitude, $i$",fontsize=16)
        plt.ylabel("Ap - Kron Magnitude, $i$",fontsize=16)
        #plt.legend(fontsize=16)
        plt.savefig("TNS_NED_Stars_vs_Gals_SVM_Contours.pdf")

        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = sns.distplot(NED_test_gals['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Galaxies',kde=False,hist_kws={"color": c_gals});
        if len(NED_test_stars) > 1:
            sns.distplot(NED_test_stars['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Stars',kde=False,hist_kws={"color": c_stars});
        sns.distplot(NED_unsure['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Total',kde=False,hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": "black"});
        #ax.set_yscale('log')
        plt.xlim(-1, 3)
        plt.xlabel("Ap - Kron Mag, $i$",fontsize=16)
        plt.xlim()
        plt.ylabel("N",fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig("TNS_iAp_iKronMag_Histogram_SVM_NEDInfo.pdf")

    # removing the ones where we have no NED identification
    df = df[~df['NED_type'].isnull()]

    # now we want to add back in the ones we identified as galaxies from our clustering above:
    df_gals = pd.concat([df, NED_test_gals, only_na])

    #removing some objects we know not to be our hosts
    df_gals = df_gals[df_gals['NED_type'] != 'HII']
    df_gals = df_gals[df_gals['NED_type'] != '*']
    df_gals = df_gals[df_gals['NED_type'] != '**']
    df_gals = df_gals[df_gals['NED_type'] != 'QSO']
    df_gals = df_gals[df_gals['NED_type'] != 'SNR']
    df_gals = df_gals[df_gals['NED_type'] != 'WD*']
    df_gals = df_gals[df_gals['NED_type'] != '!*']
    df_gals = df_gals[df_gals['NED_type'] != '!V*']
    df_gals = df_gals[df_gals['NED_type'] != 'V*']
    df_gals = df_gals[df_gals['NED_type'] != 'PN']
    df_gals = df_gals[df_gals['NED_type'] != '!Nova']

    #df = df[df['NED_type'] != 'PofG']

    df_gals.reset_index(inplace=True, drop=True)
    df_stars = pd.concat([NED_test_stars, NED_stars]).drop_duplicates()
    #df_stars.to_csv('OSC_061019_PS1_stars_109_NEDCuts.tar.gz')
    #df_gals.to_csv('TNS_PS1_gals_109_NEDCuts.tar.gz')
    return df_gals, df_stars#


def separateStars(df, plot=0):
    df_dropped = df.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    only_na = df[~df.index.isin(df_dropped.index)]
    df = df_dropped.reset_index(drop=True)

    NED_stars = df[df['NED_type'] == '*']
    NED_gals = df[df['NED_type'] == 'G']
    NED_unsure= df[df['NED_type'].isnull()]

    if plot:
        sns.set_style("dark")
        sns.set_context("talk")

        plt.figure(figsize=(10,8))
        plt.plot(NED_unsure['iApMag'], NED_unsure['iApMag_iKronMag'], 'o', alpha=0.1)
        plt.xlabel(r"Ap Mag, $i$")
        plt.ylabel(r"Ap - Kron Mag, $i$")
        plt.savefig("TNS_NoNedInfo.pdf")

    NED_stars['class'] = 1
    NED_stars = NED_stars[NED_stars['iApMag_iKronMag'] < 1.0]
    NED_gals['class'] = 0

    NED_stars_nona = NED_stars.dropna(subset=['iApMag','iApMag_iKronMag'])
    NED_gals_nona = NED_gals.dropna(subset=['iApMag','iApMag_iKronMag'])

    if plot:
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        sns.kdeplot(NED_gals_nona['iApMag'], NED_gals_nona['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Galaxies', alpha=0.5,legend=True);
        sns.kdeplot(NED_stars_nona['iApMag'], NED_stars_nona['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Stars', alpha=0.5);
        plt.legend()
        plt.xlim(xmin=12,xmax=24)
        plt.ylim(ymin=-1,ymax=4)
        plt.xlabel("Ap Magnitude, $i$",fontsize=16)
        plt.ylabel("Ap - Kron Magnitude, $i$",fontsize=16)
        plt.savefig("TNS_NED_Stars_vs_Gals_Contours.pdf")

        plt.figure(figsize=(10,8))
        plt.plot(NED_gals['7DCD'], NED_gals['iApMag_iKronMag'],'o', alpha=0.1, color='k',label='NED Galaxies')
        plt.plot(NED_stars['7DCD'], NED_stars['iApMag_iKronMag'],'o', alpha=0.1, color='r', label='NED Stars')
        plt.xlabel(r"7DCD, $i$")
        plt.xscale("log")
        plt.ylabel(r"Ap - Kron Mag, $i$")
        plt.legend(fontsize=20)
        plt.savefig("TNS_NED_Stars_vs_Gals_v7DCD.pdf")

    # create an SVM model to predict our stars and our galaxies!
    clf = svm.SVC()

    gals_OSC = pd.read_csv("/Users/alexgagliano/Documents/Research/Transient_ML_Box/tables/SVM_training_gals.tar.gz")
    stars_OSC = pd.read_csv("/Users/alexgagliano/Documents/Research/Transient_ML_Box/tables/SVM_training_stars.tar.gz")

    if (len(NED_stars) > 500) and (len(NED_gals) > 500):
        # get rid of the outliers !
        NED_stars = NED_stars[NED_stars['iApMag_iKronMag'] < 1.0]
        NED_stars['class'] = 1
        NED_gals['class'] = 0
        NED_training = pd.concat([NED_stars, NED_gals])
    else:
        NED_training = pd.concat([stars_OSC,gals_OSC])

    NED_training = NED_training.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    NED_training_X = np.matrix(NED_training[['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag']])
    NED_training_y = np.array(NED_training['class'])
    clf.fit(NED_training_X, NED_training_y)

    NED_unsure = NED_unsure.dropna(subset=['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag'])
    NED_test_X = np.matrix(NED_unsure[['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag']])
    NED_test_y = clf.predict(NED_test_X)
    NED_unsure['class'] = NED_test_y
    NED_test_stars = NED_unsure[NED_unsure['class'] == 1]
    NED_test_gals = NED_unsure[NED_unsure['class'] == 0]

    if plot:
        c_gals = '#087e8b'
        c_stars = '#ffbf00'

        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        sns.kdeplot(NED_test_gals['iApMag'], NED_test_gals['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Galaxies', color=c_gals, alpha=0.5,legend=True);
        if len(NED_test_stars) > 1:
            sns.kdeplot(NED_test_stars['iApMag'], NED_test_stars['iApMag_iKronMag'], n_levels=20, shade_lowest=False, shade=False,label='Stars', color=c_stars, alpha=0.5);
        plt.legend()
        plt.xlim(xmin=13,xmax=23)
        plt.ylim(ymin=-1,ymax=5)
        plt.xlabel("Ap Magnitude, $i$",fontsize=16)
        plt.ylabel("Ap - Kron Magnitude, $i$",fontsize=16)
        #plt.legend(fontsize=16)
        plt.savefig("TNS_NED_Stars_vs_Gals_SVM_Contours.pdf")

        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = sns.distplot(NED_test_gals['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Galaxies',kde=False,hist_kws={"color": c_gals});
        if len(NED_test_stars) > 1:
            sns.distplot(NED_test_stars['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Stars',kde=False,hist_kws={"color": c_stars});
        sns.distplot(NED_unsure['iApMag_iKronMag'],bins=np.linspace(-1, 7, num=100),label='Total',kde=False,hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1, "color": "black"});
        #ax.set_yscale('log')
        plt.xlim(-1, 3)
        plt.xlabel("Ap - Kron Mag, $i$",fontsize=16)
        plt.xlim()
        plt.ylabel("N",fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig("TNS_iAp_iKronMag_Histogram_SVM_NEDInfo.pdf")

    # removing the ones where we have no NED identification
    df = df[~df['NED_type'].isnull()]

    # now we want to add back in the ones we identified as galaxies from our clustering above:
    df_gals = pd.concat([df, NED_test_gals, only_na])

    #removing some objects we know not to be our hosts
    df_gals = df_gals[df_gals['NED_type'] != 'HII']
    df_gals = df_gals[df_gals['NED_type'] != '*']
    df_gals = df_gals[df_gals['NED_type'] != '**']
    df_gals = df_gals[df_gals['NED_type'] != 'QSO']
    df_gals = df_gals[df_gals['NED_type'] != 'SNR']
    df_gals = df_gals[df_gals['NED_type'] != 'WD*']
    df_gals = df_gals[df_gals['NED_type'] != '!*']
    df_gals = df_gals[df_gals['NED_type'] != '!V*']
    df_gals = df_gals[df_gals['NED_type'] != 'V*']
    df_gals = df_gals[df_gals['NED_type'] != 'PN']
    df_gals = df_gals[df_gals['NED_type'] != '!Nova']

    #df = df[df['NED_type'] != 'PofG']

    df_gals.reset_index(inplace=True, drop=True)
    df_stars = pd.concat([NED_test_stars, NED_stars]).drop_duplicates()
    #df_stars.to_csv('OSC_061019_PS1_stars_109_NEDCuts.tar.gz')
    #df_gals.to_csv('TNS_PS1_gals_109_NEDCuts.tar.gz')
    return df_gals, df_stars#
