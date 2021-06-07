"""
|-----------------------------------------------------------------------|
|                                                                       |
|    OQproc                                                             |
|    Toolbox for processing of                                          |
|    OpenQuake outputs                                                  |
|    Version: 1.0                                                       |
|                                                                       |
|    Created on 06/04/2020                                              |
|    Update on 08/12/2020                                               |
|    Author: Volkan Ozsarac                                             |
|    Affiliation: University School for Advanced Studies IUSS Pavia     |
|    Earthquake Engineering PhD Candidate                               |
|                                                                       |
|-----------------------------------------------------------------------|
"""


def proc_hazard(poes, path_hazard_results, output_dir='Post_Outputs', rlz='hazard_curve-mean'):
    """
    Details
    -------
    This script will save hazard curves and  iml's corresponding to the desired poes
    as .txt files, and the plot the hazard curves in the same figure.

    Parameters
    ----------
    poes : list
        Probabilities of exceedance in tw years for which im levels will be obtained.
    path_hazard_results: str
        Path to the hazard results
    output_dir: str, optional
        Save outputs to a pickle file
    rlz : str, optional
        realization name to plot.

    Returns
    -------
    None.

    """

    import matplotlib.pyplot as plt
    from matplotlib import style
    import os
    import pandas as pd
    import numpy as np

    # Initialise some lists
    lat = []
    lon = []
    im = []
    s = []
    poe = []
    apoe = []
    id_no = []
    imls = []

    # Read through each file in the outputs folder
    for file in os.listdir(path_hazard_results):
        if file.startswith(rlz):

            # print(file)
            # Strip the IM out of the file name
            im_type = (file.rsplit('-')[2]).rsplit('_')[0]

            # Get the id number of the file
            idn = (file.rsplit('_')[2]).rsplit('.')[0]

            # Load the results in as a dataframe
            df = pd.read_csv(''.join([path_hazard_results, '/', file]), skiprows=1)

            # Get the column headers (but they have a 'poe-' string in them to strip out)
            iml = list(df.columns.values)[3:]  # List of headers
            iml = [float(i[4:]) for i in iml]  # Strip out the actual IM values
            f = open(''.join([path_hazard_results, '/', file]), "r")
            temp1 = f.readline().split(',')
            temp2 = list(filter(None, temp1))
            inv_t = float(temp2[5].replace(" investigation_time=", ""))
            f.close()

            # For each of the sites investigated
            for site in np.arange(len(df)):

                # Append each site's info to the output array
                lat.append([df.lat[site]][0])
                lon.append([df.lon[site]][0])
                im.append(im_type)
                s.append(iml)
                id_no.append(idn)

                # Get the array of poe in inv_t
                poe.append(df.iloc[site, 3:].values)

                # For each array of poe, convert it to annual poe
                temp = []
                for i in np.arange(len(poe[-1])):
                    temp.append(-np.log(1 - poe[-1][i]) / inv_t)
                apoe.append(temp)

    # Get intensity measure levels corresponding to poes
    for i in range(len(s)):
        plt.loglog(s[i], apoe[i], label=im[i])
        iml = get_iml(np.asarray(poes), np.asarray(apoe[i]), np.asarray(s[i]), inv_t)
        imls.append(iml)
        fname = os.path.join(output_dir, 'imls_' + im[i] + '.out')
        f = open(fname, 'w+')
        for j in iml:
            f.write("%.3f\n" % j)
        f.close()

    fname = os.path.join(output_dir, 'poes.out')
    f = open(fname, 'w+')
    for j in poes:
        f.write("%.4f\n" % j)
    f.close()

    plt.xlabel('IM [g]')
    plt.ylabel('Annual Probability of Exceedance')
    plt.legend()
    plt.grid(True)
    plt.title('Mean Hazard Curves for Lat:%s Lon:%s' % (str(lat[0]), str(lon[0])))
    fname = os.path.join(output_dir, 'Hazard_Curves.png')
    plt.savefig(fname, format='png', dpi=220)

    for i in range(len(apoe)):
        poe = 1 - (1 - np.asarray(apoe[i])) ** inv_t
        poe.shape = (len(poe), 1)
        imls = np.asarray(s[i])
        imls.shape = (len(imls), 1)
        haz_cur = np.concatenate([imls, poe], axis=1)
        fname = os.path.join(output_dir, 'HazardCurve_' + im[i] + '.out')
        np.savetxt(fname, haz_cur)


def get_iml(poes, apoe_data, iml_data, inv_t):
    """
    Details
    -------
    This script will take results of PSHA analysis, and return
    the intensity measure levels for desired probability of exceedance values

    Parameters
    ----------
    poes: list
        desired probability of exceedance values to calculate their
        corresponding intensity measure levels
    apoe_data: list
        annual probability of exceedance values
    iml_data: list
        intensity measure levels
    inv_t: int
        investigation time

    Returns
    -------
    iml: list
        intensity measure levels corresponding to poes
    """

    import numpy as np
    from scipy import interpolate

    infs = np.isinf(apoe_data)
    apoe_data = apoe_data[~infs]
    iml_data = iml_data[~infs]
    nans = np.isnan(apoe_data)
    apoe_data = apoe_data[~nans]
    iml_data = iml_data[~nans]

    Ninterp = 1e5
    iml_range = np.arange(min(iml_data), max(iml_data), (max(iml_data) - min(iml_data)) / Ninterp)
    apoe_fit = interpolate.interp1d(iml_data, apoe_data, kind='quadratic')(iml_range)
    poe = 1 - (1 - apoe_fit) ** inv_t

    idxs = []
    for i in range(len(poes)):
        temp = abs(poe - poes[i]).tolist()
        idxs.append(temp.index(min(temp)))
        # These are actual points where the analysis are carried out and losses are calculated for
    iml = iml_range[idxs]

    return iml


def proc_disagg_MR(Mbin, dbin, poe_disagg, path_disagg_results, output_dir='Post_Outputs', n_rows=1):
    """
    Details
    -------
    This script will save disaggregation plots
    including M and R.

    Parameters
    ----------
    Mbin : int, float
        magnitude bin used in disaggregation.
    dbin : int, float
        distance bin used in disaggregation.
    poe_disagg : list
        disaggregation probability of exceedances
    path_disagg_results: str
        Path to the disaggregation results
    output_dir: str, optional
        Save outputs to a pickle file
    n_rows : int, optional
        total number of rows for subplots.

    Returns
    -------
    None.

    """
    # lets add the plotting options to make everything clearer
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm  # import colormap
    from matplotlib import style  # import syle
    import os
    import numpy as np
    import math
    import pandas as pd

    cmap = cm.get_cmap('jet')  # Get desired colormap
    lat = []
    lon = []
    modeLst, meanLst = [], []
    im = []
    poe = []
    Tr = []
    apoe_norm = []
    M, R = [], []

    for file in os.listdir(path_disagg_results):
        if file.startswith('rlz') and file.find('Mag_Dist') > 0 > file.find('Mag_Dist_Eps'):
            # Load the dataframe
            df = pd.read_csv(''.join([path_disagg_results, '/', file]), skiprows=1)

            # Strip the IM out of the file name
            im.append(file.rsplit('-')[2])

            # Get some salient values
            f = open(''.join([path_disagg_results, '/', file]), "r")
            ff = f.readline().split(',')
            try:  # for OQ version <3.11
                inv_t = float(ff[8].replace(" investigation_time=", ""))
                poe.append(float(ff[11].replace(" poe=", "").replace("'", "")))
            except:  # for OQ version 3.11
                inv_t = float(ff[5].replace(" investigation_time=", ""))
                poe.append(float(ff[-1].replace(" poe=", "").replace("\"", "").replace("\n", "")))
            lon.append(float(ff[9].replace(" lon=", "")))
            lat.append(float(ff[10].replace(" lat=", "")))
            Tr.append(-inv_t / np.log(1 - poe[-1]))

            # Extract the poe and annualise
            df['apoe'] = -np.log(1 - df['poe']) / inv_t

            # Normalise the apoe for disaggregation plotting
            df['apoe_norm'] = df['apoe'] / df['apoe'].sum()
            apoe_norm.append(df['apoe_norm'])

            # Compute the modal value (highest apoe)
            mode = df.sort_values(by='apoe_norm', ascending=False)[0:1]
            modeLst.append([mode['mag'].values[0], mode['dist'].values[0]])

            # Compute the mean value
            meanLst.append([np.sum(df['mag'] * df['apoe_norm']), np.sum(df['dist'] * df['apoe_norm'])])

            # Report the individual mangnitude and distance bins
            M.append(df['mag'])
            R.append(df['dist'])

    lon = [x for _, x in sorted(zip(Tr, lon))]
    lat = [x for _, x in sorted(zip(Tr, lat))]
    im = [x for _, x in sorted(zip(Tr, im))]
    M = [x for _, x in sorted(zip(Tr, M))]
    R = [x for _, x in sorted(zip(Tr, R))]
    apoe_norm = [x for _, x in sorted(zip(Tr, apoe_norm))]
    modeLst = [x for _, x in sorted(zip(Tr, modeLst))]
    meanLst = [x for _, x in sorted(zip(Tr, meanLst))]

    Tr = -inv_t / np.log(1 - np.asarray(poe_disagg))
    n_Tr = len(np.unique(np.asarray(Tr)))
    Tr = sorted(Tr)
    ims = np.unique(im)
    n_im = len(ims)

    lon = lon[0]
    lat = lat[0]

    mags = []
    dists = []

    n_cols = math.floor(n_Tr / n_rows)
    if np.mod(n_Tr, n_rows):
        n_cols += 1

    for idx1 in range(n_im):
        fig = plt.figure(figsize=(19.2, 10.8))
        for idx2 in range(n_Tr):
            i = idx1 * n_Tr + idx2
            ax1 = fig.add_subplot(n_rows, n_cols, idx2 + 1, projection='3d')

            X = R[i]
            Y = M[i]
            Z = np.zeros(len(X))

            dx = np.ones(len(X)) * dbin / 2
            dy = np.ones(len(X)) * Mbin / 2
            dz = apoe_norm[i] * 100

            # here we may make the colormap based on epsilon instead of hazard contribution
            max_height = np.max(dz)  # get range of colorbars so we can normalize
            min_height = np.min(dz)
            # scale each z to [0,1], and get their rgb values
            rgba = [cmap((k - min_height) / max_height) for k in dz]
            ax1.bar3d(X, Y, Z, dx, dy, dz, color=rgba, zsort='average', alpha=0.7, shade=True)

            ax1.set_xlabel('R [km]')
            ax1.set_ylabel('$M_{w}$')
            if np.mod(idx2 + 1, n_cols) == 1:
                ax1.set_zlabel('Hazard Contribution [%]')
                ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
                ax1.set_zlabel('Hazard Contribution [%]', rotation=90)
            ax1.zaxis._axinfo['juggled'] = (1, 2, 0)

            plt.title('$T_{R}$=%s years\n$M_{mod}$=%s, $R_{mod}$=%s km\n$M_{mean}$=%s, $R_{mean}$=%s km'
                      % ("{:.0f}".format(Tr[idx2]), "{:.2f}".format(modeLst[i][0]), "{:.0f}".format(modeLst[i][1]),
                         "{:.2f}".format(meanLst[i][0]), "{:.0f}".format(meanLst[i][1])),
                      fontsize=11, loc='right', verticalalignment='top')

            mags.append(meanLst[i][0])
            dists.append(meanLst[i][1])

        plt.subplots_adjust(hspace=0.1, wspace=0.05)  # adjust the subplot to the right for the legend
        fig.suptitle('Disaggregation of Seismic Hazard\nIntensity Measure: %s\nLatitude: %s, Longitude: %s' % (
            ims[idx1], "{:.4f}".format(lat), "{:.4f}".format(lon)), fontsize=14, weight='bold', ha='left', x=0.12,
                     y=0.97)

        fname = os.path.join(output_dir, 'Disaggregation_MR_' + ims[idx1] + '.png')
        plt.savefig(fname, format='png', dpi=220)

        fname = os.path.join(output_dir, 'mean_mags_' + ims[idx1] + '.out')
        np.savetxt(fname, np.asarray(mags), fmt='%.2f')
        fname = os.path.join(output_dir, 'mean_dists_' + ims[idx1] + '.out')
        np.savetxt(fname, np.asarray(dists), fmt='%.1f')


def proc_disagg_MReps(Mbin, dbin, poe_disagg, path_disagg_results, output_dir='Post_Outputs', n_rows=1):
    """
    Details
    -------
    This script will save disaggregation plots
    including M and R.

    Parameters
    ----------
    Mbin : int, float
        magnitude bin used in disaggregation.
    dbin : int, float
        distance bin used in disaggregation.
    poe_disagg : list
        disaggregation probability of exceedances
    path_disagg_results: str
        Path to the hazard results
    output_dir: str, optional
        Save outputs to a pickle file
    n_rows : int, optional
        total number of rows for subplots.

    Returns
    -------
    None.

    """
    # lets add the plotting options to make everything clearer
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm  # import colormap
    from matplotlib import style  # import syle
    from matplotlib.patches import Patch
    import os
    import numpy as np
    import math
    import pandas as pd

    cmap = cm.get_cmap('jet')  # Get desired colormap
    lat = []
    lon = []
    modeLst, meanLst = [], []
    im = []
    poe = []
    Tr = []
    apoe_norm = []
    M, R, eps = [], [], []
    mags = []
    dists = []

    for file in os.listdir(path_disagg_results):
        if file.startswith('rlz') and file.find('Mag_Dist_Eps') > 0:
            # Load the dataframe
            df = pd.read_csv(''.join([path_disagg_results, '/', file]), skiprows=1)

            # Strip the IM out of the file name
            im.append(file.rsplit('-')[2])

            # Get some salient values
            f = open(''.join([path_disagg_results, '/', file]), "r")
            ff = f.readline().split(',')
            try:  # for OQ version <3.11
                inv_t = float(ff[9].replace(" investigation_time=", ""))
                poe.append(float(ff[12].replace(" poe=", "").replace("'", "")))
            except:  # TODO-1: verify this for OQ version 3.11
                inv_t = float(ff[6].replace(" investigation_time=", ""))
                poe.append(float(ff[-1].replace(" poe=", "").replace("\"", "").replace("\n", "")))
            lon.append(float(ff[10].replace(" lon=", "")))
            lat.append(float(ff[11].replace(" lat=", "")))
            Tr.append(-inv_t / np.log(1 - poe[-1]))

            # Extract the poe and annualise
            df['apoe'] = -np.log(1 - df['poe']) / inv_t

            # Normalise the apoe for disaggregation plotting
            df['apoe_norm'] = df['apoe'] / df['apoe'].sum()
            apoe_norm.append(df['apoe_norm'])

            # Compute the modal value (highest apoe)
            mode = df.sort_values(by='apoe_norm', ascending=False)[0:1]
            modeLst.append([mode['mag'].values[0], mode['dist'].values[0], mode['eps'].values[0]])

            # Compute the mean value
            meanLst.append([np.sum(df['mag'] * df['apoe_norm']), np.sum(df['dist'] * df['apoe_norm']),
                            np.sum(df['eps'] * df['apoe_norm'])])

            M.append(df['mag'])
            R.append(df['dist'])
            eps.append(df['eps'])

    lon = [x for _, x in sorted(zip(Tr, lon))]
    lat = [x for _, x in sorted(zip(Tr, lat))]
    im = [x for _, x in sorted(zip(Tr, im))]
    M = [x for _, x in sorted(zip(Tr, M))]
    R = [x for _, x in sorted(zip(Tr, R))]
    eps = [x for _, x in sorted(zip(Tr, eps))]
    apoe_norm = [x for _, x in sorted(zip(Tr, apoe_norm))]
    modeLst = [x for _, x in sorted(zip(Tr, modeLst))]
    meanLst = [x for _, x in sorted(zip(Tr, meanLst))]

    Tr = -inv_t / np.log(1 - np.asarray(poe_disagg))
    n_Tr = len(np.unique(np.asarray(Tr)))
    Tr = sorted(Tr)
    ims = np.unique(im)
    n_im = len(ims)
    n_eps = len(np.unique(np.asarray(eps)))
    min_eps = np.min(np.unique(np.asarray(eps)))  # get range of colorbars so we can normalize
    max_eps = np.max(np.unique(np.asarray(eps)))

    lon = lon[0]
    lat = lat[0]

    n_cols = math.floor(n_Tr / n_rows)
    if np.mod(n_Tr, n_rows):
        n_cols += 1

    for idx1 in range(n_im):
        fig = plt.figure(figsize=(19.2, 10.8))
        for idx2 in range(n_Tr):
            i = idx1 * n_Tr + idx2
            ax1 = fig.add_subplot(n_rows, n_cols, idx2 + 1, projection='3d')

            # scale each eps to [0,1], and get their rgb values
            rgba = [cmap((k - min_eps) / max_eps / 2) for k in (np.unique(np.asarray(eps)))]
            num_triads_M_R_eps = len(R[i])
            Z = np.zeros(int(num_triads_M_R_eps / n_eps))

            for l in range(n_eps):
                X = np.array(R[i][np.arange(l, num_triads_M_R_eps, n_eps)])
                Y = np.array(M[i][np.arange(l, num_triads_M_R_eps, n_eps)])

                dx = np.ones(int(num_triads_M_R_eps / n_eps)) * dbin / 2
                dy = np.ones(int(num_triads_M_R_eps / n_eps)) * Mbin / 2
                dz = np.array(apoe_norm[i][np.arange(l, num_triads_M_R_eps, n_eps)]) * 100

                ax1.bar3d(X, Y, Z, dx, dy, dz, color=rgba[l], zsort='average', alpha=0.7, shade=True)
                Z += dz  # add the height of each bar to know where to start the next

            ax1.set_xlabel('R [km]')
            ax1.set_ylabel('$M_{w}$')
            if np.mod(idx2 + 1, n_cols) == 1:
                ax1.set_zlabel('Hazard Contribution [%]')
                ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
                ax1.set_zlabel('Hazard Contribution [%]', rotation=90)
            ax1.zaxis._axinfo['juggled'] = (1, 2, 0)

            plt.title(
                '$T_{R}$=%s years\n$M_{mod}$=%s, $R_{mod}$=%s km, $\epsilon_{mod}$=%s\n$M_{mean}$=%s, $R_{mean}$=%s '
                'km, $\epsilon_{mean}$=%s'
                % ("{:.0f}".format(Tr[i]), "{:.2f}".format(modeLst[i][0]), "{:.0f}".format(modeLst[i][1]),
                   "{:.1f}".format(modeLst[i][2]),
                   "{:.2f}".format(meanLst[i][0]), "{:.0f}".format(meanLst[i][1]), "{:.1f}".format(meanLst[i][2])),
                fontsize=11, loc='right', va='top')

            mags.append(meanLst[i][0])
            dists.append(meanLst[i][1])

        legend_elements = []
        for j in range(n_eps):
            legend_elements.append(Patch(facecolor=rgba[n_eps - j - 1],
                                         label='\u03B5 = %.2f' % (np.unique(np.asarray(eps))[n_eps - j - 1])))

        fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.05), borderaxespad=0.,
                   ncol=n_eps)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)  # adjust the subplot to the right for the legend
        fig.suptitle('Disaggregation of Seismic Hazard\nIntensity Measure: %s\nLatitude: %s, Longitude: %s' % (
            ims[idx1], "{:.4f}".format(lat), "{:.4f}".format(lon)), fontsize=14, weight='bold', ha='left', x=0.12,
                     y=0.97)

        fname = os.path.join(output_dir, 'Disaggregation_MReps_' + ims[idx1] + '.png')
        plt.savefig(fname, format='png', dpi=220)

        fname = os.path.join(output_dir, 'mean_mags_' + ims[idx1] + '.out')
        np.savetxt(fname, np.asarray(mags), fmt='%.2f')
        fname = os.path.join(output_dir, 'mean_dists_' + ims[idx1] + '.out')
        np.savetxt(fname, np.asarray(dists), fmt='%.1f')
