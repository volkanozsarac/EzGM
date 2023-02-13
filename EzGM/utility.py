"""
Utility functions
"""

# Import python libraries
import zipfile
import difflib
import os
import sys
import errno
import shutil
import stat
from copy import deepcopy
import re
from time import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.stats import norm, qmc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.patches import Patch
import requests
import json
from openquake.hazardlib import gsim, nrml
from openquake.baselib.node import Node


# FUNCTIONS TO POST-PROCESS OPENQUAKE PSHA RESULTS
# ---------------------------------------------------------------------

def hazard_curve(poes, path_hazard_results, output_dir='Post_Outputs', filename='hazard_curve-mean', show=1):
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
        Path to the hazard results.
    output_dir: str, optional
        Save outputs to a pickle file.
    filename : str, optional
        filename to process.
    show: int
        flag to show figure (1 or 0)

    Returns
    -------
    None.
    """

    # Initialise some lists
    lat = []
    lon = []
    im = []
    iml_data = []
    poe_data = []
    id_no = []
    imls = []

    # Read through each file in the outputs folder
    for file in os.listdir(path_hazard_results):
        if file.startswith(filename):

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
                id_no.append(idn)

                # Get the array of poe in inv_t and corresponding imls
                tmp1 = np.array(df.iloc[site, 3:].values)
                tmp2 = np.array(iml)
                # get rid of any infinite or nan value
                infs = np.isinf(tmp1)
                tmp1 = tmp1[~infs]
                tmp2 = tmp2[~infs]
                nans = np.isnan(tmp1)
                tmp1 = tmp1[~nans]
                tmp2 = tmp2[~nans]
                # append
                poe_data.append(tmp1)
                iml_data.append(tmp2)

    # Get intensity measure levels corresponding to poes
    fig = plt.figure()
    for i in range(len(iml_data)):
        plt.loglog(iml_data[i], poe_data[i], label=im[i])
        iml = interpolate.interp1d(poe_data[i], iml_data[i], kind='linear')(poes)
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
    plt.ylabel(f'Probability of Exceedance in {inv_t:.0f} years')
    plt.legend()
    plt.grid(True)
    plt.title(f"Mean Hazard Curves for Lat:{lat[0]} Lon:{lon[0]}")
    plt.tight_layout()
    fname = os.path.join(output_dir, 'Hazard_Curves.png')
    plt.savefig(fname, format='png', dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    for i, poe in enumerate(poe_data):
        poe.shape = (len(poe), 1)
        imls = np.asarray(iml_data[i])
        imls.shape = (len(imls), 1)
        haz_cur = np.concatenate([imls, poe], axis=1)
        fname = os.path.join(output_dir, 'HazardCurve_' + im[i] + '.out')
        np.savetxt(fname, haz_cur)


def disagg_MR(Mbin, dbin, path_disagg_results, output_dir='Post_Outputs', n_rows=1, filename='Mag_Dist', show=1):
    """
    Details
    -------
    This script will save disaggregation plots including M and R.

    Parameters
    ----------
    Mbin : int, float
        magnitude bin used in disaggregation.
    dbin : int, float
        distance bin used in disaggregation.
    path_disagg_results: str
        Path to the disaggregation results.
    output_dir: str, optional
        Save outputs to a pickle file.
    n_rows : int, optional
        total number of rows for subplots.
    filename : str, optional
        filename to process.
    show: int
        flag to show figure (1 or 0)

    Returns
    -------
    None.
    """

    # lets add the plotting options to make everything clearer
    cmap = cm.get_cmap('jet')  # Get desired colormap

    for file in os.listdir(path_disagg_results):
        if file.startswith(filename) and 'Mag_Dist_Eps' not in file:
            # Load the dataframe
            df = pd.read_csv(''.join([path_disagg_results, '/', file]), skiprows=1)
            poes = np.unique(df['poe']).tolist()
            poes.sort(reverse=True)
            # Get some salient values
            f = open(''.join([path_disagg_results, '/', file]), "r")
            ff = f.readline().split(',')
            lon = float(ff[-2].replace(" lon=", ""))
            lat = float(ff[-1].replace(" lat=", "").replace("\"\n", ""))
            ims = np.unique(df['imt'])
            inv_t = float(ff[7].replace(" investigation_time=", ""))
            for imt in ims:
                M, R = [], []
                hz_cont = []
                Tr = []
                modeLst, meanLst = [], []
                for poe in poes:
                    Tr.append(round(-inv_t / np.log(1 - poe)))
                    data = {}
                    data['mag'] = df['mag'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['dist'] = df['dist'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['hz_cont'] = df.iloc[:, 4][(df['poe'] == poe) & (df['imt'] == imt)]
                    hz_cont.append(data['hz_cont'] / data['hz_cont'].sum())
                    data['hz_cont'] = hz_cont[-1]
                    data = pd.DataFrame(data)
                    # Compute the modal values (highest hazard contribution)
                    mode = data.sort_values(by='hz_cont', ascending=False)[0:1]
                    modeLst.append([mode['mag'].values[0], mode['dist'].values[0]])
                    # Compute the mean value
                    meanLst.append([np.sum(data['mag'] * data['hz_cont']), np.sum(data['dist'] * data['hz_cont'])])

                    # Report the individual magnitude and distance bins
                    M.append(data['mag'])
                    R.append(data['dist'])

                n_Tr = len(Tr)
                mean_mags = []
                mean_dists = []
                mod_mags = []
                mod_dists = []

                n_cols = int(np.floor(n_Tr / n_rows))
                if np.mod(n_Tr, n_rows):
                    n_cols += 1

                fig = plt.figure(figsize=(19.2, 10.8))
                for i in range(n_Tr):
                    # Save disaggregation results
                    disagg_results = np.array([M[i], R[i], hz_cont[i]]).T
                    disagg_results = disagg_results[disagg_results[:,2] != 0]
                    fname = os.path.join(output_dir, 'MagDist_poe_' + str(poes[i]) + '_' + imt + '.out')
                    np.savetxt(fname, disagg_results)

                    ax1 = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

                    X = R[i]
                    Y = M[i]
                    Z = np.zeros(len(X))

                    dx = np.ones(len(X)) * dbin / 2
                    dy = np.ones(len(X)) * Mbin / 2
                    dz = hz_cont[i] * 100

                    # here we may make the colormap based on epsilon instead of hazard contribution
                    max_height = np.max(dz)  # get range of color bars so we can normalize
                    min_height = np.min(dz)
                    # scale each z to [0,1], and get their rgb values
                    rgba = [cmap((k - min_height) / max_height) for k in dz]
                    ax1.bar3d(X, Y, Z, dx, dy, dz, color=rgba, zsort='average', alpha=0.7, shade=True)

                    ax1.set_xlabel('R [km]')
                    ax1.set_ylabel('$M_{w}$')
                    if np.mod(i + 1, n_cols) == 1:
                        ax1.set_zlabel('Hazard Contribution [%]')
                        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
                        ax1.set_zlabel('Hazard Contribution [%]', rotation=90)
                    ax1.zaxis._axinfo['juggled'] = (1, 2, 0)

                    plt.title('$T_{R}$=%s years\n$M_{mod}$=%s, $R_{mod}$=%s km\n$M_{mean}$=%s, $R_{mean}$=%s km'
                              % ("{:.0f}".format(Tr[i]), "{:.2f}".format(modeLst[i][0]), "{:.0f}".format(modeLst[i][1]),
                                 "{:.2f}".format(meanLst[i][0]), "{:.0f}".format(meanLst[i][1])),
                              fontsize=11, loc='right', verticalalignment='top', y=0.95)

                    mean_mags.append(meanLst[i][0])
                    mean_dists.append(meanLst[i][1])
                    mod_mags.append(modeLst[i][0])
                    mod_dists.append(modeLst[i][1])

                plt.subplots_adjust(hspace=0.05, wspace=0.05)  # adjust the subplot to the right for the legend
                fig.suptitle(f"Disaggregation of Seismic Hazard\nIntensity Measure: {imt}\nLatitude: "
                             f"{lat:.4f}, Longitude: {lon:.4f}", fontsize=14, weight='bold', ha='left', x=0.0, y=1.0)

                plt.tight_layout(rect=[0, 0.0, 1, 0.94])
                fname = os.path.join(output_dir, 'Disaggregation_MR_' + imt + '.png')
                plt.savefig(fname, format='png', dpi=300)

                fname = os.path.join(output_dir, 'mean_mags_' + imt + '.out')
                np.savetxt(fname, np.asarray(mean_mags), fmt='%.2f')
                fname = os.path.join(output_dir, 'mean_dists_' + imt + '.out')
                np.savetxt(fname, np.asarray(mean_dists), fmt='%.1f')
                fname = os.path.join(output_dir, 'mod_mags_' + imt + '.out')
                np.savetxt(fname, np.asarray(mod_mags), fmt='%.2f')
                fname = os.path.join(output_dir, 'mod_dists_' + imt + '.out')
                np.savetxt(fname, np.asarray(mod_dists), fmt='%.1f')
                if show:
                    plt.show()
                plt.close(fig)


def disagg_MReps(Mbin, dbin, path_disagg_results, output_dir='Post_Outputs', n_rows=1, filename='Mag_Dist_Eps', show=1):
    """
    Details
    -------
    This script will save disaggregation plots including M, R and eps.

    Parameters
    ----------
    Mbin : int, float
        magnitude bin used in disaggregation.
    dbin : int, float
        distance bin used in disaggregation.
    path_disagg_results: str
        Path to the hazard results
    output_dir: str, optional
        Save outputs to a pickle file
    n_rows : int, optional
        total number of rows for subplots.
    filename : str, optional
        filename to process.
    show: int
        flag to show figure (1 or 0)

    Returns
    -------
    None.
    """

    # lets add the plotting options to make everything clearer
    cmap = cm.get_cmap('jet')  # Get desired colormap

    mags = []
    dists = []

    for file in os.listdir(path_disagg_results):
        if file.startswith(filename):
            # Load the dataframe
            df = pd.read_csv(''.join([path_disagg_results, '/', file]), skiprows=1)
            poes = np.unique(df['poe']).tolist()
            poes.sort(reverse=True)
            # Get some salient values
            f = open(''.join([path_disagg_results, '/', file]), "r")
            ff = f.readline().split(',')
            lon = float(ff[-2].replace(" lon=", ""))
            lat = float(ff[-1].replace(" lat=", "").replace("\"\n", ""))
            ims = np.unique(df['imt'])
            inv_t = float(ff[8].replace(" investigation_time=", ""))
            for imt in ims:
                modeLst, meanLst = [], []
                Tr = []
                hz_cont = []
                M, R, eps = [], [], []
                for poe in poes:
                    Tr.append(round(-inv_t / np.log(1 - poe)))
                    data = {}
                    data['mag'] = df['mag'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['dist'] = df['dist'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['eps'] = df['eps'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['hz_cont'] = df.iloc[:, 5][(df['poe'] == poe) & (df['imt'] == imt)]
                    hz_cont.append(np.array(data['hz_cont'] / data['hz_cont'].sum()))
                    data['hz_cont'] = hz_cont[-1]
                    data = pd.DataFrame(data)
                    data_reduced = data.groupby(['mag', 'dist']).agg(['sum'])[('hz_cont', 'sum')].reset_index().droplevel(1, axis=1)
                    # Compute the modal value (highest poe)
                    mode = data_reduced.sort_values(by='hz_cont', ascending=False)[0:1]
                    modeLst.append([mode['mag'].values[0], mode['dist'].values[0]])
                    # Compute the mean value
                    meanLst.append([np.sum(data_reduced['mag'] * data_reduced['hz_cont']), np.sum(data_reduced['dist'] * data_reduced['hz_cont'])])

                    # Report the individual magnitude and distance bins
                    M.append(np.array(data['mag']))
                    R.append(np.array(data['dist']))
                    eps.append(np.array(data['eps']))

                n_Tr = len(Tr)
                mean_mags = []
                mean_dists = []
                mod_mags = []
                mod_dists = []
                n_eps = len(np.unique(np.asarray(eps)))
                min_eps = np.min(np.unique(np.asarray(eps)))  # get range of colorbars so we can normalize
                max_eps = np.max(np.unique(np.asarray(eps)))

                n_cols = int(np.floor(n_Tr / n_rows))
                if np.mod(n_Tr, n_rows):
                    n_cols += 1

                fig = plt.figure(figsize=(19.2, 10.8))
                for i in range(n_Tr):
                    ax1 = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
                    # Save disaggregation results
                    disagg_results = np.array([M[i], R[i], eps[i], hz_cont[i]]).T
                    disagg_results = disagg_results[disagg_results[:, 3] != 0]
                    fname = os.path.join(output_dir, 'MagDistEps_poe_' + str(poes[i]) + '_' + imt + '.out')
                    np.savetxt(fname, disagg_results)
                    mean_mags.append(meanLst[i][0])
                    mean_dists.append(meanLst[i][1])
                    mod_mags.append(modeLst[i][0])
                    mod_dists.append(modeLst[i][1])

                    # scale each eps to [0,1], and get their rgb values
                    rgba = [cmap((k - min_eps) / max_eps / 2) for k in (np.unique(np.asarray(eps)))]
                    num_triads_M_R_eps = len(R[i])
                    Z = np.zeros(int(num_triads_M_R_eps / n_eps))

                    for l in range(n_eps):
                        X = np.array(R[i][np.arange(l, num_triads_M_R_eps, n_eps)])
                        Y = np.array(M[i][np.arange(l, num_triads_M_R_eps, n_eps)])

                        dx = np.ones(int(num_triads_M_R_eps / n_eps)) * dbin / 2
                        dy = np.ones(int(num_triads_M_R_eps / n_eps)) * Mbin / 2
                        dz = np.array(hz_cont[i][np.arange(l, num_triads_M_R_eps, n_eps)]) * 100

                        ax1.bar3d(X, Y, Z, dx, dy, dz, color=rgba[l], zsort='average', alpha=0.7, shade=True)
                        Z += dz  # add the height of each bar to know where to start the next

                    ax1.set_xlabel('R [km]')
                    ax1.set_ylabel('$M_{w}$')
                    if np.mod(i + 1, n_cols) == 1:
                        ax1.set_zlabel('Hazard Contribution [%]')
                        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
                        ax1.set_zlabel('Hazard Contribution [%]', rotation=90)
                    ax1.zaxis._axinfo['juggled'] = (1, 2, 0)

                    plt.title('$T_{R}$=%s years\n$M_{mod}$=%s, $R_{mod}$=%s km\n$M_{mean}$=%s, $R_{mean}$=%s km'
                              % ("{:.0f}".format(Tr[i]), "{:.2f}".format(modeLst[i][0]), "{:.0f}".format(modeLst[i][1]),
                                 "{:.2f}".format(meanLst[i][0]), "{:.0f}".format(meanLst[i][1])),
                              fontsize=11, loc='right', verticalalignment='top', y=0.95)

                    mags.append(meanLst[i][0])
                    dists.append(meanLst[i][1])

                legend_elements = []
                for j in range(n_eps):
                    legend_elements.append(Patch(facecolor=rgba[n_eps - j - 1],
                                                 label=f"\u03B5 = {np.unique(np.asarray(eps))[n_eps - j - 1]:.2f}"))

                fig.legend(handles=legend_elements, loc="lower center", borderaxespad=0., ncol=n_eps)
                plt.subplots_adjust(hspace=0.05, wspace=0.05)  # adjust the subplot to the right for the legend
                fig.suptitle(f"Disaggregation of Seismic Hazard\nIntensity Measure: {imt}\nLatitude: "
                             f"{lat:.4f}, Longitude: {lon:.4f}", fontsize=14, weight='bold', ha='left', x=0.0, y=1.0)
                plt.tight_layout(rect=[0, 0.03, 1, 0.94])
                fname = os.path.join(output_dir, 'Disaggregation_MReps_' + imt + '.png')
                plt.savefig(fname, format='png', dpi=300)

                fname = os.path.join(output_dir, 'mean_mags_' + imt + '.out')
                np.savetxt(fname, np.asarray(mean_mags), fmt='%.2f')
                fname = os.path.join(output_dir, 'mean_dists_' + imt + '.out')
                np.savetxt(fname, np.asarray(mean_dists), fmt='%.1f')
                fname = os.path.join(output_dir, 'mod_mags_' + imt + '.out')
                np.savetxt(fname, np.asarray(mod_mags), fmt='%.2f')
                fname = os.path.join(output_dir, 'mod_dists_' + imt + '.out')
                np.savetxt(fname, np.asarray(mod_dists), fmt='%.1f')
                if show:
                    plt.show()
                plt.close(fig)


# FUNCTIONS TO PARSE A LOGIC TREE FROM SA to AvgSA
# ---------------------------------------------------------------------
def parse_sa_lt_to_avgsa(input_lt_file, output_lt_file, periods, correlation):
    """
    Details
    -------
    Parses the ordinary SA ground motion logic tree to an AvgSA equivalent

    Parameters
    ----------
    input_lt_file : str
        Input GMPE logic tree for SA, e.g. 'gmmLT.xml'
    output_lt_file : str
        The output GMPE LT file, e.g. 'gmmLT_AvgSA.xml'
    periods : list
        List of periods for the AvgSA calculation
        e.g. periods = [[0.4,0.5,0.6,0.7,0.8], [1.1,1.2,1.3,1.4,1.5]]
    correlation: str
        String for one of the supported correlation models (e.g. 'akkar', 'baker_jayaram')

    Returns
    -------
    None.
    """
    # TODO: modify the method for single AvgSA case, apparently there is no way to run PSHA for multiple AvgSA

    def replace_text_str(input_str):
        """
        Details
        -------
        Replaces the text string of an uncertainty model with an alternative
        formulation in terms of Average Sa

        Parameters
        ----------
        input_str : str
            Input string (return carriage delimited) describing entire uncertainty model

        Returns
        -------
        None.
        """

        search_output = re.search(r'\[(.*?)\]', input_str)
        if search_output:
            input_gmpe = search_output.group(1)
        else:
            input_gmpe = input_str.strip()
        period_str = ",".join(["{:s}".format(str(per)) for per in periods])
        # Setup initial arguments for GenericGmpeAvgSa
        initial_set = ["[GenericGmpeAvgSA]",
                       "gmpe_name = \"{:s}\"".format(input_gmpe),
                       "avg_periods = {:s}".format(period_str),
                       "corr_func = \"{:s}\"".format(correlation)]
        if not search_output:
            # No additional arguments passed to GMPE, just return the string as is
            return "\n".join(initial_set)

        for isegment in input_str.split("\n"):
            segment = isegment.strip()
            if not segment:
                # Empty string
                continue
            if input_gmpe in segment:
                new_gmpe = segment.replace(input_gmpe,
                                           "GenericGmpeAvgSA")
                if not new_gmpe in initial_set:
                    initial_set.append(new_gmpe)
            else:
                initial_set.append(segment)
        return "\n".join(initial_set)

    [input_lt] = nrml.read(input_lt_file)
    output_lt = []
    for blev in input_lt:

        if blev.tag.endswith("logicTreeBranchingLevel"):
            # Removes the branching level
            bset = blev[0]
        else:
            # Has no branching level, only branch set
            bset = deepcopy(blev)
        bset_branches = []
        for br in bset:
            unc_model_str = br.uncertaintyModel.text
            weight = float(br.uncertaintyWeight.text)
            new_unc_model = replace_text_str(unc_model_str)
            br_node = Node("logicTreeBranch", br.attrib, nodes=[
                Node("uncertaintyModel", text=new_unc_model),
                Node("uncertaintyWeight", text=str(weight))
            ])
            bset_branches.append(br_node)
        output_bs = Node("logicTreeBranchSet",
                         bset.attrib,
                         nodes=bset_branches)
        output_lt.append(output_bs)
    output_lt = Node("logicTree",
                     {"logicTreeID": input_lt["logicTreeID"] + "AvgSA"},
                     nodes=output_lt)
    with open(output_lt_file, "wb") as f:
        nrml.write([output_lt], f, fmt="%s")
    print("Written to %s" % output_lt_file)


# FUNCTIONS TO READ GROUND MOTION RECORD FILES
# ---------------------------------------------------------------------

def ContentFromZip(paths, zipName):
    """
    Details
    -------
    This function reads the contents of all selected records
    from the zipfile in which the records are located

    Parameters
    ----------
    paths : list
        Containing file list which are going to be read from the zipfile.
    zipName    : str
        Path to the zip file where file lists defined in "paths" are located.

    Returns
    -------
    contents   : dictionary
        Containing raw contents of the files which are read from the zipfile.
    """

    contents = {}
    with zipfile.ZipFile(zipName, 'r') as myzip:
        for i in range(len(paths)):
            with myzip.open(paths[i]) as myfile:
                contents[i] = [x.decode('utf-8') for x in myfile.readlines()]

    return contents


def ReadNGA(inFilename=None, content=None, outFilename=None):
    """
    Details
    -------
    This function process acceleration history for NGA data file (.AT2 format).

    Parameters
    ----------
    inFilename : str, optional
        Location and name of the input file.
        The default is None
    content    : str, optional
        Raw content of the .AT2 file.
        The default is None
    outFilename : str, optional
        location and name of the output file.
        The default is None.

    Notes
    -----
    At least one of the two variables must be defined: inFilename, content.

    Returns
    -------
    dt   : float
        time interval of recorded points.
    npts : int
        number of points in ground motion record file.
    desc : str
        Description of the earthquake (e.g., name, year, etc).
    t    : numpy.array (n x 1)
        time array, same length with npts.
    acc  : numpy.array (n x 1)
        acceleration array, same length with time unit
        usually in (g) unless stated as other.
    """

    try:
        # Read the file content from inFilename
        if content is None:
            with open(inFilename, 'r') as inFileID:
                content = inFileID.readlines()

        # check the first line
        temp = str(content[0]).split()
        try:  # description is in the end
            float(temp[0])  # do a test with str to float conversion, this will be ok if description is in the end.
            # Description of the record
            desc = content[-2]
            # Number of points and time step of the record
            row4Val = content[-4]
            # Acceleration values
            acc_data = content[:-4]
        except ValueError:  # description is in the beginning
            # Description of the record
            desc = content[1]
            # Number of points and time step of the record
            row4Val = content[3]
            # Acceleration values
            acc_data = content[4:]

        # Description of the record
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '')
        # Number of points and time step of the record
        if row4Val[0][0] == 'N':
            val = row4Val.split()
            if 'dt=' in row4Val:
                dt_str = 'dt='
            elif 'DT=' in row4Val:
                dt_str = 'DT='
            if 'npts=' in row4Val:
                npts_str = 'npts='
            elif 'NPTS=' in row4Val:
                npts_str = 'NPTS='
            if 'sec' in row4Val:
                sec_str = 'sec'
            elif 'SEC' in row4Val:
                sec_str = 'SEC'
            npts = int(val[(val.index(npts_str)) + 1].rstrip(','))
            try:
                dt = float(val[(val.index(dt_str)) + 1])
            except ValueError:
                dt = float(val[(val.index(dt_str)) + 1].replace(sec_str + ',', ''))
        else:
            val = row4Val.split()
            npts = int(val[0])
            dt = float(val[1])

        # Acceleration values
        acc = np.array([])
        for line in acc_data:
            acc = np.append(acc, np.array(line.split(), dtype=float))
        dur = len(acc) * dt
        t = np.arange(0, dur, dt)

        if outFilename is not None:
            np.savetxt(outFilename, acc, fmt='%1.4e')

        return dt, npts, desc, t, acc

    except BaseException as error:
        print(f"Record file reader FAILED for {inFilename}: ", error)


def ReadESM(inFilename=None, content=None, outFilename=None):
    """
    Details
    -------
    This function process acceleration history for ESM data file.

    Parameters
    ----------
    inFilename : str, optional
        Location and name of the input file.
        The default is None
    content    : str, optional
        Raw content of the ESM record file.
        The default is None
    outFilename : str, optional
        location and name of the output file.
        The default is None.

    Returns
    -------
    dt   : float
        time interval of recorded points.
    npts : int
        number of points in ground motion record file.
    desc : str
        Description of the earthquake (e.g., name, year, etc).
    time : numpy.array (n x 1)
        time array, same length with npts.
    acc  : numpy.array (n x 1)
        acceleration array, same length with time unit
        usually in (g) unless stated as other.
    """

    try:
        # Read the file content from inFilename
        if content is None:
            with open(inFilename, 'r') as inFileID:
                content = inFileID.readlines()

        desc = content[:64]
        dt = float(difflib.get_close_matches('SAMPLING_INTERVAL_S', content)[0].split()[1])
        npts = len(content[64:])
        acc_data = content[64:]
        acc = np.asarray([float(data) for data in acc_data], dtype=float)
        dur = len(acc) * dt
        t = np.arange(0, dur, dt)
        acc = acc / 980.655  # cm/s**2 to g

        if outFilename is not None:
            np.savetxt(outFilename, acc, fmt='%1.4e')

        return dt, npts, desc, t, acc

    except BaseException as error:
        print(f"Record file reader FAILED for {inFilename}: ", error)


# FUNCTIONS TO CREATE BUILDING CODE DESIGN SPECTRA
# ---------------------------------------------------------------------

def Sae_ec8_part1(ag, xi, T, ImpClass, Type, SiteClass):
    """
    Details
    -------
    Calculates the design response spectrum according to EN 1998-1:2004

    References
    ----------
    CEN. Eurocode 8: Design of Structures for Earthquake Resistance -  Part 1: General Rules,
    Seismic Actions and Rules for Buildings (EN 1998-1:2004). Brussels, Belgium: 2004.

    Notes
    -----

    Parameters
    ----------
    ag: float
        Peak ground acceleration
    xi: float
        Damping ratio
    T: list or numpy.array
        Period array for which elastic response spectrum is calculated
    ImpClass: str
        Importance class ('I','II','III','IV')
    Type: str
        Type of spectrum ('Type1','Type2')
    SiteClass: str
        Site Soil Class ('A','B','C','D','E')

    Returns
    -------
    Sae: numpy.array
        Elastic acceleration response spectrum

    """

    SpecProp = {
        'Type1': {
            'A': {'S': 1.00, 'Tb': 0.15, 'Tc': 0.4, 'Td': 2.0},
            'B': {'S': 1.20, 'Tb': 0.15, 'Tc': 0.5, 'Td': 2.0},
            'C': {'S': 1.15, 'Tb': 0.20, 'Tc': 0.6, 'Td': 2.0},
            'D': {'S': 1.35, 'Tb': 0.20, 'Tc': 0.8, 'Td': 2.0},
            'E': {'S': 1.40, 'Tb': 0.15, 'Tc': 0.5, 'Td': 2.0},
        },

        'Type2': {
            'A': {'S': 1.00, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
            'B': {'S': 1.35, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
            'C': {'S': 1.50, 'Tb': 0.10, 'Tc': 0.25, 'Td': 1.2},
            'D': {'S': 1.80, 'Tb': 0.10, 'Tc': 0.30, 'Td': 1.2},
            'E': {'S': 1.60, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
        }
    }

    S = SpecProp[Type][SiteClass]['S']
    Tb = SpecProp[Type][SiteClass]['Tb']
    Tc = SpecProp[Type][SiteClass]['Tc']
    Td = SpecProp[Type][SiteClass]['Td']

    eta = max(np.sqrt(0.10 / (0.05 + xi)), 0.55)

    if ImpClass == 'I':
        I = 0.8
    elif ImpClass == 'II':
        I = 1.0
    elif ImpClass == 'III':
        I = 1.2
    elif ImpClass == 'IV':
        I = 1.4
    else:
        print('Error! Cannot compute a value of Importance Factor')

    ag = ag * I

    Sae = []
    for i in range(len(T)):
        if 0 <= T[i] <= Tb:
            Sa_el = ag * S * (1.0 + T[i] / Tb * (2.5 * eta - 1.0))
        elif Tb <= T[i] <= Tc:
            Sa_el = ag * S * 2.5 * eta
        elif Tc <= T[i] <= Td:
            Sa_el = ag * S * 2.5 * eta * (Tc / T[i])
        elif T[i] >= Td:
            Sa_el = ag * S * 2.5 * eta * (Tc * Td / T[i] / T[i])
        else:
            print('Error! Cannot compute a value of Spectral Acceleration')

        Sae.append(Sa_el)

    Sae = np.array(Sae)

    return Sae


def Sae_asce7_16(T, SDS, SD1, TL):
    """
    Details
    -------
    This method determines the design response spectrum based on ASCE 7-16.

    References
    ----------
    American Society of Civil Engineers. (2017, June). Minimum design loads and associated criteria
    for buildings and other structures. American Society of Civil Engineers.

    Notes
    -----

    Parameters
    ----------
    T:  numpy.array
        Period array for which elastic response spectrum is calculated
    SDS: float
        Numeric seismic design value (0.2 sec)
    SD1: float
        Numeric seismic design value (1.0 sec)
    TL: float
        Long-period transition period


    Returns
    -------
    Sae: numpy.array
        Elastic acceleration response spectrum
    """

    T0 = 0.2 * (SD1 / SDS)
    TS = SD1 / SDS
    Sae = np.zeros(len(T))
    for i in range(len(T)):
        if T[i] < T0:
            Sae[i] = SDS * (0.4 + 0.6 * T[i] / T0)
        if T0 <= T[i] <= TS:
            Sae[i] = SDS
        if TS <= T[i] <= TL:
            Sae[i] = SD1 / T[i]
        if TL < T[i]:
            Sae[i] = (SD1 * TL) / (T[i] ** 2)

    return Sae


def SiteParam_asce7_16(Lat, Long, RiskCat, SiteClass):
    """
    Details
    -------
    This method makes use of API developed by USGS to get spectra (ASCE7-16) info in US.
    It retrieves the design response spectrum parameters for the given site.

    References
    ----------
    https://earthquake.usgs.gov/ws/designmaps/asce7-16.html
    American Society of Civil Engineers. (2017, June). Minimum design loads and associated criteria
    for buildings and other structures. American Society of Civil Engineers.

    Notes
    -----

    Parameters
    ----------
    Lat: float
        Site latitude
    Long: float
        Site longitude
    RiskCat:  str
        Risk category for structure ('I','II','III','IV')
    SiteClass: str
        Site soil class ('A','B','C','D','E')

    Returns
    -------
    SDS: float
        Short period (0.2 sec) spectral acceleration coefficient
    SD1: float
        Spectral acceleration coefficient at period 1.0
    TL: float
        Period value for long-period transition
    """

    thisURL = 'https://earthquake.usgs.gov/ws/designmaps/asce7-16.json?latitude=' + str(Lat) + '&longitude=' + str(
        Long) + '&riskCategory=' + RiskCat + '&siteClass=' + SiteClass + '&title=Example'
    web = json.loads(requests.get(thisURL).text)  # get the info from webpage and convert json format to dictionary
    Ss = web['response']['data']['ss']
    S1 = web['response']['data']['s1']
    Fa = web['response']['data']['fa']
    Fv = web['response']['data']['fv']
    TL = web['response']['data']['tl']

    if Ss is None:
        raise ValueError('Failed to get parameter Ss, define user-defined spectrum instead.')
    if S1 is None:
        raise ValueError('Failed to get parameter S1, define user-defined spectrum instead.')
    if Fa is None:
        raise ValueError('Failed to get parameter Fa, define user-defined spectrum instead.')
    if Fv is None:
        raise ValueError('Failed to get parameter Fv, define user-defined spectrum instead.')
    if TL is None:
        raise ValueError('Failed to get parameter TL, define user-defined spectrum instead.')

    Sms = Fa * Ss
    Sm1 = Fv * S1
    SDS = (2 / 3) * Sms
    SD1 = (2 / 3) * Sm1

    return SDS, SD1, TL


def SiteParam_tbec2018(Lat, Long, DD, SiteClass):
    """
    Details
    -------
    This method retrieves the design response spectrum parameters for the given site according to TBEC2018.

    References
    ----------
    TBEC. (2018). Turkish building earthquake code.

    Notes
    -----

    Parameters
    ----------
    Lat: float
        Site latitude
    Long: float
        Site longitude
    DD:  int
        Earthquake ground motion intensity level (1,2,3,4)
    SiteClass: str
        Site soil class ('ZA','ZB','ZC','ZD','ZE')

    Returns
    -------
    PGA: float
        Peak ground acceleration
    SDS: float
        Short period (0.2 sec) spectral acceleration coefficient
    SD1: float
        Spectral acceleration coefficient at period 1.0
    TL: float
        Period value for long-period transition
    """

    csv_file = 'Parameters_TBEC2018.csv'
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Meta_Data', csv_file)
    data = pd.read_csv(file_path)

    # Check if the coordinates are within the limits
    if Long > np.max(data['Longitude']) or Long < np.min(data['Longitude']):
        raise ValueError('Longitude value must be within the limits: [24.55,45.95]')
    if Lat > np.max(data['Latitude']) or Lat < np.min(data['Latitude']):
        raise ValueError('Latitude value must be within the limits: [34.25,42.95]')

    # Targeted probability of exceedance in 50 years
    if DD == 1:
        PoE = '2'
    elif DD == 2:
        PoE = '10'
    elif DD == 3:
        PoE = '50'
    elif DD == 4:
        PoE = '68'

    # Determine Peak Ground Acceleration PGA [g]
    PGA_col = 'PGA (g) - %' + PoE
    data_pga = np.array([data['Longitude'], data['Latitude'], data[PGA_col]]).T
    PGA = interpolate.griddata(data_pga[:, 0:2], data_pga[:, 2], [(Long, Lat)], method='linear')

    # Short period map spectral acceleration coefficient [dimensionless]
    SS_col = 'SS (g) - %' + PoE
    data_ss = np.array([data['Longitude'], data['Latitude'], data[SS_col]]).T
    SS = interpolate.griddata(data_ss[:, 0:2], data_ss[:, 2], [(Long, Lat)], method='linear')

    # Map spectral acceleration coefficient for a 1.0 second period [dimensionless]
    S1_col = 'S1 (g) - %' + PoE
    data_s1 = np.array([data['Longitude'], data['Latitude'], data[S1_col]]).T
    S1 = interpolate.griddata(data_s1[:, 0:2], data_s1[:, 2], [(Long, Lat)], method='linear')

    SoilParam = {
        'FS': {
            'ZA': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            'ZB': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            'ZC': [1.3, 1.3, 1.2, 1.2, 1.2, 1.2],
            'ZD': [1.6, 1.4, 1.2, 1.1, 1.0, 1.0],
            'ZE': [2.4, 1.7, 1.3, 1.1, 0.9, 0.8]
        },

        'SS': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],

        'F1': {
            'ZA': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            'ZB': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            'ZC': [1.5, 1.5, 1.5, 1.5, 1.5, 1.4],
            'ZD': [2.4, 2.2, 2.0, 1.9, 1.8, 1.7],
            'ZE': [4.2, 3.3, 2.8, 2.4, 2.2, 2.0]
        },

        'S1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],

    }

    # Local soil response coefficient for the short period region
    if SS <= SoilParam['SS'][0]:
        FS = SoilParam['FS'][SiteClass][0]
    elif SoilParam['SS'][0] < SS <= SoilParam['SS'][1]:
        FS = (SoilParam['FS'][SiteClass][1] - SoilParam['FS'][SiteClass][0]) \
             * (SS - SoilParam['SS'][0]) / (SoilParam['SS'][1] - SoilParam['SS'][0]) \
             + SoilParam['FS'][SiteClass][0]
    elif SoilParam['SS'][1] < SS <= SoilParam['SS'][2]:
        FS = (SoilParam['FS'][SiteClass][2] - SoilParam['FS'][SiteClass][1]) \
             * (SS - SoilParam['SS'][1]) / (SoilParam['SS'][2] - SoilParam['SS'][1]) \
             + SoilParam['FS'][SiteClass][1]
    elif SoilParam['SS'][2] < SS <= SoilParam['SS'][3]:
        FS = (SoilParam['FS'][SiteClass][3] - SoilParam['FS'][SiteClass][2]) \
             * (SS - SoilParam['SS'][2]) / (SoilParam['SS'][3] - SoilParam['SS'][2]) \
             + SoilParam['FS'][SiteClass][2]
    elif SoilParam['SS'][3] < SS <= SoilParam['SS'][4]:
        FS = (SoilParam['FS'][SiteClass][4] - SoilParam['FS'][SiteClass][3]) \
             * (SS - SoilParam['SS'][3]) / (SoilParam['SS'][4] - SoilParam['SS'][3]) \
             + SoilParam['FS'][SiteClass][3]
    elif SoilParam['SS'][4] < SS <= SoilParam['SS'][5]:
        FS = (SoilParam['FS'][SiteClass][5] - SoilParam['FS'][SiteClass][4]) \
             * (SS - SoilParam['SS'][4]) / (SoilParam['SS'][5] - SoilParam['SS'][4]) \
             + SoilParam['FS'][SiteClass][4]
    elif SS >= SoilParam['SS'][5]:
        FS = SoilParam['FS'][SiteClass][5]

    # Local soil response coefficient for 1.0 second period
    if S1 <= SoilParam['S1'][0]:
        F1 = SoilParam['F1'][SiteClass][0]
    elif SoilParam['S1'][0] < S1 <= SoilParam['S1'][1]:
        F1 = (SoilParam['F1'][SiteClass][1] - SoilParam['F1'][SiteClass][0]) \
             * (S1 - SoilParam['S1'][0]) / (SoilParam['S1'][1] - SoilParam['S1'][0]) \
             + SoilParam['F1'][SiteClass][0]
    elif SoilParam['S1'][1] < S1 <= SoilParam['S1'][2]:
        F1 = (SoilParam['F1'][SiteClass][2] - SoilParam['F1'][SiteClass][1]) \
             * (S1 - SoilParam['S1'][1]) / (SoilParam['S1'][2] - SoilParam['S1'][1]) \
             + SoilParam['F1'][SiteClass][1]
    elif SoilParam['S1'][2] < S1 <= SoilParam['S1'][3]:
        F1 = (SoilParam['F1'][SiteClass][3] - SoilParam['F1'][SiteClass][2]) \
             * (S1 - SoilParam['S1'][2]) / (SoilParam['S1'][3] - SoilParam['S1'][2]) \
             + SoilParam['F1'][SiteClass][2]
    elif SoilParam['S1'][3] < S1 <= SoilParam['S1'][4]:
        F1 = (SoilParam['F1'][SiteClass][4] - SoilParam['F1'][SiteClass][3]) \
             * (S1 - SoilParam['S1'][3]) / (SoilParam['S1'][4] - SoilParam['S1'][3]) \
             + SoilParam['F1'][SiteClass][3]
    elif SoilParam['S1'][4] < S1 <= SoilParam['S1'][5]:
        F1 = (SoilParam['F1'][SiteClass][5] - SoilParam['F1'][SiteClass][4]) \
             * (S1 - SoilParam['S1'][4]) / (SoilParam['S1'][5] - SoilParam['S1'][4]) \
             + SoilParam['F1'][SiteClass][4]
    elif S1 >= SoilParam['S1'][5]:
        F1 = SoilParam['F1'][SiteClass][5]

    SDS = SS * FS
    SD1 = S1 * F1
    TL = 6

    return PGA, SDS, SD1, TL


def Sae_tbec2018(T, PGA, SDS, SD1, TL):
    """
    Details
    -------
    This method calculates the design response spectrum according to TBEC2018.

    References
    ----------
    TBEC. (2018). Turkish building earthquake code.

    Notes
    -----

    Parameters
    ----------
    T:  numpy.array
        Period array for which elastic response spectrum is calculated
    PGA: float
        Peak ground acceleration
    SDS: float
        Short period (0.2 sec) spectral acceleration coefficient
    SD1: float
        Spectral acceleration coefficient at period 1.0
    TL: float
        Period value for long-period transition

    Returns
    -------
    Sae: numpy.array
        Elastic acceleration response spectrum
    """

    TA = 0.2 * SD1 / SDS
    TB = SD1 / SDS
    Sae = np.zeros(len(T))

    for i in range(len(T)):
        if T[i] == 0:
            Sae[i] = PGA
        elif T[i] <= TA:
            Sae[i] = (0.4 + 0.6 * T[i] / TA) * SDS
        elif TA < T[i] <= TB:
            Sae[i] = SDS
        elif TB < T[i] <= TL:
            Sae[i] = SD1 / T[i]
        elif T[i] > TL:
            Sae[i] = SD1 * TL / T[i] ** 2

    return Sae


def Sae_TBEC2007(T, zone, soil):

    """
    Details
    -------
    This method calculates the design response spectrum according to TBEC2007.

    References
    ----------
    TBEC. (2007). Turkish building earthquake code.

    Notes
    -----

    Parameters
    ----------
    T:  numpy.array
        Period array for which elastic response spectrum is calculated
    zone: int
        Seismic zone (1, 2, 3, 4)
    soil: str
        Site class ('Z1', 'Z2', 'Z3', 'Z4')
    SD1: float
        Spectral acceleration coefficient at period 1.0

    Returns
    -------
    Sae: numpy.array
        Elastic acceleration response spectrum
    """

    # Seismic zone
    if zone == 1:
      A0 = 0.1
    elif zone == 2:
      A0 = 0.2
    elif zone == 3:
      A0 = 0.3
    elif zone == 4:
      A0 = 0.4

    # site class
    if soil == 'Z1':
      TA = 0.1
      TB = 0.30
    elif soil == 'Z2':
       TA = 0.15
       TB = 0.40     
    elif soil == 'Z3':
       TA = 0.15
       TB = 0.60   
    elif soil == 'Z4':
       TA = 0.2
       TB = 0.90   

    # Response spectrum
    Sae = np.zeros(len(T))
    for i in range(len(T)):
        if T[i] <= TA:
            Sae[i] = 1+1.5*T[i]/TA
        elif T[i] > TA and T[i]<=TB:       
            Sae[i] = 2.5
        elif T[i] > TB:       
            Sae[i] = 2.5*(TB/T[i])**0.8  
        
    Sae = A0*Sae
    
    return Sae


# FUNCTIONS TO CHECK GMPES IMPLEMENTED IN OPENQUAKE
# ---------------------------------------------------------------------

def get_available_gmpes():
    """
    Details
    -------
    Retrieves available ground motion prediction equations (gmpe) in OpenQuake.

    Parameters
    ----------
    None.

    Returns
    -------
    gmpes : dict
        Dictionary which contains available gmpes in openquake.
    """

    gmpes = {}
    for name, gmpe in gsim.get_available_gsims().items():
        gmpes[name] = gmpe

    return gmpes


def check_gmpe_attributes(gmpe):
    """
    Details
    -------
    Checks the attributes for ground motion prediction equation (gmpe).

    Parameters
    ----------
    gmpe : str
        gmpe name for which attributes going to be checked

    Returns
    -------
    None.
    """

    try:  # this is smth like self.bgmpe = gsim.boore_2014.BooreEtAl2014()
        oq_gmpe = gsim.get_available_gsims()[gmpe]()

        print(f"GMPE name: {gmpe}")
        print(f"Supported tectonic region: {oq_gmpe.DEFINED_FOR_TECTONIC_REGION_TYPE.name}")
        print(
            f"Supported standard deviation: {', '.join([std for std in oq_gmpe.DEFINED_FOR_STANDARD_DEVIATION_TYPES])}")
        print(f"Supported intensity measure: "
              f"{', '.join([imt.__name__ for imt in oq_gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES])}")
        print(f"Supported intensity measure component: {oq_gmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT.name}")
        try:
            coeffs_labels = difflib.get_close_matches('COEFFS', dir(oq_gmpe))
            sa_lowers = []
            sa_uppers = []
            for tmp in coeffs_labels:
                sa_keys = eval('list(oq_gmpe.' + tmp + '.sa_coeffs.keys())')
                if sa_keys:
                    sa_lowers.append(sa_keys[0].period)
                    sa_uppers.append(sa_keys[-1].period)
            if sa_lowers:
                sa_lower = max(sa_lowers)
                sa_upper = min(sa_uppers)
                print(f"Supported SA period range: {' - '.join([str(sa_lower), str(sa_upper)])}")
            else:
                print("The supported SA period range is unknown")
        except AttributeError:
            print("The supported SA period range is unknown")
        print(f"Required distance parameters: {', '.join([dist for dist in oq_gmpe.REQUIRES_DISTANCES])}")
        print(f"Required rupture parameters: {', '.join([rup for rup in oq_gmpe.REQUIRES_RUPTURE_PARAMETERS])}")
        print(f"Required site parameters: {', '.join([site for site in oq_gmpe.REQUIRES_SITES_PARAMETERS])}")

    except KeyError:
        raise KeyError(f'{gmpe} is not a valid gmpe name')
    except BaseException as e:
        raise e


# MISCELLANEOUS FUNCTIONS
# ---------------------------------------------------------------------

def get_esm_token(username, pwd):
    """
    Details
    -------
    This function retrieves ESM database token.

    Notes
    -------
    Data must be obtained using any program supporting the HTTP-POST method, e.g. CURL.
    see: https://esm-db.eu/esmws/generate-signed-message/1/query-options.html
    Credentials must have been retrieved from https://esm-db.eu/#/home.

    Parameters
    ----------
    username     : str
        Account username (e-mail),  e.g. 'username@mail.com'.
    pwd          : str
        Account password, e.g. 'password!12345'.

    Returns
    -------
    None.
    """

    if sys.platform.startswith('win'):
        command = 'curl --ssl-no-revoke -X POST -F ' + '\"' + \
                  'message={' + '\\\"' + 'user_email' + '\\\": ' + '\\\"' + username + '\\\", ' + \
                  '\\\"' + 'user_password' + '\\\": ' + '\\\"' + pwd + '\\\"}' + \
                  '\" ' + '\"https://esm-db.eu/esmws/generate-signed-message/1/query\" > token.txt'
    else:
        command = 'curl -X POST -F \'message={\"user_email\": \"' + \
                  username + '\",\"user_password\": \"' + pwd + \
                  '\"}\' \"https://esm-db.eu/esmws/generate-signed-message/1/query\" > token.txt'

    os.system(command)


def create_dir(dir_path):
    """
    Details
    -------
    Creates a clean directory by deleting it if it exists.

    Parameters
    ----------
    dir_path : str
        name of directory to create.

    None.
    """

    def handle_remove_read_only(func, path, exc):
        excvalue = exc[1]
        if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
            func(path)
        else:
            raise Warning("Path is being used by at the moment.",
                          "It cannot be recreated.")

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=False, onerror=handle_remove_read_only)
    os.makedirs(dir_path)


def run_time(start_time):
    """
    Details
    -------
    Prints the time passed between startTime and FinishTime (now)
    in hours, minutes, seconds. startTime is a global variable.

    Parameters
    ----------
    start_time : int
        The initial time obtained via time().

    Returns
    -------
    None.
    """

    finish_time = time()
    # Procedure to obtained elapsed time in Hr, Min, and Sec
    time_seconds = finish_time - start_time
    time_minutes = int(time_seconds / 60)
    time_hours = int(time_seconds / 3600)
    time_minutes = int(time_minutes - time_hours * 60)
    time_seconds = time_seconds - time_minutes * 60 - time_hours * 3600
    print(f"Run time: {time_hours:.0f} hours: {time_minutes:.0f} minutes: {time_seconds:.2f} seconds")


def random_uniform(num_dimensions, num_samples, sampling_type):
    """
    Details
    -------
    Used to perform sampling based on Monte Carlo Simulation or Latin Hypercube Sampling

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html#scipy.stats.qmc.LatinHypercube

    Parameters
    ----------
    num_dimensions: int
        number of dimensions
    num_samples: int
        number of samples
    sampling_type: str
        type of sampling.
        Monte Carlo Sampling: 'MCS'
        Latin Hypercube Sampling: 'LHS'

    Returns
    -------
    sample: numpy.ndarray (num_samples x num_dimensions)
        Array which contains randomly generated numbers between 0 and 1
    """
    # Not really required, but will ensure different realizations each time
    seed = int(datetime.today().strftime("%H%M%S"))
    if sampling_type == 'MCS':
        # Do Monte Carlo Sampling without any grid
        np.random.seed(seed)
        sample = np.random.uniform(size=[num_dimensions, num_samples]).T
    elif sampling_type == 'LHS':
        # A Latin hypercube sample generates n points in [0, 1)^d.
        # Each univariate marginal distribution is stratified, placing exactly one point in each possible grid.
        sampler = qmc.LatinHypercube(d=num_dimensions, seed=seed)
        sample = sampler.random(n=num_samples)

    return sample


def random_multivariate_normal(mu, cov, num_samples, sampling_option):
    """
    Details
    -------
    Used to generate multivariate correlated normal samples

    References
    ----------
    Yang, T. Y., Moehle, J., Stojadinovic, B., & Der Kiureghian, A. (2009).
    Seismic Performance Evaluation of Facilities: Methodology and Implementation.
    In Journal of Structural Engineering (Vol. 135, Issue 10, pp. 1146–1154).
    American Society of Civil Engineers (ASCE). https://doi.org/10.1061/(asce)0733-9445(2009)135:10(1146)

    Parameters
    ----------
    mu: numpy.ndarray (1-D)
        Mean value vector
    cov: numpy.ndarray (2-D)
        Covariance matrix
    num_samples: int
        number of samples
    sampling_option: str
        Monte Carlo Sampling: 'MCS'
        Latin Hypercube Sampling: 'LHS'

    Returns
    -------
    z : numpy.ndarray (num_samples x num_dimensions)
        Array which contains randomly generated numbers between 0 and 1
    """
    num_dimensions = len(mu)
    if mu.size == mu.shape[0]:
        mu = mu.reshape(-1, 1)
    my = mu @ np.ones([1, num_samples])
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    # The lower-triangular decomposition of the correlation matrix
    ly = eigen_vectors
    # Standard deviations
    dy = np.diag(eigen_values ** 0.5)
    # Generate uniformly distributed between 0 and 1
    u = random_uniform(num_dimensions, num_samples, sampling_option)
    # Compute standard random numbers
    u = norm(loc=0, scale=1).ppf(u)
    # Create realization matrix (Eqn. 4) - @ is the matrix multiplication
    z = (ly @ dy @ u.T + my).T

    return z
