"""
Ground motion record selection toolbox
"""

# Import python libraries
import copy
import os
import pickle
import shutil
import sys
import zipfile
from time import gmtime, sleep
import numpy as np
import numpy.matlib
from scipy import interpolate, integrate
from scipy.io import loadmat
from scipy.stats import skew
import matplotlib
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from .webdriverdownloader import ChromeDriverDownloader, GeckoDriverDownloader
from numba import njit
from openquake.hazardlib import gsim, imt, const
from .utility import create_dir, ContentFromZip, ReadNGA, ReadESM
from .utility import SiteParam_tbec2018, Sae_tbec2018, SiteParam_asce7_16, Sae_asce7_16, Sae_ec8_part1
from .utility import random_multivariate_normal


class _subclass_:
    """
    Details
    -------
    This subclass contains common methods inherited by the two parent classes:
     conditional_spectrum and code_spectrum.
    """

    def __init__(self):
        """
        Details
        -------
        Checks if Meta_Data folder exist inside EzGM. If it does not exist it is going to be retrieved
        using the shared link to the zip file in Google Drive.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        directory_to_extract_to = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Meta_Data')
        if os.path.isdir(directory_to_extract_to):  # if meta data is retrieved before no need to do anything
            pass
        else:  # if meta data is not retrieved before, download it from the shared link
            # File id from google drive: last part of the shared link
            file_id = '15cfA8rVB6uLG7T85HOrar7u0AaCOUdxt'
            path_to_zip_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Meta_Data.zip')
            URL = "https://docs.google.com/uc?export&confirm=download"
            CHUNK_SIZE = 32768

            session = requests.Session()
            response = session.get(URL, params={'id': file_id}, stream=True)
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break

            if token:
                params = {'id': id, 'confirm': token}
                response = session.get(URL, params=params, stream=True)

            with open(path_to_zip_file, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory_to_extract_to)
            # Remove the zip file after extracting the files
            os.remove(path_to_zip_file)

    def _search_database(self):
        """
        Details
        -------
        Searches the database and does the filtering.

        Notes:
        ------
        If any value in database file is -1, it means that the value is unknown.

        Parameters
        ----------
        None.

        Returns
        -------
        sampleBig : numpy.array
            An array which contains the IMLs from filtered database.
        soil_Vs30 : numpy.array
            An array which contains the Vs30s from filtered database.
        Mw : numpy.array
            An array which contains the magnitudes from filtered database.
        Rjb : numpy.array
            An array which contains the Rjbs from filtered database.
        fault : numpy.array
            An array which contains the fault type info from filtered database.
        Filename_1 : numpy.array
            An array which contains the filename of 1st gm component from filtered database.
            If selection is set to 1, it will include filenames of both components.
        Filename_2 : numpy.array
            An array which contains the filename of 2nd gm component filtered database.
            If selection is set to 1, it will be None value.
        NGA_num : numpy.array
            If NGA_W2 is used as record database, record sequence numbers from filtered
            database will be saved, for other databases this variable is None.
        eq_ID : numpy.array
            An array which contains event ids from filtered database.
        station_code : numpy.array
            If ESM_2018 is used as record database, station codes from filtered
            database will be saved, for other databases this variable is None.
        """

        if self.selection == 1:  # SaKnown = Sa_arb

            SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            soil_Vs30 = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
            Mw = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)
            Rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
            fault = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
            Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            eq_ID = np.append(self.database['EQID'], self.database['EQID'], axis=0)

            if self.database['Name'] == "NGA_W2":
                NGA_num = np.append(self.database['NGA_num'], self.database['NGA_num'], axis=0)

            elif self.database['Name'] == "ESM_2018":
                station_code = np.append(self.database['station_code'], self.database['station_code'], axis=0)

        elif self.selection == 2:

            if self.Sa_def == 'GeoMean':
                SaKnown = np.sqrt(self.database['Sa_1'] * self.database['Sa_2'])
            elif self.Sa_def == 'SRSS':
                SaKnown = np.sqrt(self.database['Sa_1'] ** 2 + self.database['Sa_2'] ** 2)
            elif self.Sa_def == 'ArithmeticMean':
                SaKnown = (self.database['Sa_1'] + self.database['Sa_2']) / 2
            elif self.Sa_def == 'RotD50':  # SaKnown = Sa_RotD50.
                SaKnown = self.database['Sa_RotD50']
            elif self.Sa_def == 'RotD100':  # SaKnown = Sa_RotD100.
                SaKnown = self.database['Sa_RotD100']
            else:
                raise ValueError('Unexpected Sa definition, exiting...')

            soil_Vs30 = self.database['soil_Vs30']
            Mw = self.database['magnitude']
            Rjb = self.database['Rjb']
            fault = self.database['mechanism']
            Filename_1 = self.database['Filename_1']
            Filename_2 = self.database['Filename_2']
            eq_ID = self.database['EQID']

            if self.database['Name'] == "NGA_W2":
                NGA_num = self.database['NGA_num']

            elif self.database['Name'] == "ESM_2018":
                station_code = self.database['station_code']

        else:
            raise ValueError('Selection can only be performed for one or two components at the moment, exiting...')

        # Limiting the records to be considered using the `notAllowed' variable
        # Sa cannot be negative or zero, remove these.
        notAllowed = np.unique(np.where(SaKnown <= 0)[0]).tolist()

        if self.Vs30_lim is not None:  # limiting values on soil exist
            mask = (soil_Vs30 > min(self.Vs30_lim)) * (soil_Vs30 < max(self.Vs30_lim))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Mw_lim is not None:  # limiting values on magnitude exist
            mask = (Mw > min(self.Mw_lim)) * (Mw < max(self.Mw_lim))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Rjb_lim is not None:  # limiting values on Rjb exist
            mask = (Rjb > min(self.Rjb_lim)) * (Rjb < max(self.Rjb_lim))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.fault_lim is not None:  # limiting values on mechanism exist
            for fault_i in range(len(self.fault_lim)):
                if fault_i == 0:
                    mask = fault == self.fault_lim[fault_i]
                else:
                    mask = np.logical_or(mask, fault == self.fault_lim[fault_i])
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        # get the unique values
        notAllowed = (list(set(notAllowed)))
        Allowed = [i for i in range(SaKnown.shape[0])]
        for i in notAllowed:
            Allowed.remove(i)

        # Use only allowed records
        SaKnown = SaKnown[Allowed, :]
        soil_Vs30 = soil_Vs30[Allowed]
        Mw = Mw[Allowed]
        Rjb = Rjb[Allowed]
        fault = fault[Allowed]
        eq_ID = eq_ID[Allowed]
        Filename_1 = Filename_1[Allowed]

        if self.selection == 1:
            Filename_2 = None
        else:
            Filename_2 = Filename_2[Allowed]

        if self.database['Name'] == "NGA_W2":
            NGA_num = NGA_num[Allowed]
            station_code = None
        elif self.database['Name'] == "ESM_2018":
            NGA_num = None
            station_code = station_code[Allowed]

        # Arrange the available spectra in a usable format and check for invalid input
        # Match periods (known periods and periods for error computations)
        recPer = []
        for i in range(len(self.T)):
            recPer.append(np.where(self.database['Periods'] == self.T[i])[0][0])

        # Check for invalid input
        sampleBig = SaKnown[:, recPer]
        if np.any(np.isnan(sampleBig)):
            raise ValueError('NaNs found in input response spectra')

        if self.nGM > len(eq_ID):
            raise ValueError('There are not enough records which satisfy',
                             'the given record selection criteria...',
                             'Please use broaden your selection criteria...')

        return sampleBig, soil_Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num, eq_ID, station_code

    def write(self, obj=0, recs=1, rtype='acc', recs_f=''):
        """
        Details
        -------
        Writes the object as pickle, selected and scaled records as .txt files.

        Parameters
        ----------
        obj : int, optional
            flag to write the object into the pickle file.
            The default is 0.
        recs : int, optional
            flag to write the selected and scaled time histories.
            The default is 1.
        rtype : str, optional
            option to choose the type of time history to be written.
            'acc' : for the acceleration series
            'vel' : for the velocity series
            'disp': for the displacement series
        recs_f : str, optional
            This is option could be used if the user already has all the
            records in database. This is the folder path which contains
            "database.zip" file. The records must be placed inside
            recs_f/database.zip/database/
            The default is ''.

        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """
        def save_signal(path, acc, sf, dt):
            """
            Details
            -------
            Saves the final signal to the specified path.

            Parameters
            ----------
            path : str
                path of the file to save
            uns_acc : numpy.ndarray
                unscaled acceleration series
            sf : float
                scaling factor
            dt : float
                time step 

            Returns
            -------
            None.
            """

            if rtype == 'vel':  # integrate once if velocity
                signal = integrate.cumtrapz(uns_acc * sf, dx=dt, initial=0)

            elif rtype == 'disp':  # integrate twice if displacement
                signal = integrate.cumtrapz(integrate.cumtrapz(uns_acc * sf, dx=dt, initial=0), dx=dt, initial=0)

            else:
                signal = uns_acc * sf

            np.savetxt(path, signal, fmt='%1.5e')

        if recs == 1:
            # set the directories and file names
            try:  # this will work if records are downloaded
                zipName = self.Unscaled_rec_file
            except AttributeError:
                zipName = os.path.join(recs_f, self.database['Name'] + '.zip')
            n = len(self.rec_h1)
            path_dts = os.path.join(self.outdir, 'GMR_dts.txt')
            dts = np.zeros(n)
            path_H1 = os.path.join(self.outdir, 'GMR_names.txt')
            if self.selection == 2:
                path_H1 = os.path.join(self.outdir, 'GMR_H1_names.txt')
                path_H2 = os.path.join(self.outdir, 'GMR_H2_names.txt')
                h2s = open(path_H2, 'w')
            h1s = open(path_H1, 'w')

            # Get record paths for # NGA_W2 or ESM_2018
            if zipName != os.path.join(recs_f, self.database['Name'] + '.zip'):
                rec_paths1 = self.rec_h1
                if self.selection == 2:
                    rec_paths2 = self.rec_h2
            else:
                rec_paths1 = [self.database['Name'] + '/' + self.rec_h1[i] for i in range(n)]
                if self.selection == 2:
                    rec_paths2 = [self.database['Name'] + '/' + self.rec_h2[i] for i in range(n)]

            # Read contents from zipfile
            contents1 = ContentFromZip(rec_paths1, zipName)  # H1 gm components
            if self.selection == 2:
                contents2 = ContentFromZip(rec_paths2, zipName)  # H2 gm components

            # Start saving records
            for i in range(n):

                # Read the record files
                if self.database['Name'].startswith('NGA'):  # NGA
                    dts[i], npts1, _, _, inp_acc1 = ReadNGA(inFilename=self.rec_h1[i], content=contents1[i])
                    gmr_file1 = self.rec_h1[i].replace('/', '_')[:-4] + '_' + rtype.upper() + '.txt'

                    if self.selection == 2:  # H2 component
                        _, npts2, _, _, inp_acc2 = ReadNGA(inFilename=self.rec_h2[i], content=contents2[i])
                        gmr_file2 = self.rec_h2[i].replace('/', '_')[:-4] + '_' + rtype.upper() + '.txt'

                elif self.database['Name'].startswith('ESM'):  # ESM
                    dts[i], npts1, _, _, inp_acc1 = ReadESM(inFilename=self.rec_h1[i], content=contents1[i])
                    gmr_file1 = self.rec_h1[i].replace('/', '_')[:-11] + '_' + rtype.upper() + '.txt'
                    if self.selection == 2:  # H2 component
                        _, npts2, _, _, inp_acc2 = ReadESM(inFilename=self.rec_h2[i], content=contents2[i])
                        gmr_file2 = self.rec_h2[i].replace('/', '_')[:-11] + '_' + rtype.upper() + '.txt'

                # Write the record files
                if self.selection == 2:
                    # ensure that two acceleration signals have the same length, if not add zeros.
                    npts = max(npts1, npts2)
                    temp1 = np.zeros(npts)
                    temp1[:npts1] = inp_acc1
                    inp_acc1 = temp1.copy()
                    temp2 = np.zeros(npts)
                    temp2[:npts2] = inp_acc2
                    inp_acc2 = temp2.copy()

                    # H2 component
                    save_signal(path=os.path.join(self.outdir, gmr_file2), inp_acc2, self.rec_scale[i], dts[i])
                    h2s.write(gmr_file2 + '\n')

                # H1 component
                save_signal(path=os.path.join(self.outdir, gmr_file1), inp_acc1, self.rec_scale[i], dts[i])
                h1s.write(gmr_file1 + '\n')

            # Time steps
            np.savetxt(path_dts, dts, fmt='%.5f')
            h1s.close()
            if self.selection == 2:
                h2s.close()

        if obj == 1:
            # save some info as pickle obj
            path_obj = os.path.join(self.outdir, 'obj.pkl')
            obj = vars(copy.deepcopy(self))  # use copy.deepcopy to create independent obj
            obj['database'] = self.database['Name']
            del obj['outdir']

            if 'bgmpe' in obj:
                del obj['bgmpe']

            with open(path_obj, 'wb') as file:
                pickle.dump(obj, file)

        print(f"Finished writing process, the files are located in\n{self.outdir}")

    def plot(self, tgt=0, sim=0, rec=1, save=0, show=1):
        """
        Details
        -------
        Plots the spectra of selected and simulated records,
        and/or target spectrum.

        Parameters
        ----------
        tgt    : int, optional for conditional_spectrum
            Flag to plot target spectrum.
            The default is 1.
        sim    : int, optional for conditional_spectrum
            Flag to plot simulated response spectra vs. target spectrum.
            The default is 0.
        rec    : int, optional for conditional_spectrum
            Flag to plot Selected response spectra of selected records
            vs. target spectrum.
            The default is 1.
        save   : int, optional for all selection options
            Flag to save plotted figures in pdf format.
            The default is 0.
        show  : int, optional for all selection options
            Flag to show figures
            The default is 0.

        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """

        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIG_SIZE = 14
        BIGGER_SIZE = 18

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.ioff()

        if type(self).__name__ == 'conditional_spectrum':

            if self.cond == 1:
                if len(self.Tstar) == 1:
                    hatch = [float(self.Tstar * 0.98), float(self.Tstar * 1.02)]
                else:
                    hatch = [float(self.Tstar.min()), float(self.Tstar.max())]

            if tgt == 1:
                # Plot Target spectrum vs. Simulated response spectra
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                plt.suptitle('Target Spectrum', y=0.95)
                ax[0].loglog(self.T, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
                if self.useVar == 1:
                    ax[0].loglog(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.T, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
                ax[0].set_xlabel('Period [sec]')
                ax[0].set_ylabel('Spectral Acceleration [g]')
                ax[0].grid(True)

                handles, labels = ax[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
                ax[0].set_xlim([self.T[0], self.T[-1]])
                if self.cond == 1:
                    ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                # Sample and target standard deviations
                if self.useVar == 1:
                    ax[1].semilogx(self.T, self.sigma_ln, color='red', linestyle='--', lw=2,
                                   label='Target - $\sigma_{ln}$')
                    ax[1].set_xlabel('Period [sec]')
                    ax[1].set_ylabel('Dispersion')
                    ax[1].grid(True)
                    ax[1].legend(frameon=False)
                    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                    ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax[0].set_xlim([self.T[0], self.T[-1]])
                    ax[1].set_ylim(bottom=0)
                    if self.cond == 1:
                        ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if save == 1:
                    plt.savefig(os.path.join(self.outdir, 'Targeted.pdf'))

            if sim == 1:
                # Plot Target spectrum vs. Simulated response spectra
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                plt.suptitle('Target Spectrum vs. Simulated Spectra', y=0.95)

                for i in range(self.nGM):
                    ax[0].loglog(self.T, np.exp(self.sim_spec[i, :]), color='gray', lw=1, label='Selected')

                ax[0].loglog(self.T, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
                if self.useVar == 1:
                    ax[0].loglog(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.T, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].loglog(self.T, np.exp(np.mean(self.sim_spec, axis=0)), color='blue', lw=2,
                             label='Selected - $e^{\mu_{ln}}$')
                if self.useVar == 1:
                    ax[0].loglog(self.T, np.exp(np.mean(self.sim_spec, axis=0) + 2 * np.std(self.sim_spec, axis=0)),
                                 color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.T, np.exp(np.mean(self.sim_spec, axis=0) - 2 * np.std(self.sim_spec, axis=0)),
                                 color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
                ax[0].set_xlabel('Period [sec]')
                ax[0].set_ylabel('Spectral Acceleration [g]')
                ax[0].grid(True)
                handles, labels = ax[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
                ax[0].set_xlim([self.T[0], self.T[-1]])
                if self.cond == 1:
                    ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if self.useVar == 1:
                    # Sample and target standard deviations
                    ax[1].semilogx(self.T, self.sigma_ln, color='red', linestyle='--', lw=2,
                                   label='Target - $\sigma_{ln}$')
                    ax[1].semilogx(self.T, np.std(self.sim_spec, axis=0), color='black', linestyle='--', lw=2,
                                   label='Selected - $\sigma_{ln}$')
                    ax[1].set_xlabel('Period [sec]')
                    ax[1].set_ylabel('Dispersion')
                    ax[1].grid(True)
                    ax[1].legend(frameon=False)
                    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                    ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax[1].set_xlim([self.T[0], self.T[-1]])
                    ax[1].set_ylim(bottom=0)
                    if self.cond == 1:
                        ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if save == 1:
                    plt.savefig(os.path.join(self.outdir, 'Simulated.pdf'))

            if rec == 1:
                # Plot Target spectrum vs. Selected response spectra
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                plt.suptitle('Target Spectrum vs. Spectra of Selected Records', y=0.95)

                for i in range(self.nGM):
                    ax[0].loglog(self.T, np.exp(self.rec_spec[i, :]), color='gray', lw=1, label='Selected')

                ax[0].loglog(self.T, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
                if self.useVar == 1:
                    ax[0].loglog(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.T, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].loglog(self.T, np.exp(np.mean(self.rec_spec, axis=0)), color='blue', lw=2,
                             label='Selected - $e^{\mu_{ln}}$')
                ax[0].loglog(self.T, np.exp(np.mean(self.rec_spec, axis=0) + 2 * np.std(self.rec_spec, axis=0)),
                             color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T, np.exp(np.mean(self.rec_spec, axis=0) - 2 * np.std(self.rec_spec, axis=0)),
                             color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
                ax[0].set_xlabel('Period [sec]')
                ax[0].set_ylabel('Spectral Acceleration [g]')
                ax[0].grid(True)
                handles, labels = ax[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
                ax[0].set_xlim([self.T[0], self.T[-1]])
                if self.cond == 1:
                    ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                # Sample and target standard deviations
                ax[1].semilogx(self.T, self.sigma_ln, color='red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
                ax[1].semilogx(self.T, np.std(self.rec_spec, axis=0), color='black', linestyle='--', lw=2,
                               label='Selected - $\sigma_{ln}$')
                ax[1].set_xlabel('Period [sec]')
                ax[1].set_ylabel('Dispersion')
                ax[1].grid(True)
                ax[1].legend(frameon=False)
                ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[0].set_xlim([self.T[0], self.T[-1]])
                ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_ylim(bottom=0)
                if self.cond == 1:
                    ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if save == 1:
                    plt.savefig(os.path.join(self.outdir, 'Selected.pdf'))

        if type(self).__name__ == 'code_spectrum':

            hatch = [self.Tlower, self.Tupper]
            # Plot Target spectrum vs. Selected response spectra
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            for i in range(self.rec_spec.shape[0]):
                ax.plot(self.T, self.rec_spec[i, :] * self.rec_scale[i], color='gray', lw=1, label='Selected')
            ax.plot(self.T, np.mean(self.rec_spec * self.rec_scale.reshape(-1, 1), axis=0), color='black', lw=2,
                    label='Selected Mean')

            if self.code == 'TBEC 2018':
                ax.plot(self.T, self.target, color='red', lw=2, label='Design Response Spectrum')
                if self.selection == 2:
                    ax.plot(self.T, 1.3 * self.target, color='red', ls='--', lw=2,
                            label='1.3 x Design Response Spectrum')

            if self.code == 'ASCE 7-16':
                ax.plot(self.T, self.target, color='red', lw=2, label='$MCE_{R}$ Response Spectrum')
                ax.plot(self.T, 0.9 * self.target, color='red', ls='--', lw=2,
                        label='0.9 x $MCE_{R}$ Response Spectrum')

            if self.code == 'EC8-Part1':
                ax.plot(self.T, self.target, color='red', lw=2, label='Design Response Spectrum')
                ax.plot(self.T, 0.9 * self.target, color='red', lw=2, ls='--', label='0.9 x Design Response Spectrum')

            ax.axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
            ax.set_xlabel('Period [sec]')
            ax.set_ylabel('Spectral Acceleration [g]')
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), frameon=False)
            ax.set_xlim([self.T[0], self.Tupper * 2])
            plt.suptitle(f'Spectra of Selected Records ({self.code})', y=0.95)

            if save == 1:
                plt.savefig(os.path.join(self.outdir, 'Selected.pdf'))

        # Show the figure
        if show == 1:
            plt.show()

        plt.close('all')

    def esm2018_download(self):
        """

        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from ESM database
        (https://esm-db.eu/) based on their event ID and station_codes.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        if self.database['Name'] == 'ESM_2018':
            print('\nStarted executing esm2018_download method...')

            # temporary zipfile name
            zip_temp = os.path.join(self.outdir, 'output_temp.zip')
            # temporary folder to extract files
            folder_temp = os.path.join(self.outdir, 'output_temp')

            for i in range(self.nGM):
                print('Downloading %d/%d...' % (i + 1, self.nGM))
                event = self.rec_eqID[i]
                station = self.rec_station_code[i]
                params = (
                    ('eventid', event),
                    ('data-type', 'ACC'),
                    ('station', station),
                    ('format', 'ascii'),
                )
                files = {
                    'message': ('path/to/token.txt', open('token.txt', 'rb')),
                }

                url = 'https://esm-db.eu/esmws/eventdata/1/query'

                req = requests.post(url=url, params=params, files=files)

                if req.status_code == 200:
                    with open(zip_temp, "wb") as zf:
                        zf.write(req.content)

                    with zipfile.ZipFile(zip_temp, 'r') as zipObj:
                        zipObj.extractall(folder_temp)
                    os.remove(zip_temp)

                else:
                    if req.status_code == 403:
                        sys.exit('Problem with ESM download. Maybe the token is no longer valid')
                    else:
                        sys.exit('Problem with ESM download. Status code: ' + str(req.status_code))

            # create the output zipfile for all downloaded records
            time_tag = gmtime()
            time_tag_str = f'{time_tag[0]}'
            for i in range(1, len(time_tag)):
                time_tag_str += f'_{time_tag[i]}'
            file_name = os.path.join(self.outdir, f'unscaled_records_{time_tag_str}.zip')
            with zipfile.ZipFile(file_name, 'w') as zipObj:
                len_dir_path = len(folder_temp)
                for root, _, files in os.walk(folder_temp):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipObj.write(file_path, file_path[len_dir_path:])

            shutil.rmtree(folder_temp)
            self.Unscaled_rec_file = file_name
            print(f'Downloaded files are located in\n{self.Unscaled_rec_file}')

        else:
            raise ValueError('You have to use ESM_2018 database to use esm2018_download method.')

    def ngaw2_download(self, username, pwd, sleeptime=2, browser='chrome'):
        """
        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from NGA-West2 Database
        (https://ngawest2.berkeley.edu/) by Record Sequence Numbers (RSNs).

        Notes
        -----
        Either of google-chrome or mozilla-firefox should have been installed priorly.

        Parameters
        ----------
        username     : str
            Account username (e-mail),  e.g. 'username@mail.com'.
        pwd          : str
            Account password, e.g. 'password!12345'.
        sleeptime    : int, default is 3
            Time (sec) spent between each browser operation. This can be increased or decreased depending on the internet speed.
        browser       : str, default is 'chrome'
            The browser to use for download purposes. Valid entries are: 'chrome' or 'firefox'.

        Returns
        -------
        None
        """

        def dir_size(download_dir):
            """
            Details
            -------
            Measures download directory size

            Parameters
            ----------
            download_dir     : str
                Directory for the output time histories to be downloaded

            Returns
            -------
            total_size      : float
                Measured size of the download directory

            """

            total_size = 0
            for path, dirs, files in os.walk(download_dir):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            return total_size

        def download_wait(download_dir):
            """
            Details
            -------
            Waits for download to finish, and an additional amount of time based on the predefined sleeptime variable.

            Parameters
            ----------
            download_dir     : str
                Directory for the output time histories to be downloaded.

            Returns
            -------
            None
            """
            delta_size = 100
            flag = 0
            flag_lim = 5
            while delta_size > 0 and flag < flag_lim:
                size_0 = dir_size(download_dir)
                sleep(1.5 * sleeptime)
                size_1 = dir_size(download_dir)
                if size_1 - size_0 > 0:
                    delta_size = size_1 - size_0
                else:
                    flag += 1
                    print('Finishing in', flag_lim - flag, '...')

        def set_driver(download_dir, browser):
            """
            Details
            -------
            This function starts the webdriver in headless mode.

            Parameters
            ----------
            download_dir     : str
                Directory for the output time histories to be downloaded.
            browser       : str, default is 'chrome'
                The browser to use for download purposes. Valid entries are: 'chrome' or 'firefox'

            Returns
            -------
            driver      : selenium webdriver object
                Driver object used to download NGA_W2 records.
            """

            print('Getting the webdriver to use...')

            # Check if ipython is installed
            try:
                __IPYTHON__
                _in_ipython_session = True
            except NameError:
                _in_ipython_session = False

            try:
                # Running on Google Colab
                if _in_ipython_session and 'google.colab' in str(get_ipython()):
                    os.system('apt-get update')
                    os.system('sudo apt install chromium-chromedriver')
                    os.system('sudo cp /usr/lib/chromium-browser/chromedriver /usr/bin')
                    options = webdriver.ChromeOptions()
                    options.add_argument('-headless')
                    options.add_argument('-no-sandbox')
                    options.add_argument('-disable-dev-shm-usage')
                    prefs = {"download.default_directory": download_dir}
                    options.add_experimental_option("prefs", prefs)
                    driver = webdriver.Chrome('chromedriver', options=options)

                # Running on Binder or Running on personal computer (PC) using firefox
                elif (_in_ipython_session and 'jovyan' in os.getcwd()) or browser == 'firefox':
                    gdd = GeckoDriverDownloader()
                    driver_path = gdd.download_and_install()
                    options = webdriver.firefox.options.Options()
                    options.headless = True
                    options.set_preference("browser.download.folderList", 2)
                    options.set_preference("browser.download.dir", download_dir)
                    options.set_preference('browser.download.useDownloadDir', True)
                    options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/zip')
                    driver = webdriver.Firefox(executable_path=driver_path[1], options=options)

                # Running on personal computer (PC) using chrome
                elif browser == 'chrome':
                    gdd = ChromeDriverDownloader()
                    driver_path = gdd.download_and_install()
                    ChromeOptions = webdriver.ChromeOptions()
                    prefs = {"download.default_directory": download_dir}
                    ChromeOptions.add_experimental_option("prefs", prefs)
                    ChromeOptions.headless = True
                    driver = webdriver.Chrome(executable_path=driver_path[1], options=ChromeOptions)

                print('Webdriver is obtained successfully.')

                return driver

            except RuntimeError:
                print('Failed to get webdriver.')
                raise

        def sign_in(driver, USERNAME, PASSWORD):
            """

            Details
            -------
            This function signs in to 'https://ngawest2.berkeley.edu/' with
            given account credentials.

            Parameters
            ----------
            driver     : selenium webdriver object
                Driver object used to download NGA_W2 records.
            USERNAME   : str
                Account username (e-mail), e.g.: 'username@mail.com'.
            PASSWORD   : str
                Account password, e.g.: 'password!12345'.

            Returns
            -------
            driver      : selenium webdriver object
                Driver object used to download NGA_W2 records.

            """
            # TODO: Selenium 4.3.0
            #  Deprecated find_element_by_* and find_elements_by_* are now removed (#10712)
            #  See: https://stackoverflow.com/questions/72773206/selenium-python-attributeerror-webdriver-object-has-no-attribute-find-el
            print("Signing in with credentials...")
            driver.get('https://ngawest2.berkeley.edu/users/sign_in')
            driver.find_element_by_id('user_email').send_keys(USERNAME)
            driver.find_element_by_id('user_password').send_keys(PASSWORD)
            driver.find_element_by_id('user_submit').click()

            try:
                alert = driver.find_element_by_css_selector('p.alert')
                warn = alert.text
            except BaseException as e:
                warn = None
                print(e)

            if str(warn) == 'Invalid email or password.':
                driver.quit()
                raise Warning('Invalid email or password.')
            else:
                print('Signed in successfully.')

            return driver

        def download(RSNs, download_dir, driver):
            """

            Details
            -------
            This function dowloads the timehistories which have been indicated with their RSNs
            from 'https://ngawest2.berkeley.edu/'.

            Parameters
            ----------
            RSNs     : str
                A string variable contains RSNs to be downloaded which uses ',' as delimiter
                between RNSs, e.g.: '1,5,91,35,468'.
            download_dir     : str
                Directory for the output time histories to be downloaded.
            driver     : selenium webdriver object
                Driver object used to download NGA_W2 records.

            Returns
            -------
            None

            """
            print("Listing the Records...")
            driver.get('https://ngawest2.berkeley.edu/spectras/new?sourceDb_flag=1')
            sleep(sleeptime)
            driver.find_element_by_xpath("//button[@type='button']").submit()
            sleep(sleeptime)
            driver.find_element_by_id('search_search_nga_number').send_keys(RSNs)
            sleep(sleeptime)
            driver.find_element_by_xpath(
                "//button[@type='button' and @onclick='uncheck_plot_selected();reset_selectedResult();OnSubmit();']").submit()
            sleep(1.5 * sleeptime)
            try:
                note = driver.find_element_by_id('notice').text
                print(note)
            except BaseException as e:
                note = 'NO'
                error = e

            if 'NO' in note:
                driver.set_window_size(800, 800)
                driver.save_screenshot(os.path.join(self.outdir, 'download_error.png'))
                driver.quit()
                raise Warning("Could not be able to download records!"
                              "Either they no longer exist in database"
                              "or you have exceeded the download limit")
            else:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                sleep(sleeptime)
                driver.find_element_by_xpath("//button[@type='button' and @onclick='getSelectedResult(true)']").click()
                obj = driver.switch_to.alert
                msg = obj.text
                print(msg)
                sleep(sleeptime)
                obj.accept()
                sleep(sleeptime)
                obj = driver.switch_to.alert
                msg = obj.text
                print(msg)
                sleep(sleeptime)
                obj.accept()
                sleep(sleeptime)
                download_wait(download_dir)
                driver.quit()

        if self.database['Name'] == 'NGA_W2':
            print('\nStarted executing ngaw2_download method...')

            self.username = username
            self.pwd = pwd
            driver = set_driver(self.outdir, browser)
            driver = sign_in(driver, self.username, self.pwd)
            RSNs = ''
            for i in self.rec_rsn:
                RSNs += str(int(i)) + ','
            RSNs = RSNs[:-1:]
            files_before_download = set(os.listdir(self.outdir))
            download(RSNs, self.outdir, driver)
            files_after_download = set(os.listdir(self.outdir))
            Downloaded_File = str(list(files_after_download.difference(files_before_download))[0])
            file_extension = Downloaded_File[Downloaded_File.find('.')::]
            time_tag = gmtime()
            time_tag_str = f'{time_tag[0]}'
            for i in range(1, len(time_tag)):
                time_tag_str += f'_{time_tag[i]}'
            new_file_name = f'unscaled_records_{time_tag_str}{file_extension}'
            Downloaded_File = os.path.join(self.outdir, Downloaded_File)
            Downloaded_File_Rename = os.path.join(self.outdir, new_file_name)
            os.rename(Downloaded_File, Downloaded_File_Rename)
            self.Unscaled_rec_file = Downloaded_File_Rename
            print(f'Downloaded files are located in\n{self.Unscaled_rec_file}')
        else:
            raise ValueError('You have to use NGA_W2 database to use ngaw2_download method.')


class conditional_spectrum(_subclass_):
    """
    This class is used to
        1) Create target spectrum
            Unconditional spectrum using specified gmpe
            Conditional spectrum using average spectral acceleration
            Conditional spectrum using spectral acceleration
            with and without considering variance
        2) Selecting suitable ground motion sets for target spectrum
        3) Scaling and processing of selected ground motion records
    """

    def __init__(self, database='NGA_W2', outdir='Outputs'):
        # TODO: Combine all metadata into single sql file.
        """
        Details
        -------
        Loads the database and add spectral values for Tstar 
        if they are not present via interpolation.
        
        Parameters
        ----------
        database : str, optional
            Database to use: NGA_W2, ESM_2018
            The default is NGA_W2.
        outdir     : str, optional, the default is 'Outputs'.
            output directory to create.
            
        Returns
        -------
        None.
        """

        # Add the input the ground motion database to use
        super().__init__()
        matfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Meta_Data', database)
        self.database = loadmat(matfile, squeeze_me=True)
        self.database['Name'] = database

        # create the output directory and add the path to self
        cwd = os.getcwd()
        outdir_path = os.path.join(cwd, outdir)
        self.outdir = outdir_path
        create_dir(self.outdir)

    @staticmethod
    def _BakerJayaramCorrelationModel(T1, T2):
        """
        Details
        -------
        Valid for T = 0.01-10sec
    
        References
        ----------
        Baker JW, Jayaram N. Correlation of Spectral Acceleration Values from NGA Ground Motion Models.
        Earthquake Spectra 2008; 24(1): 299–317. DOI: 10.1193/1.2857544.
    
        Parameters
        ----------
        T1: float
            First period
        T2: float
            Second period
    
        Returns
        -------
        rho: float
             Predicted correlation coefficient
        """

        t_min = min(T1, T2)
        t_max = max(T1, T2)

        c1 = 1.0 - np.cos(np.pi / 2.0 - np.log(t_max / max(t_min, 0.109)) * 0.366)

        if t_max < 0.2:
            c2 = 1.0 - 0.105 * (1.0 - 1.0 / (1.0 + np.exp(100.0 * t_max - 5.0))) * (t_max - t_min) / (t_max - 0.0099)
        else:
            c2 = 0

        if t_max < 0.109:
            c3 = c2
        else:
            c3 = c1

        c4 = c1 + 0.5 * (np.sqrt(c3) - c3) * (1.0 + np.cos(np.pi * t_min / 0.109))

        if t_max <= 0.109:
            rho = c2
        elif t_min > 0.109:
            rho = c1
        elif t_max < 0.2:
            rho = min(c2, c4)
        else:
            rho = c4

        return rho

    @staticmethod
    def _AkkarCorrelationModel(T1, T2):
        """
        Details
        -------
        Valid for T = 0.01-4sec
        
        References
        ----------
        Akkar S., Sandikkaya MA., Ay BO., 2014, Compatible ground-motion prediction equations for damping scaling factors and vertical to
        horizontal spectral amplitude ratios for the broader Europe region, Bull Earthquake Eng, 12, pp. 517-547.
    
        Parameters
        ----------
        T1: float
            First period
        T2: float
            Second period
                
        Returns
        -------
        rho: float
             Predicted correlation coefficient
        """
        periods = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14,
                            0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3,
                            0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6,
                            0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.3, 1.4, 1.5,
                            1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4])

        coeff_table = np.fromstring("""
        1.000000000 0.998459319 0.992959119 0.982544599 0.968049306 0.935891712 0.935340260 0.932144334 0.931751872 0.926430615 0.923321139 0.918888775 0.917030483 0.911639361 0.907129682 0.902402109 0.896946675 0.888773486 0.877134367 0.860750213 0.845001970 0.831329156 0.819119695 0.807827754 0.796605604 0.786554638 0.774125984 0.763692083 0.754128870 0.744273823 0.734216226 0.721137528 0.698803347 0.670189828 0.648595259 0.632936382 0.616540526 0.600130851 0.586637070 0.570441490 0.556540670 0.541626641 0.528993811 0.518067400 0.494511874 0.476965753 0.467681175 0.459943411 0.467091314 0.466125719 0.465891941 0.460384183 0.447657660 0.441378795 0.419929380 0.402770275 0.409048639 0.407614656 0.396021594 0.388023676 0.374414280 0.327591227
        0.998459319 1.000000000 0.996082278 0.986394923 0.972268175 0.937204379 0.933767512 0.929590027 0.928836303 0.923372977 0.920334784 0.915475852 0.912990886 0.906737398 0.901559172 0.896179331 0.890297784 0.881662523 0.869699785 0.852589636 0.836312297 0.822782849 0.810458847 0.799463856 0.788538167 0.778435229 0.765844047 0.755309136 0.745418032 0.735529894 0.725528527 0.712604437 0.691230242 0.663650097 0.642851442 0.627911218 0.612094672 0.596170832 0.582828669 0.566721300 0.553020290 0.538275353 0.525393138 0.515402026 0.491995546 0.474471605 0.465274680 0.457684886 0.465508019 0.464989384 0.464672388 0.458927370 0.445960781 0.440101928 0.418578711 0.401214002 0.407688681 0.405891106 0.394456219 0.387127235 0.374517227 0.327475695
        0.992959119 0.996082278 1.000000000 0.992649681 0.980086257 0.941197598 0.932602717 0.926505266 0.924007674 0.917523889 0.913586346 0.907977885 0.904686047 0.896912271 0.890031983 0.883376006 0.875689545 0.864581272 0.850486709 0.831422657 0.814069837 0.799595438 0.786465899 0.775232085 0.764169780 0.753866668 0.741140232 0.730545470 0.720452785 0.710961627 0.701394512 0.688744127 0.668279624 0.641661255 0.621614187 0.607454047 0.592405222 0.577170682 0.563905922 0.547714590 0.534528870 0.520236877 0.507911645 0.499247836 0.476489272 0.459554239 0.450632108 0.443887177 0.453676323 0.454063440 0.454152128 0.447873723 0.434358458 0.428919977 0.407199580 0.389843410 0.397056695 0.395959027 0.384472121 0.379463066 0.366941903 0.319627396
        0.982544599 0.986394923 0.992649681 1.000000000 0.991868613 0.950464336 0.931563247 0.922118516 0.917281952 0.908226354 0.901735251 0.894372183 0.888666445 0.878296160 0.869775534 0.862499538 0.854123051 0.840604566 0.824033973 0.802856644 0.783631523 0.767268157 0.753425444 0.742402848 0.730649645 0.719527880 0.707013849 0.696663368 0.686897701 0.677316536 0.668071521 0.655568121 0.635949994 0.610222984 0.590334250 0.576468647 0.561786372 0.545516568 0.532012677 0.516083694 0.504048834 0.490431526 0.479238029 0.474326737 0.451941391 0.436068643 0.429005528 0.423980829 0.437159862 0.439031229 0.439119060 0.432376218 0.417956810 0.412327683 0.390718170 0.373967595 0.381848592 0.382643147 0.371369841 0.368779338 0.357389576 0.309845437
        0.968049306 0.972268175 0.980086257 0.991868613 1.000000000 0.962264296 0.934853114 0.922355940 0.914208237 0.902171642 0.892493486 0.883087807 0.874993779 0.862778465 0.852803082 0.843630234 0.833315659 0.816144288 0.797090103 0.773647699 0.752093427 0.734182280 0.719758912 0.708530778 0.695805176 0.683868700 0.670988645 0.660532586 0.651228551 0.641719672 0.632237674 0.619768824 0.600363810 0.575458810 0.554978237 0.540885496 0.526141485 0.509086783 0.495417398 0.480078437 0.469347369 0.456101348 0.445579100 0.441320331 0.420154846 0.404865399 0.400049713 0.395769563 0.410990864 0.414447111 0.414626756 0.407621549 0.392294122 0.385783148 0.364703220 0.348892179 0.357830909 0.361144444 0.347895826 0.346769536 0.335928741 0.287061335
        0.935891712 0.937204379 0.941197598 0.950464336 0.962264296 1.000000000 0.962615197 0.946983083 0.932736048 0.916504479 0.900697611 0.888448780 0.877582842 0.864031331 0.850833987 0.837208610 0.823523241 0.800934568 0.777560336 0.751236470 0.725236033 0.703276899 0.686195477 0.673138508 0.657750742 0.642103499 0.626024605 0.611941667 0.600548216 0.588997705 0.577499384 0.563139923 0.537065551 0.506654407 0.484245782 0.469499474 0.453353691 0.436416368 0.421481671 0.405428093 0.393791269 0.380915465 0.371943130 0.368567710 0.347857881 0.330967414 0.323162605 0.321046152 0.333837862 0.336077484 0.335971173 0.329302063 0.315903531 0.312372640 0.292619233 0.280248030 0.290766108 0.306970917 0.299373684 0.297351745 0.290186844 0.244808717
        0.935340260 0.933767512 0.932602717 0.931563247 0.934853114 0.962615197 1.000000000 0.990116513 0.975788829 0.961626511 0.947901807 0.936198102 0.923631053 0.907921861 0.892327730 0.877831671 0.863571118 0.838276836 0.812031528 0.781002635 0.752320254 0.729561627 0.710692090 0.694875287 0.678554053 0.664029001 0.647212006 0.631109273 0.617157180 0.602721669 0.588311292 0.572201368 0.542964437 0.509879102 0.485605758 0.468426247 0.451981143 0.435643477 0.421173623 0.404768931 0.391713589 0.378805455 0.371762114 0.364754462 0.342586661 0.326163442 0.317761408 0.313455213 0.322883212 0.326897555 0.326786547 0.320791399 0.306947067 0.305667553 0.287341638 0.275970914 0.286063278 0.300263857 0.293958754 0.290136746 0.282338477 0.246124740
        0.932144334 0.929590027 0.926505266 0.922118516 0.922355940 0.946983083 0.990116513 1.000000000 0.989689204 0.975580866 0.961221799 0.948766314 0.935866119 0.920979743 0.905594068 0.890796992 0.875864129 0.848031444 0.820169896 0.788388068 0.760461637 0.737174069 0.718138704 0.702866619 0.685968272 0.671563802 0.654484046 0.637750236 0.624565342 0.610451743 0.595774089 0.579109592 0.546793309 0.510790181 0.486449733 0.468315941 0.450400928 0.433411425 0.418927736 0.402123223 0.388393163 0.374883426 0.368103453 0.357924339 0.335222120 0.318370632 0.309724837 0.305682446 0.313064776 0.316347853 0.316156760 0.310549151 0.297864416 0.298356521 0.281200152 0.271399645 0.283219496 0.293617699 0.287088479 0.283841030 0.277461634 0.243767437
        0.931751872 0.928836303 0.924007674 0.917281952 0.914208237 0.932736048 0.975788829 0.989689204 1.000000000 0.990692102 0.976279079 0.963261390 0.949927381 0.935380214 0.920197544 0.905478473 0.889847615 0.860228360 0.832742946 0.801648000 0.774197968 0.750437814 0.730425623 0.713926358 0.695476994 0.679925434 0.662604323 0.645538715 0.632107890 0.617992508 0.603504205 0.586348664 0.554477063 0.518415584 0.492098090 0.474119641 0.455368512 0.437661685 0.423477464 0.406548167 0.391564950 0.377626453 0.369422885 0.356726995 0.333375827 0.316061615 0.307808198 0.303464717 0.309172957 0.313023066 0.313680039 0.308066980 0.295076975 0.295461037 0.278371488 0.268622071 0.282411518 0.290206583 0.285660879 0.282668509 0.276289052 0.246189710
        0.926430615 0.923372977 0.917523889 0.908226354 0.902171642 0.916504479 0.961626511 0.975580866 0.990692102 1.000000000 0.990603920 0.977076607 0.963343377 0.948772358 0.933778755 0.918866639 0.902998893 0.872080552 0.843009308 0.810775785 0.781796847 0.757163377 0.736115563 0.717277070 0.697503117 0.681458848 0.663781344 0.646399290 0.632348726 0.618139584 0.603625471 0.586337604 0.555585542 0.520652476 0.493648277 0.474623746 0.455144819 0.437389777 0.424032826 0.407091063 0.392164268 0.377080765 0.367321133 0.353822231 0.330522280 0.312790035 0.305808990 0.301518184 0.305559435 0.310789045 0.311188137 0.306296417 0.295131464 0.295872886 0.278990978 0.267324448 0.282158234 0.292110467 0.289709619 0.286648955 0.279832113 0.251575366
        0.923321139 0.920334784 0.913586346 0.901735251 0.892493486 0.900697611 0.947901807 0.961221799 0.976279079 0.990603920 1.000000000 0.991234852 0.977263347 0.962050918 0.947410572 0.932539689 0.916532455 0.886035092 0.856693255 0.824512878 0.795275895 0.771099389 0.749696094 0.730131944 0.710191567 0.693831629 0.675757722 0.657705784 0.642968742 0.627951273 0.613110783 0.595944989 0.564526872 0.529205600 0.501994765 0.483210664 0.463201523 0.445657936 0.432123543 0.414445146 0.398937719 0.383427624 0.372218848 0.355890544 0.332325762 0.313675494 0.307423499 0.301935949 0.303933831 0.309031533 0.309192298 0.304332821 0.293449546 0.293011088 0.274593213 0.262019434 0.278051027 0.288451408 0.286332638 0.282976017 0.276909997 0.249258668
        0.918888775 0.915475852 0.907977885 0.894372183 0.883087807 0.888448780 0.936198102 0.948766314 0.963261390 0.977076607 0.991234852 1.000000000 0.991461875 0.977249387 0.962684428 0.948288687 0.932397769 0.902401297 0.873944904 0.843125168 0.813827100 0.789043206 0.766759290 0.746920939 0.727166482 0.710549016 0.691277639 0.671783979 0.656435586 0.641231214 0.626363300 0.609761022 0.577624812 0.541308857 0.513780642 0.495161104 0.474055020 0.456058811 0.441204827 0.422575253 0.405919014 0.389474729 0.375384279 0.357490128 0.332760862 0.313967306 0.309698716 0.302466457 0.302581884 0.306199900 0.305430485 0.300673821 0.290090051 0.286175681 0.266462404 0.253842515 0.268766903 0.277634726 0.274937010 0.271795167 0.265717553 0.239620288
        0.917030483 0.912990886 0.904686047 0.888666445 0.874993779 0.877582842 0.923631053 0.935866119 0.949927381 0.963343377 0.977263347 0.991461875 1.000000000 0.991878466 0.978090238 0.964094767 0.948920678 0.919713105 0.891089102 0.861176855 0.833550120 0.809178023 0.786747467 0.766899229 0.747966585 0.731413777 0.712126250 0.692921732 0.677778238 0.662436373 0.647637045 0.631107390 0.597819630 0.561148767 0.533010875 0.513356262 0.491251279 0.472668535 0.457987065 0.439130079 0.422142181 0.405149095 0.390700404 0.371814153 0.346475259 0.327027891 0.321883357 0.314095902 0.312982403 0.315845362 0.313717071 0.309522978 0.299951916 0.294943875 0.275501618 0.262448020 0.276276401 0.281420646 0.279922024 0.276053857 0.270090412 0.245635104
        0.911639361 0.906737398 0.896912271 0.878296160 0.862778465 0.864031331 0.907921861 0.920979743 0.935380214 0.948772358 0.962050918 0.977249387 0.991878466 1.000000000 0.992580846 0.979232390 0.964160461 0.935228508 0.906816850 0.878590479 0.852353167 0.827736074 0.805368248 0.785788107 0.767419299 0.751070583 0.732484692 0.713720892 0.698680156 0.683301523 0.668330341 0.651886869 0.617640927 0.580147557 0.552081622 0.531657547 0.507920297 0.488383415 0.472946962 0.453366508 0.436991418 0.419899796 0.404944759 0.385413845 0.360995737 0.341744537 0.334913853 0.326304502 0.323121214 0.325465647 0.323318314 0.319365187 0.310962838 0.306165169 0.287931470 0.274861110 0.288000123 0.294133111 0.292219131 0.287580691 0.280332211 0.255837514
        0.907129682 0.901559172 0.890031983 0.869775534 0.852803082 0.850833987 0.892327730 0.905594068 0.920197544 0.933778755 0.947410572 0.962684428 0.978090238 0.992580846 1.000000000 0.992392183 0.978861488 0.950315509 0.922825230 0.896620663 0.871711984 0.847817658 0.825830210 0.805999642 0.788128435 0.771518085 0.753500199 0.735242143 0.720374704 0.704624017 0.689728673 0.673641884 0.638863972 0.600435509 0.571959727 0.550977150 0.526602351 0.506006960 0.490156485 0.470058406 0.454634719 0.437505595 0.421968892 0.400600713 0.375894609 0.356111071 0.347073209 0.337578577 0.333131292 0.334086897 0.332882535 0.329531653 0.322391412 0.317042426 0.300098589 0.286926765 0.302118806 0.306425395 0.305054461 0.299876324 0.290794284 0.264918559
        0.902402109 0.896179331 0.883376006 0.862499538 0.843630234 0.837208610 0.877831671 0.890796992 0.905478473 0.918866639 0.932539689 0.948288687 0.964094767 0.979232390 0.992392183 1.000000000 0.992919392 0.965554280 0.937751155 0.912374316 0.889244215 0.865558453 0.843304535 0.822777811 0.805076491 0.789310781 0.771800064 0.753686174 0.739043451 0.722863867 0.707648072 0.691643410 0.657405998 0.619981357 0.590898312 0.570314641 0.545948008 0.524637823 0.508238513 0.488292909 0.472911017 0.455759977 0.439640108 0.416870141 0.390834026 0.370873976 0.361475680 0.351910460 0.347919156 0.347781531 0.347503075 0.344891466 0.337223378 0.329883434 0.312553164 0.298647778 0.315799115 0.316085084 0.313419871 0.308003502 0.297064428 0.271319428
        0.896946675 0.890297784 0.875689545 0.854123051 0.833315659 0.823523241 0.863571118 0.875864129 0.889847615 0.902998893 0.916532455 0.932397769 0.948920678 0.964160461 0.978861488 0.992919392 1.000000000 0.981526197 0.953935871 0.928608956 0.906378431 0.882858511 0.860361416 0.839579829 0.822247992 0.806750736 0.789389481 0.771138927 0.756374021 0.740260214 0.724966726 0.709026958 0.674134702 0.637174840 0.607599704 0.586767148 0.562652094 0.540713747 0.523841189 0.504457115 0.489175229 0.472345733 0.455559793 0.431793892 0.404815844 0.383968164 0.373417718 0.362866364 0.358636506 0.357603963 0.356952598 0.354545886 0.346082342 0.337175955 0.318916422 0.304391592 0.321091927 0.318255597 0.315234055 0.308869052 0.296407372 0.267233297
        0.888773486 0.881662523 0.864581272 0.840604566 0.816144288 0.800934568 0.838276836 0.848031444 0.860228360 0.872080552 0.886035092 0.902401297 0.919713105 0.935228508 0.950315509 0.965554280 0.981526197 1.000000000 0.983188132 0.959078850 0.937509858 0.914343183 0.893018761 0.873570904 0.857413372 0.841800847 0.823681570 0.805410987 0.789930483 0.773231093 0.757567039 0.742559390 0.708014570 0.671646264 0.642542342 0.622684252 0.599796782 0.577728753 0.562207880 0.544157126 0.527351829 0.510342804 0.493508756 0.467832893 0.439714609 0.418897169 0.406342268 0.393734228 0.387257753 0.385395888 0.384024849 0.381568435 0.371714812 0.359793112 0.338666531 0.321962039 0.337149268 0.331189588 0.327291574 0.317595706 0.299746833 0.264459231
        0.877134367 0.869699785 0.850486709 0.824033973 0.797090103 0.777560336 0.812031528 0.820169896 0.832742946 0.843009308 0.856693255 0.873944904 0.891089102 0.906816850 0.922825230 0.937751155 0.953935871 0.983188132 1.000000000 0.985080494 0.963602517 0.941486205 0.921278855 0.902148604 0.886447380 0.870474302 0.852466430 0.834969148 0.819602813 0.803049263 0.788046832 0.773774047 0.738699435 0.703518087 0.673287574 0.652934282 0.631539980 0.610301027 0.594634258 0.576942969 0.559225210 0.541523783 0.520310447 0.492785876 0.464495956 0.443024302 0.429168012 0.416706147 0.409062703 0.405902074 0.404829344 0.401684269 0.390242030 0.377716151 0.355556786 0.338060659 0.351054868 0.343741052 0.337998036 0.327063915 0.309826727 0.276507212
        0.860750213 0.852589636 0.831422657 0.802856644 0.773647699 0.751236470 0.781002635 0.788388068 0.801648000 0.810775785 0.824512878 0.843125168 0.861176855 0.878590479 0.896620663 0.912374316 0.928608956 0.959078850 0.985080494 1.000000000 0.987076214 0.966251550 0.946367672 0.927192867 0.911111755 0.895184724 0.878182524 0.861043012 0.845272097 0.828166207 0.812508502 0.797578221 0.763621195 0.728218280 0.697059999 0.675306078 0.653618636 0.632259339 0.616606764 0.598515398 0.579387213 0.561003780 0.536573129 0.506393869 0.478261417 0.456448470 0.443626467 0.431722335 0.423844795 0.419800364 0.418384709 0.414824131 0.405052153 0.392564773 0.371408735 0.354258036 0.364224754 0.354811270 0.349292709 0.338303391 0.321379107 0.286839305
        0.845001970 0.836312297 0.814069837 0.783631523 0.752093427 0.725236033 0.752320254 0.760461637 0.774197968 0.781796847 0.795275895 0.813827100 0.833550120 0.852353167 0.871711984 0.889244215 0.906378431 0.937509858 0.963602517 0.987076214 1.000000000 0.987765466 0.968682433 0.949866317 0.934015674 0.917973989 0.901245207 0.883761969 0.867553991 0.849570378 0.833425822 0.818373340 0.785161324 0.750268840 0.719150115 0.697607895 0.675987578 0.654132757 0.638293914 0.620318278 0.600279507 0.581596469 0.556366577 0.524838746 0.495871759 0.474379462 0.461080448 0.449401990 0.441288735 0.435855549 0.433573677 0.429647421 0.419633298 0.408018661 0.386465311 0.368352442 0.376066786 0.362685030 0.355378232 0.343260457 0.325705273 0.290190152
        0.831329156 0.822782849 0.799595438 0.767268157 0.734182280 0.703276899 0.729561627 0.737174069 0.750437814 0.757163377 0.771099389 0.789043206 0.809178023 0.827736074 0.847817658 0.865558453 0.882858511 0.914343183 0.941486205 0.966251550 0.987765466 1.000000000 0.989282518 0.971720671 0.955831640 0.939380610 0.922432333 0.905063844 0.888609307 0.870698403 0.855015340 0.840698575 0.807476852 0.772566613 0.741638159 0.721218078 0.699089479 0.676804998 0.661404125 0.644427961 0.625063502 0.607281741 0.581131788 0.547731229 0.519871077 0.498788919 0.484955629 0.473600121 0.466111433 0.460100699 0.457359995 0.453103576 0.442158187 0.431052969 0.409003716 0.390815816 0.400032678 0.384069571 0.376313732 0.364472634 0.345928556 0.306793984
        0.819119695 0.810458847 0.786465899 0.753425444 0.719758912 0.686195477 0.710692090 0.718138704 0.730425623 0.736115563 0.749696094 0.766759290 0.786747467 0.805368248 0.825830210 0.843304535 0.860361416 0.893018761 0.921278855 0.946367672 0.968682433 0.989282518 1.000000000 0.990629419 0.974968587 0.958407552 0.941853460 0.925378025 0.909698324 0.892671841 0.877214810 0.862767082 0.829433708 0.793831747 0.762869579 0.741160415 0.718422078 0.696553647 0.682235979 0.665373624 0.646403599 0.629783687 0.602463002 0.567565408 0.539635481 0.518485506 0.503506571 0.491449857 0.483229527 0.477268008 0.474731561 0.470019281 0.457572144 0.447043980 0.424053991 0.405330662 0.413916872 0.395008131 0.385989904 0.373069949 0.353855693 0.313364781
        0.807827754 0.799463856 0.775232085 0.742402848 0.708530778 0.673138508 0.694875287 0.702866619 0.713926358 0.717277070 0.730131944 0.746920939 0.766899229 0.785788107 0.805999642 0.822777811 0.839579829 0.873570904 0.902148604 0.927192867 0.949866317 0.971720671 0.990629419 1.000000000 0.991282675 0.975928093 0.959847321 0.944296101 0.928890910 0.912357429 0.896957753 0.882018410 0.848157370 0.811757887 0.781082395 0.759724730 0.737819383 0.715817133 0.700585483 0.683203100 0.664038261 0.648323052 0.620371924 0.584327139 0.556168042 0.535213259 0.519933278 0.507589302 0.498505417 0.491561153 0.488620707 0.483146893 0.470172841 0.460470291 0.436623890 0.417916853 0.425563532 0.406239072 0.396941043 0.383166929 0.363913564 0.322319411
        0.796605604 0.788538167 0.764169780 0.730649645 0.695805176 0.657750742 0.678554053 0.685968272 0.695476994 0.697503117 0.710191567 0.727166482 0.747966585 0.767419299 0.788128435 0.805076491 0.822247992 0.857413372 0.886447380 0.911111755 0.934015674 0.955831640 0.974968587 0.991282675 1.000000000 0.992231729 0.977895397 0.962872808 0.947813449 0.931756306 0.916463866 0.901675012 0.867264671 0.831044135 0.801167763 0.780863356 0.759567978 0.737886670 0.722133559 0.705172595 0.686196619 0.669768852 0.641667578 0.606345410 0.578838233 0.557998057 0.541884883 0.528614796 0.518232765 0.510171160 0.505388571 0.498916704 0.486192237 0.477849233 0.454711228 0.435764958 0.440615871 0.421772989 0.412532101 0.397753600 0.376298229 0.334237875
        0.786554638 0.778435229 0.753866668 0.719527880 0.683868700 0.642103499 0.664029001 0.671563802 0.679925434 0.681458848 0.693831629 0.710549016 0.731413777 0.751070583 0.771518085 0.789310781 0.806750736 0.841800847 0.870474302 0.895184724 0.917973989 0.939380610 0.958407552 0.975928093 0.992231729 1.000000000 0.992886390 0.979229414 0.965013670 0.949508507 0.934415360 0.920269764 0.886803709 0.851776068 0.822415252 0.801817269 0.780666343 0.759635640 0.743702403 0.726734948 0.707464283 0.690327886 0.661915271 0.626890816 0.599082332 0.577875045 0.560794732 0.547203765 0.535994440 0.526901124 0.522052463 0.515492441 0.501977186 0.492295849 0.469513920 0.450580072 0.454558823 0.433855649 0.423004536 0.408583246 0.385674105 0.343985400
        0.774125984 0.765844047 0.741140232 0.707013849 0.670988645 0.626024605 0.647212006 0.654484046 0.662604323 0.663781344 0.675757722 0.691277639 0.712126250 0.732484692 0.753500199 0.771800064 0.789389481 0.823681570 0.852466430 0.878182524 0.901245207 0.922432333 0.941853460 0.959847321 0.977895397 0.992886390 1.000000000 0.992808304 0.980090092 0.965331640 0.950869846 0.937118859 0.904898169 0.871028107 0.842076036 0.820933796 0.799359151 0.777955636 0.761134676 0.743106649 0.723754413 0.706304648 0.677781683 0.642930622 0.615216256 0.593588952 0.575227915 0.561490726 0.549372457 0.539720625 0.535504521 0.529099197 0.515518910 0.505143371 0.482675351 0.463576007 0.467255409 0.444875501 0.430804997 0.417351819 0.395571593 0.354611000
        0.763692083 0.755309136 0.730545470 0.696663368 0.660532586 0.611941667 0.631109273 0.637750236 0.645538715 0.646399290 0.657705784 0.671783979 0.692921732 0.713720892 0.735242143 0.753686174 0.771138927 0.805410987 0.834969148 0.861043012 0.883761969 0.905063844 0.925378025 0.944296101 0.962872808 0.979229414 0.992808304 1.000000000 0.994185652 0.981802282 0.968440758 0.955409933 0.924320950 0.892540331 0.864393754 0.843527095 0.821581741 0.800999927 0.783559096 0.764937200 0.745564760 0.728161523 0.700357766 0.665526203 0.637592712 0.615790491 0.597177845 0.582705679 0.569768836 0.559412431 0.555081316 0.548653967 0.535255802 0.525695845 0.503170824 0.483643149 0.487189652 0.463713683 0.446156606 0.432274411 0.411058267 0.368638617
        0.754128870 0.745418032 0.720452785 0.686897701 0.651228551 0.600548216 0.617157180 0.624565342 0.632107890 0.632348726 0.642968742 0.656435586 0.677778238 0.698680156 0.720374704 0.739043451 0.756374021 0.789930483 0.819602813 0.845272097 0.867553991 0.888609307 0.909698324 0.928890910 0.947813449 0.965013670 0.980090092 0.994185652 1.000000000 0.994150208 0.982835607 0.970485740 0.939991268 0.909741470 0.881586720 0.860532215 0.838367280 0.818487924 0.801119936 0.782567374 0.763602019 0.746717998 0.719024783 0.683789946 0.655538484 0.633702095 0.614992987 0.600499845 0.587545104 0.576184757 0.571577541 0.565437777 0.552107561 0.542907232 0.519931980 0.500366990 0.502578859 0.476248728 0.456731904 0.443049297 0.422456988 0.380051200
        0.744273823 0.735529894 0.710961627 0.677316536 0.641719672 0.588997705 0.602721669 0.610451743 0.617992508 0.618139584 0.627951273 0.641231214 0.662436373 0.683301523 0.704624017 0.722863867 0.740260214 0.773231093 0.803049263 0.828166207 0.849570378 0.870698403 0.892671841 0.912357429 0.931756306 0.949508507 0.965331640 0.981802282 0.994150208 1.000000000 0.994555047 0.983719712 0.954312613 0.924174192 0.896260824 0.874840008 0.852317783 0.832037032 0.815217626 0.796680218 0.777923937 0.761547132 0.733472602 0.698582474 0.669915966 0.647582337 0.628420526 0.613959562 0.600818227 0.589218289 0.584936885 0.579064469 0.565956094 0.556431534 0.532382930 0.511986949 0.513872534 0.485714453 0.465636186 0.450877709 0.429516192 0.387956859
        0.734216226 0.725528527 0.701394512 0.668071521 0.632237674 0.577499384 0.588311292 0.595774089 0.603504205 0.603625471 0.613110783 0.626363300 0.647637045 0.668330341 0.689728673 0.707648072 0.724966726 0.757567039 0.788046832 0.812508502 0.833425822 0.855015340 0.877214810 0.896957753 0.916463866 0.934415360 0.950869846 0.968440758 0.982835607 0.994555047 1.000000000 0.994932974 0.968477098 0.939367475 0.911424608 0.889977691 0.867212691 0.846636991 0.829794296 0.811124571 0.792293383 0.776040171 0.747688255 0.713561606 0.685762837 0.664129457 0.644999957 0.631124803 0.618007559 0.606465425 0.601737888 0.596158189 0.582728871 0.573077355 0.548453559 0.528033221 0.529457292 0.501197105 0.482174544 0.467548386 0.446761605 0.405181228
        0.721137528 0.712604437 0.688744127 0.655568121 0.619768824 0.563139923 0.572201368 0.579109592 0.586348664 0.586337604 0.595944989 0.609761022 0.631107390 0.651886869 0.673641884 0.691643410 0.709026958 0.742559390 0.773774047 0.797578221 0.818373340 0.840698575 0.862767082 0.882018410 0.901675012 0.920269764 0.937118859 0.955409933 0.970485740 0.983719712 0.994932974 1.000000000 0.981638075 0.954092405 0.926524920 0.904877787 0.882388756 0.862391992 0.845381362 0.826792297 0.808581809 0.793082751 0.764927183 0.731068984 0.704570520 0.683596881 0.664736330 0.650453459 0.636667129 0.625260732 0.620739371 0.615430371 0.601426344 0.591997526 0.567241734 0.546880052 0.546781464 0.518704609 0.499605944 0.485384263 0.465544137 0.425482934
        0.698803347 0.691230242 0.668279624 0.635949994 0.600363810 0.537065551 0.542964437 0.546793309 0.554477063 0.555585542 0.564526872 0.577624812 0.597819630 0.617640927 0.638863972 0.657405998 0.674134702 0.708014570 0.738699435 0.763621195 0.785161324 0.807476852 0.829433708 0.848157370 0.867264671 0.886803709 0.904898169 0.924320950 0.939991268 0.954312613 0.968477098 0.981638075 1.000000000 0.983538995 0.957706913 0.935987500 0.914651577 0.896060700 0.880637230 0.862615735 0.844845397 0.830346503 0.801991687 0.768905919 0.743132239 0.721921168 0.703627843 0.689035140 0.674635233 0.663626956 0.659754589 0.654866569 0.639836919 0.629700713 0.604121667 0.582286864 0.579423117 0.552881805 0.533306855 0.519094244 0.500329740 0.465522040
        0.670189828 0.663650097 0.641661255 0.610222984 0.575458810 0.506654407 0.509879102 0.510790181 0.518415584 0.520652476 0.529205600 0.541308857 0.561148767 0.580147557 0.600435509 0.619981357 0.637174840 0.671646264 0.703518087 0.728218280 0.750268840 0.772566613 0.793831747 0.811757887 0.831044135 0.851776068 0.871028107 0.892540331 0.909741470 0.924174192 0.939367475 0.954092405 0.983538995 1.000000000 0.984806869 0.964020174 0.944105733 0.927117267 0.912213758 0.894385260 0.877046294 0.862770842 0.833322890 0.800972880 0.776004836 0.755464962 0.737592266 0.722535643 0.707403957 0.695789125 0.690373274 0.684148162 0.668454083 0.657335280 0.632856163 0.610867431 0.603220954 0.575140732 0.557374629 0.543168691 0.526976523 0.492870578
        0.648595259 0.642851442 0.621614187 0.590334250 0.554978237 0.484245782 0.485605758 0.486449733 0.492098090 0.493648277 0.501994765 0.513780642 0.533010875 0.552081622 0.571959727 0.590898312 0.607599704 0.642542342 0.673287574 0.697059999 0.719150115 0.741638159 0.762869579 0.781082395 0.801167763 0.822415252 0.842076036 0.864393754 0.881586720 0.896260824 0.911424608 0.926524920 0.957706913 0.984806869 1.000000000 0.988020841 0.968971132 0.952712011 0.938234892 0.921340738 0.904874530 0.890757284 0.861116679 0.829658593 0.805850152 0.785750422 0.767604210 0.752100140 0.735084454 0.722595689 0.715787268 0.709200479 0.693805940 0.681300783 0.656543099 0.633142087 0.621907186 0.591795296 0.573567528 0.560578412 0.544366449 0.510064415
        0.632936382 0.627911218 0.607454047 0.576468647 0.540885496 0.469499474 0.468426247 0.468315941 0.474119641 0.474623746 0.483210664 0.495161104 0.513356262 0.531657547 0.550977150 0.570314641 0.586767148 0.622684252 0.652934282 0.675306078 0.697607895 0.721218078 0.741160415 0.759724730 0.780863356 0.801817269 0.820933796 0.843527095 0.860532215 0.874840008 0.889977691 0.904877787 0.935987500 0.964020174 0.988020841 1.000000000 0.989458956 0.972837209 0.957207608 0.939870396 0.922988452 0.908409398 0.880065763 0.848911408 0.824047696 0.803986210 0.786179568 0.771028953 0.754201133 0.740603551 0.732520263 0.725460822 0.709601264 0.695778468 0.670348259 0.645039908 0.631700322 0.600662073 0.579974232 0.566198740 0.551510187 0.517257200
        0.616540526 0.612094672 0.592405222 0.561786372 0.526141485 0.453353691 0.451981143 0.450400928 0.455368512 0.455144819 0.463201523 0.474055020 0.491251279 0.507920297 0.526602351 0.545948008 0.562652094 0.599796782 0.631539980 0.653618636 0.675987578 0.699089479 0.718422078 0.737819383 0.759567978 0.780666343 0.799359151 0.821581741 0.838367280 0.852317783 0.867212691 0.882388756 0.914651577 0.944105733 0.968971132 0.989458956 1.000000000 0.990408854 0.975177292 0.958632867 0.942649345 0.928382259 0.901113039 0.870727077 0.845009399 0.824246463 0.806384739 0.791568177 0.774522102 0.759074388 0.749745124 0.741741703 0.724200868 0.710545791 0.685810148 0.660042678 0.645414202 0.614480633 0.592939430 0.579314389 0.564841776 0.530236122
        0.600130851 0.596170832 0.577170682 0.545516568 0.509086783 0.436416368 0.435643477 0.433411425 0.437661685 0.437389777 0.445657936 0.456058811 0.472668535 0.488383415 0.506006960 0.524637823 0.540713747 0.577728753 0.610301027 0.632259339 0.654132757 0.676804998 0.696553647 0.715817133 0.737886670 0.759635640 0.777955636 0.800999927 0.818487924 0.832037032 0.846636991 0.862391992 0.896060700 0.927117267 0.952712011 0.972837209 0.990408854 1.000000000 0.991733987 0.976909073 0.962537972 0.949136410 0.921870988 0.892485945 0.867343881 0.846295877 0.828619122 0.814638293 0.798568991 0.783111212 0.773161456 0.764747360 0.746822297 0.732786176 0.709329624 0.684850707 0.670670382 0.640059806 0.619237511 0.605167550 0.589955841 0.553791800
        0.586637070 0.582828669 0.563905922 0.532012677 0.495417398 0.421481671 0.421173623 0.418927736 0.423477464 0.424032826 0.432123543 0.441204827 0.457987065 0.472946962 0.490156485 0.508238513 0.523841189 0.562207880 0.594634258 0.616606764 0.638293914 0.661404125 0.682235979 0.700585483 0.722133559 0.743702403 0.761134676 0.783559096 0.801119936 0.815217626 0.829794296 0.845381362 0.880637230 0.912213758 0.938234892 0.957207608 0.975177292 0.991733987 1.000000000 0.992377958 0.979232998 0.965800580 0.939084462 0.911245119 0.886837082 0.865834217 0.847403772 0.833921319 0.818701100 0.803914476 0.794865503 0.786350505 0.767764507 0.752900160 0.729191661 0.705203128 0.693968207 0.662520328 0.640719458 0.626871803 0.611834493 0.575691608
        0.570441490 0.566721300 0.547714590 0.516083694 0.480078437 0.405428093 0.404768931 0.402123223 0.406548167 0.407091063 0.414445146 0.422575253 0.439130079 0.453366508 0.470058406 0.488292909 0.504457115 0.544157126 0.576942969 0.598515398 0.620318278 0.644427961 0.665373624 0.683203100 0.705172595 0.726734948 0.743106649 0.764937200 0.782567374 0.796680218 0.811124571 0.826792297 0.862615735 0.894385260 0.921340738 0.939870396 0.958632867 0.976909073 0.992377958 1.000000000 0.992940910 0.980287770 0.954405592 0.928704925 0.905747913 0.886158403 0.867291194 0.853743911 0.839055275 0.824901370 0.815971896 0.807383716 0.788481252 0.773058485 0.749552436 0.725430312 0.715306652 0.684390394 0.661759771 0.649062742 0.633832190 0.595807459
        0.556540670 0.553020290 0.534528870 0.504048834 0.469347369 0.393791269 0.391713589 0.388393163 0.391564950 0.392164268 0.398937719 0.405919014 0.422142181 0.436991418 0.454634719 0.472911017 0.489175229 0.527351829 0.559225210 0.579387213 0.600279507 0.625063502 0.646403599 0.664038261 0.686196619 0.707464283 0.723754413 0.745564760 0.763602019 0.777923937 0.792293383 0.808581809 0.844845397 0.877046294 0.904874530 0.922988452 0.942649345 0.962537972 0.979232998 0.992940910 1.000000000 0.993504221 0.970181484 0.945335811 0.923915471 0.905084353 0.886252938 0.872661766 0.858618893 0.844604622 0.835605457 0.826948653 0.808414658 0.792965711 0.769848189 0.745251644 0.734809306 0.705186975 0.683325524 0.671466883 0.655909098 0.616808159
        0.541626641 0.538275353 0.520236877 0.490431526 0.456101348 0.380915465 0.378805455 0.374883426 0.377626453 0.377080765 0.383427624 0.389474729 0.405149095 0.419899796 0.437505595 0.455759977 0.472345733 0.510342804 0.541523783 0.561003780 0.581596469 0.607281741 0.629783687 0.648323052 0.669768852 0.690327886 0.706304648 0.728161523 0.746717998 0.761547132 0.776040171 0.793082751 0.830346503 0.862770842 0.890757284 0.908409398 0.928382259 0.949136410 0.965800580 0.980287770 0.993504221 1.000000000 0.983960501 0.960158691 0.939613425 0.921947907 0.904582604 0.890713276 0.877104493 0.862695121 0.852413387 0.843449825 0.825422708 0.810842147 0.787762244 0.763535774 0.752315603 0.723105274 0.701638253 0.690121946 0.673817186 0.635307705
        0.528993811 0.525393138 0.507911645 0.479238029 0.445579100 0.371943130 0.371762114 0.368103453 0.369422885 0.367321133 0.372218848 0.375384279 0.390700404 0.404944759 0.421968892 0.439640108 0.455559793 0.493508756 0.520310447 0.536573129 0.556366577 0.581131788 0.602463002 0.620371924 0.641667578 0.661915271 0.677781683 0.700357766 0.719024783 0.733472602 0.747688255 0.764927183 0.801991687 0.833322890 0.861116679 0.880065763 0.901113039 0.921870988 0.939084462 0.954405592 0.970181484 0.983960501 1.000000000 0.985418150 0.965400860 0.947574842 0.931611460 0.917473723 0.903713867 0.889462692 0.879048666 0.870018452 0.851700395 0.837876146 0.815395745 0.791163409 0.779925869 0.750903010 0.730395857 0.717967558 0.700891537 0.664055282
        0.518067400 0.515402026 0.499247836 0.474326737 0.441320331 0.368567710 0.364754462 0.357924339 0.356726995 0.353822231 0.355890544 0.357490128 0.371814153 0.385413845 0.400600713 0.416870141 0.431793892 0.467832893 0.492785876 0.506393869 0.524838746 0.547731229 0.567565408 0.584327139 0.606345410 0.626890816 0.642930622 0.665526203 0.683789946 0.698582474 0.713561606 0.731068984 0.768905919 0.800972880 0.829658593 0.848911408 0.870727077 0.892485945 0.911245119 0.928704925 0.945335811 0.960158691 0.985418150 1.000000000 0.987181943 0.969560274 0.953707715 0.939787871 0.926192955 0.911688060 0.900327674 0.890977737 0.871625115 0.858428147 0.837467685 0.814512727 0.803058000 0.775111246 0.752923049 0.738669004 0.720369328 0.684394961
        0.494511874 0.491995546 0.476489272 0.451941391 0.420154846 0.347857881 0.342586661 0.335222120 0.333375827 0.330522280 0.332325762 0.332760862 0.346475259 0.360995737 0.375894609 0.390834026 0.404815844 0.439714609 0.464495956 0.478261417 0.495871759 0.519871077 0.539635481 0.556168042 0.578838233 0.599082332 0.615216256 0.637592712 0.655538484 0.669915966 0.685762837 0.704570520 0.743132239 0.776004836 0.805850152 0.824047696 0.845009399 0.867343881 0.886837082 0.905747913 0.923915471 0.939613425 0.965400860 0.987181943 1.000000000 0.989545067 0.973868486 0.960494874 0.946919518 0.932410480 0.921253735 0.911724860 0.892732944 0.878497770 0.857382559 0.835510836 0.823044717 0.797159548 0.776927194 0.762772025 0.743974698 0.706169171
        0.476965753 0.474471605 0.459554239 0.436068643 0.404865399 0.330967414 0.326163442 0.318370632 0.316061615 0.312790035 0.313675494 0.313967306 0.327027891 0.341744537 0.356111071 0.370873976 0.383968164 0.418897169 0.443024302 0.456448470 0.474379462 0.498788919 0.518485506 0.535213259 0.557998057 0.577875045 0.593588952 0.615790491 0.633702095 0.647582337 0.664129457 0.683596881 0.721921168 0.755464962 0.785750422 0.803986210 0.824246463 0.846295877 0.865834217 0.886158403 0.905084353 0.921947907 0.947574842 0.969560274 0.989545067 1.000000000 0.991217256 0.978036795 0.964074132 0.949752301 0.938072033 0.928156333 0.909188154 0.894858501 0.874422893 0.854208639 0.841685230 0.816003233 0.794462365 0.779529183 0.760413104 0.721304101
        0.467681175 0.465274680 0.450632108 0.429005528 0.400049713 0.323162605 0.317761408 0.309724837 0.307808198 0.305808990 0.307423499 0.309698716 0.321883357 0.334913853 0.347073209 0.361475680 0.373417718 0.406342268 0.429168012 0.443626467 0.461080448 0.484955629 0.503506571 0.519933278 0.541884883 0.560794732 0.575227915 0.597177845 0.614992987 0.628420526 0.644999957 0.664736330 0.703627843 0.737592266 0.767604210 0.786179568 0.806384739 0.828619122 0.847403772 0.867291194 0.886252938 0.904582604 0.931611460 0.953707715 0.973868486 0.991217256 1.000000000 0.992481165 0.979031709 0.964487154 0.951984229 0.941102825 0.921942653 0.906937861 0.887275050 0.866761340 0.854546917 0.830185558 0.809841631 0.793789262 0.774388186 0.734778759
        0.459943411 0.457684886 0.443887177 0.423980829 0.395769563 0.321046152 0.313455213 0.305682446 0.303464717 0.301518184 0.301935949 0.302466457 0.314095902 0.326304502 0.337578577 0.351910460 0.362866364 0.393734228 0.416706147 0.431722335 0.449401990 0.473600121 0.491449857 0.507589302 0.528614796 0.547203765 0.561490726 0.582705679 0.600499845 0.613959562 0.631124803 0.650453459 0.689035140 0.722535643 0.752100140 0.771028953 0.791568177 0.814638293 0.833921319 0.853743911 0.872661766 0.890713276 0.917473723 0.939787871 0.960494874 0.978036795 0.992481165 1.000000000 0.992680886 0.979415209 0.967423180 0.956408601 0.937147632 0.920515510 0.901559359 0.880728349 0.868467283 0.844973436 0.825545291 0.809170358 0.789750612 0.750310219
        0.467091314 0.465508019 0.453676323 0.437159862 0.410990864 0.333837862 0.322883212 0.313064776 0.309172957 0.305559435 0.303933831 0.302581884 0.312982403 0.323121214 0.333131292 0.347919156 0.358636506 0.387257753 0.409062703 0.423844795 0.441288735 0.466111433 0.483229527 0.498505417 0.518232765 0.535994440 0.549372457 0.569768836 0.587545104 0.600818227 0.618007559 0.636667129 0.674635233 0.707403957 0.735084454 0.754201133 0.774522102 0.798568991 0.818701100 0.839055275 0.858618893 0.877104493 0.903713867 0.926192955 0.946919518 0.964074132 0.979031709 0.992680886 1.000000000 0.993376358 0.982983662 0.972752863 0.954049179 0.936444204 0.917149007 0.896086823 0.883274424 0.860372144 0.840698066 0.824522815 0.803606701 0.763724118
        0.466125719 0.464989384 0.454063440 0.439031229 0.414447111 0.336077484 0.326897555 0.316347853 0.313023066 0.310789045 0.309031533 0.306199900 0.315845362 0.325465647 0.334086897 0.347781531 0.357603963 0.385395888 0.405902074 0.419800364 0.435855549 0.460100699 0.477268008 0.491561153 0.510171160 0.526901124 0.539720625 0.559412431 0.576184757 0.589218289 0.606465425 0.625260732 0.663626956 0.695789125 0.722595689 0.740603551 0.759074388 0.783111212 0.803914476 0.824901370 0.844604622 0.862695121 0.889462692 0.911688060 0.932410480 0.949752301 0.964487154 0.979415209 0.993376358 1.000000000 0.994831778 0.985876009 0.968379551 0.950555401 0.931415874 0.911013019 0.896470404 0.873284821 0.853349923 0.838230714 0.816197107 0.777354214
        0.465891941 0.464672388 0.454152128 0.439119060 0.414626756 0.335971173 0.326786547 0.316156760 0.313680039 0.311188137 0.309192298 0.305430485 0.313717071 0.323318314 0.332882535 0.347503075 0.356952598 0.384024849 0.404829344 0.418384709 0.433573677 0.457359995 0.474731561 0.488620707 0.505388571 0.522052463 0.535504521 0.555081316 0.571577541 0.584936885 0.601737888 0.620739371 0.659754589 0.690373274 0.715787268 0.732520263 0.749745124 0.773161456 0.794865503 0.815971896 0.835605457 0.852413387 0.879048666 0.900327674 0.921253735 0.938072033 0.951984229 0.967423180 0.982983662 0.994831778 1.000000000 0.995572903 0.978819124 0.960520046 0.941679844 0.921935693 0.906534958 0.883214300 0.862440172 0.846539221 0.825195593 0.788230830
        0.460384183 0.458927370 0.447873723 0.432376218 0.407621549 0.329302063 0.320791399 0.310549151 0.308066980 0.306296417 0.304332821 0.300673821 0.309522978 0.319365187 0.329531653 0.344891466 0.354545886 0.381568435 0.401684269 0.414824131 0.429647421 0.453103576 0.470019281 0.483146893 0.498916704 0.515492441 0.529099197 0.548653967 0.565437777 0.579064469 0.596158189 0.615430371 0.654866569 0.684148162 0.709200479 0.725460822 0.741741703 0.764747360 0.786350505 0.807383716 0.826948653 0.843449825 0.870018452 0.890977737 0.911724860 0.928156333 0.941102825 0.956408601 0.972752863 0.985876009 0.995572903 1.000000000 0.987265647 0.968490735 0.949398628 0.929693100 0.914076122 0.890966585 0.870858532 0.855151540 0.834216891 0.798942709
        0.447657660 0.445960781 0.434358458 0.417956810 0.392294122 0.315903531 0.306947067 0.297864416 0.295076975 0.295131464 0.293449546 0.290090051 0.299951916 0.310962838 0.322391412 0.337223378 0.346082342 0.371714812 0.390242030 0.405052153 0.419633298 0.442158187 0.457572144 0.470172841 0.486192237 0.501977186 0.515518910 0.535255802 0.552107561 0.565956094 0.582728871 0.601426344 0.639836919 0.668454083 0.693805940 0.709601264 0.724200868 0.746822297 0.767764507 0.788481252 0.808414658 0.825422708 0.851700395 0.871625115 0.892732944 0.909188154 0.921942653 0.937147632 0.954049179 0.968379551 0.978819124 0.987265647 1.000000000 0.988905546 0.971156177 0.952268832 0.936202255 0.915648423 0.898224725 0.882650968 0.861796234 0.830129231
        0.441378795 0.440101928 0.428919977 0.412327683 0.385783148 0.312372640 0.305667553 0.298356521 0.295461037 0.295872886 0.293011088 0.286175681 0.294943875 0.306165169 0.317042426 0.329883434 0.337175955 0.359793112 0.377716151 0.392564773 0.408018661 0.431052969 0.447043980 0.460470291 0.477849233 0.492295849 0.505143371 0.525695845 0.542907232 0.556431534 0.573077355 0.591997526 0.629700713 0.657335280 0.681300783 0.695778468 0.710545791 0.732786176 0.752900160 0.773058485 0.792965711 0.810842147 0.837876146 0.858428147 0.878497770 0.894858501 0.906937861 0.920515510 0.936444204 0.950555401 0.960520046 0.968490735 0.988905546 1.000000000 0.989935594 0.973239320 0.958068275 0.940025622 0.923972912 0.909276123 0.890558074 0.863419172
        0.419929380 0.418578711 0.407199580 0.390718170 0.364703220 0.292619233 0.287341638 0.281200152 0.278371488 0.278990978 0.274593213 0.266462404 0.275501618 0.287931470 0.300098589 0.312553164 0.318916422 0.338666531 0.355556786 0.371408735 0.386465311 0.409003716 0.424053991 0.436623890 0.454711228 0.469513920 0.482675351 0.503170824 0.519931980 0.532382930 0.548453559 0.567241734 0.604121667 0.632856163 0.656543099 0.670348259 0.685810148 0.709329624 0.729191661 0.749552436 0.769848189 0.787762244 0.815395745 0.837467685 0.857382559 0.874422893 0.887275050 0.901559359 0.917149007 0.931415874 0.941679844 0.949398628 0.971156177 0.989935594 1.000000000 0.991691687 0.978763164 0.963690232 0.948213960 0.933929338 0.916174312 0.891643991
        0.402770275 0.401214002 0.389843410 0.373967595 0.348892179 0.280248030 0.275970914 0.271399645 0.268622071 0.267324448 0.262019434 0.253842515 0.262448020 0.274861110 0.286926765 0.298647778 0.304391592 0.321962039 0.338060659 0.354258036 0.368352442 0.390815816 0.405330662 0.417916853 0.435764958 0.450580072 0.463576007 0.483643149 0.500366990 0.511986949 0.528033221 0.546880052 0.582286864 0.610867431 0.633142087 0.645039908 0.660042678 0.684850707 0.705203128 0.725430312 0.745251644 0.763535774 0.791163409 0.814512727 0.835510836 0.854208639 0.866761340 0.880728349 0.896086823 0.911013019 0.921935693 0.929693100 0.952268832 0.973239320 0.991691687 1.000000000 0.993595248 0.980406376 0.965648634 0.950884209 0.934166833 0.911343164
        0.409048639 0.407688681 0.397056695 0.381848592 0.357830909 0.290766108 0.286063278 0.283219496 0.282411518 0.282158234 0.278051027 0.268766903 0.276276401 0.288000123 0.302118806 0.315799115 0.321091927 0.337149268 0.351054868 0.364224754 0.376066786 0.400032678 0.413916872 0.425563532 0.440615871 0.454558823 0.467255409 0.487189652 0.502578859 0.513872534 0.529457292 0.546781464 0.579423117 0.603220954 0.621907186 0.631700322 0.645414202 0.670670382 0.693968207 0.715306652 0.734809306 0.752315603 0.779925869 0.803058000 0.823044717 0.841685230 0.854546917 0.868467283 0.883274424 0.896470404 0.906534958 0.914076122 0.936202255 0.958068275 0.978763164 0.993595248 1.000000000 0.993026697 0.981015917 0.967271742 0.951464983 0.931015807
        0.407614656 0.405891106 0.395959027 0.382643147 0.361144444 0.306970917 0.300263857 0.293617699 0.290206583 0.292110467 0.288451408 0.277634726 0.281420646 0.294133111 0.306425395 0.316085084 0.318255597 0.331189588 0.343741052 0.354811270 0.362685030 0.384069571 0.395008131 0.406239072 0.421772989 0.433855649 0.444875501 0.463713683 0.476248728 0.485714453 0.501197105 0.518704609 0.552881805 0.575140732 0.591795296 0.600662073 0.614480633 0.640059806 0.662520328 0.684390394 0.705186975 0.723105274 0.750903010 0.775111246 0.797159548 0.816003233 0.830185558 0.844973436 0.860372144 0.873284821 0.883214300 0.890966585 0.915648423 0.940025622 0.963690232 0.980406376 0.993026697 1.000000000 0.993978542 0.981867206 0.966916287 0.948046812
        0.396021594 0.394456219 0.384472121 0.371369841 0.347895826 0.299373684 0.293958754 0.287088479 0.285660879 0.289709619 0.286332638 0.274937010 0.279922024 0.292219131 0.305054461 0.313419871 0.315234055 0.327291574 0.337998036 0.349292709 0.355378232 0.376313732 0.385989904 0.396941043 0.412532101 0.423004536 0.430804997 0.446156606 0.456731904 0.465636186 0.482174544 0.499605944 0.533306855 0.557374629 0.573567528 0.579974232 0.592939430 0.619237511 0.640719458 0.661759771 0.683325524 0.701638253 0.730395857 0.752923049 0.776927194 0.794462365 0.809841631 0.825545291 0.840698066 0.853349923 0.862440172 0.870858532 0.898224725 0.923972912 0.948213960 0.965648634 0.981015917 0.993978542 1.000000000 0.994161539 0.982017362 0.965411448
        0.388023676 0.387127235 0.379463066 0.368779338 0.346769536 0.297351745 0.290136746 0.283841030 0.282668509 0.286648955 0.282976017 0.271795167 0.276053857 0.287580691 0.299876324 0.308003502 0.308869052 0.317595706 0.327063915 0.338303391 0.343260457 0.364472634 0.373069949 0.383166929 0.397753600 0.408583246 0.417351819 0.432274411 0.443049297 0.450877709 0.467548386 0.485384263 0.519094244 0.543168691 0.560578412 0.566198740 0.579314389 0.605167550 0.626871803 0.649062742 0.671466883 0.690121946 0.717967558 0.738669004 0.762772025 0.779529183 0.793789262 0.809170358 0.824522815 0.838230714 0.846539221 0.855151540 0.882650968 0.909276123 0.933929338 0.950884209 0.967271742 0.981867206 0.994161539 1.000000000 0.994786494 0.982542990
        0.374414280 0.374517227 0.366941903 0.357389576 0.335928741 0.290186844 0.282338477 0.277461634 0.276289052 0.279832113 0.276909997 0.265717553 0.270090412 0.280332211 0.290794284 0.297064428 0.296407372 0.299746833 0.309826727 0.321379107 0.325705273 0.345928556 0.353855693 0.363913564 0.376298229 0.385674105 0.395571593 0.411058267 0.422456988 0.429516192 0.446761605 0.465544137 0.500329740 0.526976523 0.544366449 0.551510187 0.564841776 0.589955841 0.611834493 0.633832190 0.655909098 0.673817186 0.700891537 0.720369328 0.743974698 0.760413104 0.774388186 0.789750612 0.803606701 0.816197107 0.825195593 0.834216891 0.861796234 0.890558074 0.916174312 0.934166833 0.951464983 0.966916287 0.982017362 0.994786494 1.000000000 0.994598788
        0.327591227 0.327475695 0.319627396 0.309845437 0.287061335 0.244808717 0.246124740 0.243767437 0.246189710 0.251575366 0.249258668 0.239620288 0.245635104 0.255837514 0.264918559 0.271319428 0.267233297 0.264459231 0.276507212 0.286839305 0.290190152 0.306793984 0.313364781 0.322319411 0.334237875 0.343985400 0.354611000 0.368638617 0.380051200 0.387956859 0.405181228 0.425482934 0.465522040 0.492870578 0.510064415 0.517257200 0.530236122 0.553791800 0.575691608 0.595807459 0.616808159 0.635307705 0.664055282 0.684394961 0.706169171 0.721304101 0.734778759 0.750310219 0.763724118 0.777354214 0.788230830 0.798942709 0.830129231 0.863419172 0.891643991 0.911343164 0.931015807 0.948046812 0.965411448 0.982542990 0.994598788 1.000000000
        """, dtype=float, sep=" ").reshape(-1, len(periods))

        if np.any([T1, T2] < periods[0]) or \
                np.any([T1, T2] > periods[-1]):
            raise ValueError("Period array contains values outside of the "
                             "range supported by the Akkar et al. (2014) "
                             "correlation model")

        if T1 == T2:
            rho = 1.0
        else:
            rho = interpolate.interp2d(periods, periods, coeff_table, kind='linear')(T1, T2)[0]

        return rho

    def _get_correlation(self, T1, T2):
        """
        Details
        -------
        Compute the inter-period correlation for any two Sa(T) values.
        
        Parameters
        ----------
        T1: float
            First period
        T2: float
            Second period
                
        Returns
        -------
        rho: float
             Predicted correlation coefficient
        """

        correlation_function_handles = {
            'baker_jayaram': self._BakerJayaramCorrelationModel,
            'akkar': self._AkkarCorrelationModel,
        }

        # Check for existing correlation function
        if self.corr_func not in correlation_function_handles:
            raise ValueError('Not a valid correlation function')
        else:
            rho = \
                correlation_function_handles[self.corr_func](T1, T2)

        return rho

    def _gmpe_sb_2014_ratios(self, T):
        """
        Details
        -------
        Computes Sa_RotD100/Sa_RotD50 ratios.

        References
        ----------
        Shahi, S. K., and Baker, J. W. (2014). "NGA-West2 models for ground-
        motion directionality." Earthquake Spectra, 30(3), 1285-1300.

        Parameters
        ----------
        T: numpy.ndarray
            Period(s) of interest (sec)

        Returns
        -------
        ratio: float
             geometric mean of Sa_RotD100/Sa_RotD50
        sigma: float
            standard deviation of log(Sa_RotD100/Sa_RotD50)
        """

        # Model coefficient values from Table 1 of the above-reference paper
        periods_orig = np.array(
            [0.0100000000000000, 0.0200000000000000, 0.0300000000000000, 0.0500000000000000, 0.0750000000000000,
             0.100000000000000, 0.150000000000000, 0.200000000000000, 0.250000000000000, 0.300000000000000,
             0.400000000000000, 0.500000000000000, 0.750000000000000, 1, 1.50000000000000, 2, 3, 4, 5,
             7.50000000000000, 10])
        ratios_orig = np.array(
            [1.19243805900000, 1.19124621700000, 1.18767783300000, 1.18649074900000, 1.18767783300000,
             1.18767783300000, 1.19961419400000, 1.20562728500000, 1.21652690500000, 1.21896239400000,
             1.22875320400000, 1.22875320400000, 1.23738465100000, 1.24110237900000, 1.24234410200000,
             1.24358706800000, 1.24732343100000, 1.25985923900000, 1.264908769000, 1.28531008400000,
             1.29433881900000])
        sigma_orig = np.array(
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
             0.08, 0.08, 0.08, 0.08])

        # Interpolate to compute values for the user-specified periods
        f = interpolate.interp1d(np.log(periods_orig), ratios_orig)(np.log(T))
        ratio = interpolate.interp1d(np.log(periods_orig), ratios_orig)(np.log(T))
        sigma = interpolate.interp1d(np.log(periods_orig), sigma_orig)(np.log(T))

        return ratio, sigma

    def _get_cond_param(self, sctx, rctx, dctx):
        """
        Details
        -------
        This function calculates the logarithmic mean and standard deviation of intensity measure
        predicted by the selected GMPM at conditioning periods.
        Moreover, it calculates the correlation coefficients between any period and conditioning period Tstar.
    
        Parameters
        ----------
        sctx : openquake.hazardlib.gsim object
            An instance of SitesContext with sites information to calculate PoEs on.
        rctx : openquake.hazardlib.gsim object
            An instance of RuptureContext with a single rupture information.
        dctx : openquake.hazardlib.gsim object
            An instance of DistancesContext with information about the distances between sites and a rupture.
    
        Returns
        -------
        mu_lnSaTstar : float
            Logarithmic mean of intensity measure according to the selected GMPM.
        sigma_lnSaTstar : float
           Logarithmic standard deviation of intensity measure according to the selected GMPM.
        rho_T_Tstar : numpy.array
            Correlation coefficients.
        """

        n = len(self.Tstar)
        mu_lnSaT = np.zeros(n)
        sigma_lnSaT = np.zeros(n)
        MoC = np.zeros((n, n))

        # Get the GMPE output
        for i in range(n):
            mu_lnSaT[i], stddvs_lnSa = self.bgmpe.get_mean_and_stddevs(sctx, rctx, dctx, imt.SA(period=self.Tstar[i]),
                                                                       [const.StdDev.TOTAL])
            sigma_lnSaT[i] = stddvs_lnSa[0]

            # modify spectral targets if RotD100 values were specified for two-component selection
            if self.Sa_def == 'RotD100' and not 'RotD100' in self.bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT and self.selection == 2:
                rotD100Ratio, rotD100Sigma = self._gmpe_sb_2014_ratios(self.Tstar[i])
                mu_lnSaT[i] = mu_lnSaT[i] + np.log(rotD100Ratio)
                sigma_lnSaT[i] = (sigma_lnSaT[i] ** 2 + rotD100Sigma ** 2) ** 0.5

            for j in range(n):
                rho = self._get_correlation(self.Tstar[i], self.Tstar[j])
                MoC[i, j] = rho

        Sa_avg_meanLn = (1 / n) * sum(mu_lnSaT)  # logarithmic mean of AvgSa

        Sa_avg_std = 0
        for i in range(n):
            for j in range(n):
                Sa_avg_std = Sa_avg_std + (MoC[i, j] * sigma_lnSaT[i] * sigma_lnSaT[j])  # logarithmic Var of the AvgSa

        Sa_avg_std = Sa_avg_std * (1 / n) ** 2

        # compute mean of logarithmic average spectral acceleration and logarithmic standard deviation of spectral
        # acceleration prediction
        mu_lnSaTstar = Sa_avg_meanLn
        sigma_lnSaTstar = np.sqrt(Sa_avg_std)

        # compute correlation coefficients
        rho_T_Tstar = np.zeros(len(self.T))
        for i in range(len(self.T)):
            for j in range(len(self.Tstar)):
                rho_bj = self._get_correlation(self.T[i], self.Tstar[j])
                rho_T_Tstar[i] = rho_bj * sigma_lnSaT[j] + rho_T_Tstar[i]

            rho_T_Tstar[i] = rho_T_Tstar[i] / (len(self.Tstar) * sigma_lnSaTstar)

        return mu_lnSaTstar, sigma_lnSaTstar, rho_T_Tstar

    def _set_contexts(self, index):

        """
        Details
        -------
        Sets the parameters for the computation of a ground motion model. If
        not defined by the user as input parameters, most parameters (dip,
        hypocentral depth, fault width, ztor, azimuth, source-to-site distances
        based on extended sources, z2pt5, z1pt0) are defined according to the
        relationships included in Kaklamanos et al. 2011.

        References
        ----------
        Kaklamanos J, Baise LG, Boore DM. (2011) Estimating unknown input parameters
        when implementing the NGA ground-motion prediction equations in engineering
        practice. Earthquake Spectra 27: 1219-1235.
        https://doi.org/10.1193/1.3650372.

        Parameters
        ----------
        index: int
            The scenario index for which gmm attributes are set

        Returns
        -------
        sctx : openquake.hazardlib.contexts.SitesContext
            An instance of SitesContext with sites information to calculate PoEs on.
        rctx : openquake.hazardlib.contexts.RuptureContext
            An instance of RuptureContext with a single rupture information.
        dctx : openquake.hazardlib.contexts.DistancesContext
            An instance of DistancesContext with information about the distances between sites and a rupture.
        """

        # Initialize, the contexts for the scenario
        sctx = gsim.base.SitesContext()
        rctx = gsim.base.RuptureContext()
        dctx = gsim.base.DistancesContext()

        # RUPTURE PARAMETERS
        # -------------------------------------
        mag = self.rup_param['mag'][index]  # Earthquake magnitude
        rake = self.rup_param['rake'][index]  # Fault rake

        # Hypocentral depth
        if 'hypo_depth' in self.rup_param.keys():
            hypo_depth = self.rup_param['hypo_depth'][index]
        else:
            if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
                hypo_depth = 5.63 + 0.68 * mag
            else:
                hypo_depth = 11.24 - 0.2 * mag

        # Fault dip
        if 'dip' in self.rup_param.keys():
            dip = self.rup_param['dip'][index]
        else:
            if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
                dip = 90
            elif rake > 0:
                dip = 40
            else:
                dip = 50

        # Upper and lower seismogenic depths
        if 'upper_sd' in self.rup_param.keys():
            upper_sd = self.rup_param['upper_sd'][index]
        else:
            upper_sd = 0
        if 'lower_sd' in self.rup_param.keys():
            lower_sd = self.rup_param['lower_sd'][index]
        else:
            lower_sd = 500

        # Rupture width and depth to top of coseismic rupture (km)
        if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            width = 10.0 ** (-0.76 + 0.27 * mag)
        elif rake > 0:
            # thrust/reverse
            width = 10.0 ** (-1.61 + 0.41 * mag)
        else:
            # normal
            width = 10.0 ** (-1.14 + 0.35 * mag)
        source_vertical_width = width * np.sin(np.radians(dip))
        ztor = max(hypo_depth - 0.6 * source_vertical_width, upper_sd)
        if (ztor + source_vertical_width) > lower_sd:
            source_vertical_width = lower_sd - ztor
            width = source_vertical_width / np.sin(np.radians(dip))
        if 'width' in self.rup_param.keys():
            width = self.rup_param['width'][index]
        if 'ztor' in self.rup_param.keys():
            ztor = self.rup_param['ztor'][index]

        # Hanging-wall factor
        if 'fhw' in self.rup_param.keys():
            fhw = self.rup_param['fhw'][index]
        else:
            fhw = 0

        # Source-to-site azimuth, alternative of hanging wall factor
        if 'azimuth' in self.rup_param.keys():
            azimuth = self.rup_param['azimuth'][index]
        else:
            if fhw == 1:
                azimuth = 50
            elif fhw == 0:
                azimuth = -50

        # Fault rake
        setattr(rctx, 'rake', np.array([rake], dtype='float64'))
        # Fault dip
        setattr(rctx, 'dip', np.array([dip], dtype='float64'))
        # Earthquake magnitude
        setattr(rctx, 'mag', np.array([mag], dtype='float64'))
        # Rupture width
        setattr(rctx, 'width', np.array([width], dtype='float64'))
        # Hypocentral depth of the rupture
        setattr(rctx, 'hypo_depth', np.array([hypo_depth], dtype='float64'))
        # Depth to top of coseismic rupture (km)
        setattr(rctx, 'ztor', np.array([ztor], dtype='float64'))
        # Annual rate of occurrence, not really required, setting this to zero
        setattr(rctx, 'occurrence_rate', np.array([0.0], dtype='float64'))
        # Do another loop in case some other rupture parameters which I do not recall are used.
        pass_keys = ['azimuth', 'fhw', 'lower_sd', 'upper_sd']
        for key in self.rup_param.keys():
            if not key in pass_keys:
                temp = np.array([self.rup_param[key][index]], dtype='float64')
                setattr(rctx, key, temp)

        # DISTANCE PARAMETERS
        # -------------------------------------
        # rjb and rx
        if 'rjb' in self.dist_param.keys():
            rjb = self.dist_param['rjb'][index]
            if rjb == 0:
                rx = 0.5 * width * np.cos(np.radians(dip))
            else:
                if dip == 90:
                    rx = rjb * np.sin(np.radians(azimuth))
                else:
                    if (0 <= azimuth < 90) or (90 < azimuth <= 180):
                        if rjb * np.abs(np.tan(np.radians(azimuth))) <= width * np.cos(np.radians(dip)):
                            rx = rjb * np.abs(np.tan(np.radians(azimuth)))
                        else:
                            rx = rjb * np.tan(np.radians(azimuth)) * np.cos(np.radians(azimuth) -
                                                                            np.arcsin(width * np.cos(
                                                                                np.radians(dip)) * np.cos(
                                                                                np.radians(azimuth)) / rjb))
                    elif azimuth == 90:  # we assume that Rjb>0
                        rx = rjb + width * np.cos(np.radians(dip))
                    else:
                        rx = rjb * np.sin(np.radians(azimuth))
        elif 'rx' in self.dist_param.keys():
            rx = self.dist_param['rx'][index]
            rjb = None
        else:
            rx = None

        # ry0
        if azimuth == 90 or azimuth == -90:
            ry0 = 0
        elif azimuth == 0 or azimuth == 180 or azimuth == -180 and rjb:
            ry0 = rjb
        elif rx:
            ry0 = np.abs(rx * 1. / np.tan(np.radians(azimuth)))
        else:
            ry0 = None

        # rrup
        if rjb and dip == 90:
            rrup = np.sqrt(np.square(rjb) + np.square(ztor))
        elif rx:
            if rx < ztor * np.tan(np.radians(dip)):
                rrup1 = np.sqrt(np.square(rx) + np.square(ztor))
            if ztor * np.tan(np.radians(dip)) <= rx <= ztor * np.tan(np.radians(dip)) + width * 1. / np.cos(
                    np.radians(dip)):
                rrup1 = rx * np.sin(np.radians(dip)) + ztor * np.cos(np.radians(dip))
            if rx > ztor * np.tan(np.radians(dip)) + width * 1. / np.cos(np.radians(dip)):
                rrup1 = np.sqrt(
                    np.square(rx - width * np.cos(np.radians(dip))) + np.square(ztor + width * np.sin(np.radians(dip))))
            rrup = np.sqrt(np.square(rrup1) + np.square(ry0))
        elif not 'rrup' in self.dist_param.keys():
            if 'rhypo' in self.dist_param.keys():
                rrup = self.dist_param['rhypo'][index]
            elif 'repi' in self.dist_param.keys():
                rrup = self.dist_param['repi'][index]
            else:
                raise ValueError('No distance parameter is defined!')

        # Closest distance to coseismic rupture (km)
        setattr(dctx, 'rrup', np.array([rrup], dtype='float64'))
        if rx:  # Horizontal distance from top of rupture measured perpendicular to fault strike (km)
            setattr(dctx, 'rx', np.array([rx], dtype='float64'))
        if ry0:  # The horizontal distance off the end of the rupture measured parallel to strike (km)
            setattr(dctx, 'ry0', np.array([ry0], dtype='float64'))
        if rjb:  # Closest distance to surface projection of coseismic rupture (km)
            setattr(dctx, 'rjb', np.array([rjb], dtype='float64'))
        # Do another loop in case some other distance parameters which I do not recall are used
        for key in self.dist_param.keys():
            temp = np.array([self.dist_param[key][index]], dtype='float64')
            setattr(dctx, key, temp)

        # ADDITIONAL SITE PARAMETERS
        # -------------------------------------
        vs30 = self.site_param['vs30']

        if 'vs30measured' in self.site_param.keys():
            vs30measured = self.site_param['vs30measured']
        else:
            vs30measured = True

        if 'z1pt0' in self.site_param.keys():
            z1pt0 = self.site_param['z1pt0']
        else:
            z1pt0 = None

        if 'z2pt5' in self.site_param.keys():
            z2pt5 = self.site_param['z2pt5']
        else:
            z2pt5 = None

        if z1pt0 is None:
            if 'ChiouYoungs' in self.gmpe:
                z1pt0 = np.exp(28.5 - 3.82 / 8 * np.log(vs30 ** 8 + 378.7 ** 8))
            else:
                if vs30 < 180:
                    z1pt0 = np.exp(6.745)
                elif 180 <= vs30 <= 500:
                    z1pt0 = np.exp(6.745 - 1.35 * np.log(vs30 / 180))
                else:
                    z1pt0 = np.exp(5.394 - 4.48 * np.log(vs30 / 500))

        if z2pt5 is None:
            z2pt5 = 519 + 3.595 * z1pt0

        # Site id
        setattr(sctx, 'sids', np.array([0], dtype='float64'))
        # Average shear-wave velocity of the site
        setattr(sctx, 'vs30', np.array([vs30], dtype='float64'))
        # vs30 type, True (measured) or False (inferred)
        setattr(sctx, 'vs30measured', np.array([vs30measured]))
        # Depth to Vs=1 km/sec
        setattr(sctx, 'z1pt0', np.array([z1pt0], dtype='float64'))
        # Depth to Vs=2.5 km/sec
        setattr(sctx, 'z2pt5', np.array([z2pt5], dtype='float64'))
        # Do another loop in case some other site parameters which I do not recall are used
        for key in self.site_param.keys():
            if isinstance(self.site_param[key], bool):
                temp = np.array([self.site_param[key]])
            else:
                temp = np.array([self.site_param[key]], dtype='float64')
            setattr(sctx, key, temp)

        return sctx, rctx, dctx

    def create(self, Tstar=0.5, gmpe='BooreEtAl2014', selection=1, Sa_def='RotD50',
               site_param={'vs30': 520}, rup_param={'rake': [0.0, 45.0], 'mag': [7.2, 6.5]},
               dist_param={'rjb': [20, 5]}, Hcont=[0.6, 0.4], T_Tgt_range=[0.01, 4],
               im_Tstar=1.0, epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram'):
        """
        Details
        -------
        Creates the target spectrum (conditional or unconditional).
    
        Notes
        -----
        See https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html
        in order to check required input parameters for the ground motion models.
        e.g. rupture parameters (rup_param), site parameters (site_param), distance parameters (dist_param).
        Rupture parameters 'fhw', 'azimuth', 'upper_sd' and 'lower_sd' are used to derive some gmm parameters
        in accordance with Kaklamanos et al. 2011 within conditional_spectrum._set_contexts method. They are not
        required by any gmm.

        References
        ----------
        Baker JW. Conditional Mean Spectrum: Tool for Ground-Motion Selection.
        Journal of Structural Engineering 2011; 137(3): 322–331.
        DOI: 10.1061/(ASCE)ST.1943-541X.0000215.

        Lin, T., Harmsen, S. C., Baker, J. W., & Luco, N. (2013). 
        Conditional Spectrum Computation Incorporating Multiple Causal Earthquakes 
        and Ground-Motion Prediction Models. In Bulletin of the Seismological Society 
        of America (Vol. 103, Issue 2A, pp. 1103–1116). Seismological Society of 
        America (SSA). https://doi.org/10.1785/0120110293

        Kohrangi, M., Bazzurro, P., Vamvatsikos, D., and Spillatura, A.
        Conditional spectrum-based ground motion record selection using average 
        spectral acceleration. Earthquake Engineering & Structural Dynamics, 
        2017, 46(10): 1667–1685.

        Parameters
        ----------
        Tstar    : int, float, numpy.array, the default is None.
            Conditioning period or periods in case of AvgSa [sec].
        gmpe     : str, optional
            GMPE model (see OpenQuake library).
            The default is 'BooreEtAl2014'.
        selection : int, optional, The default is 1.
            1 for single-component selection and arbitrary component sigma.
            2 for two-component selection and average component sigma.
        Sa_def : str, optional, the default is 'RotD50'.
            The spectra definition. Necessary if selection = 2.
            'GeoMean', 'RotD50', 'RotD100'.
        site_param : dictionary, The default is {'vs30': 520}
            Contains site parameters to define target spectrum.
            Dictionary keys (parameters) are not list type. Same parameters are used for each scenario.
            Some parameters are:
            'vs30': Average shear-wave velocity of the site
            'vs30measured': vs30 type, True (measured) or False (inferred)
            'z1pt0': Depth to Vs=1 km/sec from the site
            'z2pt5': Depth to Vs=2.5 km/sec from the site
        rup_param  : dictionary, The default is {'rake': [0.0, 45.0], 'mag': [7.2, 6.5]}
            Contains rupture parameters to define target spectrum.
            Dictionary keys (parameters) are list type. Each item in the list corresponds to a scenario.
            Some parameters are:
            'mag': Magnitude of the earthquake (required by all gmm)
            'rake': Fault rake
            'dip': Fault dip
            'width': Fault width
            'hypo_depth': Hypocentral depth of the rupture
            'ztor': Depth to top of coseismic rupture (km)
            'fhw': Hanging-wall factor, 1 for site on down-dip side of top of rupture; 0 otherwise (optional)
            'azimuth': Source-to-site azimuth, alternative of hanging wall factor (optional)
            'upper_sd': Upper seismogenic depth (optional)
            'lower_sd': Lower seismogenic depth (optional)
        dist_param : dictionary, The default is {'rjb': [20, 5]}
            Contains distance parameters to define target spectrum.
            Dictionary keys (parameters) are list type. Each item in the list corresponds to a scenario.
            Some parameters are:
            'rjb': Closest distance to surface projection of coseismic rupture (km)
            'rrup': Closest distance to coseismic rupture (km)
            'repi': Epicentral distance (km)
            'rhypo': Hypocentral distance (km)
            'rx': Horizontal distance from top of rupture measured perpendicular to fault strike (km)
            'ry0': The horizontal distance off the end of the rupture measured parallel to strike (km)
        Hcont      : list, optional, the default is None.
            Hazard contribution for considered scenarios. 
            If None hazard contribution is the same for all scenarios.
        im_Tstar   : int, float, optional, the default is 1.
            Conditioning intensity measure level [g] (conditional selection)
        epsilon    : list, optional, the default is None.
            Epsilon values for considered scenarios (conditional selection)
        T_Tgt_range: list, optional, the default is [0.01,4].
            Lower and upper bound values for the period range of target spectrum.
        cond       : int, optional
            0 to run unconditional selection
            1 to run conditional selection
        useVar     : int, optional, the default is 1.
            0 not to use variance in target spectrum
            1 to use variance in target spectrum
        corr_func: str, optional, the default is baker_jayaram
            correlation model to use "baker_jayaram","akkar"

        Returns
        -------
        None.                    
        """
        # TODO: gsim.get_mean_and_stddevs is deprecated, use ContextMaker.get_mean_stds in the future.
        # see https://docs.openquake.org/oq-engine/advanced/developing.html#working-with-gmpes-directly-the-contextmaker
        # TODO: make the step size equal in period array. This will result in more realistic matching

        if cond == 1:

            # add Tstar to self
            if isinstance(Tstar, int) or isinstance(Tstar, float):
                self.Tstar = np.array([Tstar])
            elif isinstance(Tstar, numpy.ndarray):
                self.Tstar = Tstar

            # check if AvgSa or Sa is used as IM, then in case of Sa(T*) add T* and Sa(T*) if not present
            if not self.Tstar[0] in self.database['Periods'] and len(self.Tstar) == 1:
                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_1'], axis=1)
                Sa_int = f(self.Tstar[0])
                Sa_int.shape = (len(Sa_int), 1)
                Sa = np.append(self.database['Sa_1'], Sa_int, axis=1)
                Periods = np.append(self.database['Periods'], self.Tstar[0])
                self.database['Sa_1'] = Sa[:, np.argsort(Periods)]

                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_2'], axis=1)
                Sa_int = f(self.Tstar[0])
                Sa_int.shape = (len(Sa_int), 1)
                Sa = np.append(self.database['Sa_2'], Sa_int, axis=1)
                self.database['Sa_2'] = Sa[:, np.argsort(Periods)]

                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_RotD50'], axis=1)
                Sa_int = f(self.Tstar[0])
                Sa_int.shape = (len(Sa_int), 1)
                Sa = np.append(self.database['Sa_RotD50'], Sa_int, axis=1)
                self.database['Sa_RotD50'] = Sa[:, np.argsort(Periods)]

                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_RotD100'], axis=1)
                Sa_int = f(self.Tstar[0])
                Sa_int.shape = (len(Sa_int), 1)
                Sa = np.append(self.database['Sa_RotD100'], Sa_int, axis=1)
                self.database['Sa_RotD100'] = Sa[:, np.argsort(Periods)]

                self.database['Periods'] = Periods[np.argsort(Periods)]

        try:  # this is smth like self.bgmpe = gsim.boore_2014.BooreEtAl2014()
            self.bgmpe = gsim.get_available_gsims()[gmpe]()
            self.gmpe = gmpe

        except KeyError:
            print(f'{gmpe} is not a valid gmpe name')
            raise

        # add target spectrum settings to self
        self.selection = selection
        self.Sa_def = Sa_def
        self.cond = cond
        self.useVar = useVar
        self.corr_func = corr_func
        self.site_param = site_param
        self.rup_param = rup_param
        self.dist_param = dist_param

        nScenarios = len(rup_param['mag'])  # number of scenarios
        if Hcont is None:  # equal for all
            self.Hcont = [1 / nScenarios for _ in range(nScenarios)]
        else:
            self.Hcont = Hcont

        # Period range of the target spectrum
        temp = np.abs(self.database['Periods'] - np.min(T_Tgt_range))
        idx1 = np.where(temp == np.min(temp))[0][0]
        temp = np.abs(self.database['Periods'] - np.max(T_Tgt_range))
        idx2 = np.where(temp == np.min(temp))[0][0]
        self.T = self.database['Periods'][idx1:idx2 + 1]

        # Get number of scenarios, and their contribution
        Hcont_mat = np.matlib.repmat(np.asarray(self.Hcont), len(self.T), 1)

        # Conditional spectrum, log parameters
        TgtMean = np.zeros((len(self.T), nScenarios))

        # Covariance
        TgtCov = np.zeros((nScenarios, len(self.T), len(self.T)))

        for n in range(nScenarios):

            # gmpe spectral values
            mu_lnSaT = np.zeros(len(self.T))
            sigma_lnSaT = np.zeros(len(self.T))

            # correlation coefficients
            rho_T_Tstar = np.zeros(len(self.T))

            # Covariance
            Cov = np.zeros((len(self.T), len(self.T)))

            # Set the contexts for the scenario
            sctx, rctx, dctx = self._set_contexts(n)

            # TODO: Should try to use array operations instead
            for i in range(len(self.T)):
                # Get the GMPE output for a rupture scenario
                mu, sigma = self.bgmpe.get_mean_and_stddevs(sctx, rctx, dctx, imt.SA(period=self.T[i]),
                                                            [const.StdDev.TOTAL])
                mu_lnSaT[i] = mu
                sigma_lnSaT[i] = sigma[0]
                # modify spectral targets if RotD100 values were specified for two-component selection:
                if self.Sa_def == 'RotD100' and not 'RotD100' in self.bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT and self.selection == 2:
                    rotD100Ratio, rotD100Sigma = self._gmpe_sb_2014_ratios(self.T[i])
                    mu_lnSaT[i] = mu_lnSaT[i] + np.log(rotD100Ratio)
                    sigma_lnSaT[i] = (sigma_lnSaT[i] ** 2 + rotD100Sigma ** 2) ** 0.5

            if self.cond == 1:
                # Get the GMPE output and calculate AvgSa_Tstar and associated dispersion
                mu_lnSaTstar, sigma_lnSaTstar, rho_T_Tstar = self._get_cond_param(sctx, rctx, dctx)

                if epsilon is None:
                    # Back calculate epsilon
                    rup_eps = (np.log(im_Tstar) - mu_lnSaTstar) / sigma_lnSaTstar
                else:
                    rup_eps = epsilon[n]

                # Get the value of the ln(CMS), conditioned on T_star
                TgtMean[:, n] = mu_lnSaT + rho_T_Tstar * rup_eps * sigma_lnSaT

            elif self.cond == 0:
                TgtMean[:, n] = mu_lnSaT

            for i in range(len(self.T)):
                for j in range(len(self.T)):

                    var1 = sigma_lnSaT[i] ** 2
                    var2 = sigma_lnSaT[j] ** 2

                    rho = self._get_correlation(self.T[i], self.T[j])
                    sigma_Corr = rho * np.sqrt(var1 * var2)

                    if self.cond == 1:
                        varTstar = sigma_lnSaTstar ** 2
                        sigma11 = np.matrix([[var1, sigma_Corr], [sigma_Corr, var2]])
                        sigma22 = np.array([varTstar])
                        sigma12 = np.array([rho_T_Tstar[i] * np.sqrt(var1 * varTstar),
                                            rho_T_Tstar[j] * np.sqrt(varTstar * var2)])
                        sigma12.shape = (2, 1)
                        sigma22.shape = (1, 1)
                        sigma_cond = sigma11 - sigma12 * 1. / sigma22 * sigma12.T
                        Cov[i, j] = sigma_cond[0, 1]

                    elif self.cond == 0:
                        Cov[i, j] = sigma_Corr

            # Get the value of standard deviation of target spectrum
            TgtCov[n, :, :] = Cov

        # over-write covariance matrix with zeros if no variance is desired in the ground motion selection
        if self.useVar == 0:
            TgtCov = np.zeros(TgtCov.shape)

        TgtMean_fin = np.sum(TgtMean * Hcont_mat, 1)
        # all 2D matrices are the same for each kk scenario, since sigma is only T dependent
        TgtCov_fin = TgtCov[0, :, :]
        Cov_elms = np.zeros((len(self.T), nScenarios))
        for ii in range(len(self.T)):
            for kk in range(nScenarios):
                # Hcont[kk] is hazard contribution of the k-th scenario
                Cov_elms[ii, kk] = (TgtCov[kk, ii, ii] + (TgtMean[ii, kk] - TgtMean_fin[ii]) ** 2) * self.Hcont[kk]

        # Compute the final covariance matrix
        cov_diag = np.sum(Cov_elms, 1)
        TgtCov_fin[np.eye(len(self.T)) == 1] = cov_diag
        TgtSigma_fin = np.sqrt(np.diagonal(TgtCov_fin))

        # Add target spectrum to self
        self.mu_ln = TgtMean_fin
        self.sigma_ln = TgtSigma_fin
        self.cov = TgtCov_fin

        if cond == 1:
            # add intensity measure level to self
            if epsilon is None:
                self.im_Tstar = im_Tstar
            else:
                f = interpolate.interp1d(self.T, np.exp(self.mu_ln))
                Sa_int = f(self.Tstar)
                self.im_Tstar = np.exp(np.sum(np.log(Sa_int)) / len(self.Tstar))
                self.epsilon = epsilon

        print('Target spectrum is created.')

    def _simulate_spectra(self):
        """
        Details
        -------
        Generates simulated response spectra with best matches to the target values.

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """

        # Set initial seed for simulation
        if self.seedValue:
            np.random.seed(self.seedValue)
        else:
            np.random.seed(sum(gmtime()[:6]))

        # Avoid positive semi-definite covariance matrix with several eigenvalues being exactly zero.
        # See: https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
        # Covariance matrix is important and going to be used to perform the initial record selection!
        cov = self.cov.copy()
        min_eig = np.min(np.real(np.linalg.eigvals(cov)))
        if min_eig < 0:
            cov -= min_eig * np.eye(*self.cov.shape)

        devTotalSim = np.zeros((self.nTrials, 1))
        specDict = {}
        # Generate simulated response spectra with best matches to the target values
        for j in range(self.nTrials):
            specDict[j] = np.exp(random_multivariate_normal(self.mu_ln, cov, self.nGM, 'LHS'))
            # specDict[j] = np.exp(np.random.multivariate_normal(self.mu_ln, self.cov, size=self.nGM))

            # how close is the mean of the spectra to the target
            devMeanSim = np.mean(np.log(specDict[j]), axis=0) - self.mu_ln 
            # how close is the mean of the spectra to the target
            devSigSim = np.std(np.log(specDict[j]), axis=0) - self.sigma_ln
            # how close is the skewness of the spectra to zero (i.e., the target)  
            devSkewSim = skew(np.log(specDict[j]), axis=0)  
            # combine the three error metrics to compute a total error
            devTotalSim[j] = self.weights[0] * np.sum(devMeanSim ** 2) + \
                             self.weights[1] * np.sum(devSigSim ** 2) + \
                             0.1 * (self.weights[2]) * np.sum(devSkewSim ** 2)

        recUse = np.argmin(np.abs(devTotalSim))  # find the simulated spectra that best match the targets
        self.sim_spec = np.log(specDict[recUse])  # return the best set of simulations

    def select(self, nGM=30, isScaled=1, maxScale=4,
               Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None,
               nTrials=20, seedValue=None, weights=[1, 2, 0.3], nLoop=2, penalty=0, tol=10):
        """
        Details
        -------
        Perform the ground motion selection.
        
        References
        ----------
        Jayaram, N., Lin, T., and Baker, J. W. (2011). 
        A computationally efficient ground-motion selection algorithm for 
        matching a target response spectrum mean and variance.
        Earthquake Spectra, 27(3), 797-815.
        
        Parameters
        ----------
        nGM : int, optional, the default is 30.
            Number of ground motions to be selected.
        isScaled : int, optional, the default is 1.
            0 not to allow use of amplitude scaling for spectral matching.
            1 to allow use of amplitude scaling for spectral matching.
        maxScale : float, optional, the default is 4.
            The maximum allowable scale factor
        Mw_lim : list, optional, the default is None.
            The limiting values on magnitude. 
        Vs30_lim : list, optional, the default is None.
            The limiting values on Vs30. 
        Rjb_lim : list, optional, the default is None.
            The limiting values on Rjb. 
        fault_lim : list, optional, the default is None.
            The limiting fault mechanisms.
            For NGA_W2 database:
                0 for unspecified fault
                1 for strike-slip fault
                2 for normal fault
                3 for reverse fault
            For ESM_2018 database:
                'NF' for normal faulting
                'NS' for predominately normal with strike-slip component
                'O' for oblique
                'SS' for strike-slip faulting
                'TF' for thrust faulting
                'TS' for predominately thrust with strike-slip component
                'U' for unknown
        nTrials : int, optional, the default is 20.
            nTrials sets of response spectra are simulated and the best set (in terms of
            matching means, variances and skewness is chosen as the seed). The user
            can also optionally rerun this segment multiple times before deciding to
            proceed with the rest of the algorithm. It is to be noted, however, that
            the greedy improvement technique significantly improves the match between
            the means and the variances subsequently.
        seedValue  : int, optional, the default is None.
            For repeatability. For a particular seedValue not equal to
            zero, the code will output the same set of ground motions.
            The set will change when the seedValue changes. If set to
            zero, the code randomizes the algorithm and different sets of
            ground motions (satisfying the target mean and variance) are
            generated each time.
        weights : numpy.array or list, optional, the default is [1,2,0.3].
            Weights for error in mean, standard deviation and skewness
        nLoop   : int, optional, the default is 2.
            Number of loops of optimization to perform.
        penalty : int, optional, the default is 0.
            > 0 to penalize selected spectra more than 
            3 sigma from the target at any period, = 0 otherwise.
        tol     : int, optional, the default is 10.
            Tolerable percent error to skip optimization 

        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.nGM = nGM
        self.isScaled = isScaled
        self.Mw_lim = Mw_lim
        self.Vs30_lim = Vs30_lim
        self.Rjb_lim = Rjb_lim
        self.fault_lim = fault_lim
        self.seedValue = seedValue
        self.weights = weights
        self.nTrials = nTrials
        self.maxScale = maxScale
        self.nLoop = nLoop
        self.tol = tol
        self.penalty = penalty

        # Simulate response spectra
        self._simulate_spectra()

        # Search the database and filter
        sampleBig, Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num, eq_ID, station_code = self._search_database()

        # Processing available spectra
        sampleBig = np.log(sampleBig)
        nBig = sampleBig.shape[0]

        # Find best matches to the simulated spectra from ground-motion database
        recID = np.ones(self.nGM, dtype=int) * (-1)
        finalScaleFac = np.ones(self.nGM)
        sampleSmall = np.ones((self.nGM, sampleBig.shape[1]))
        weights = np.array(weights)

        if self.cond == 1 and self.isScaled == 1:
            # Calculate IMLs for the sample
            f = interpolate.interp1d(self.T, np.exp(sampleBig), axis=1)
            sampleBig_imls = np.exp(np.sum(np.log(f(self.Tstar)), axis=1) / len(self.Tstar))

        if self.cond == 1 and len(self.Tstar) == 1:
            # These indices are required in case IM = Sa(T) to break the loop
            ind2 = (np.where(self.T != self.Tstar[0])[0][0]).tolist()

        # Find nGM ground motions, initial subset
        for i in range(self.nGM):
            err = np.zeros(nBig)
            scaleFac = np.ones(nBig)

            # Calculate the scaling factor
            if self.isScaled == 1:
                # using conditioning IML
                if self.cond == 1:
                    scaleFac = self.im_Tstar / sampleBig_imls
                # using error minimization
                elif self.cond == 0:
                    scaleFac = np.sum(np.exp(sampleBig) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(
                        np.exp(sampleBig) ** 2, axis=1)
            else:
                scaleFac = np.ones(nBig)

            mask = (1 / self.maxScale < scaleFac) * (scaleFac < self.maxScale)
            idxs = np.where(mask)[0]
            err[~mask] = 1000000
            err[mask] = np.sum((np.log(
                np.exp(sampleBig[idxs, :]) * scaleFac[mask].reshape(len(scaleFac[mask]), 1)) -
                                self.sim_spec[i, :]) ** 2, axis=1)

            recID[i] = int(np.argsort(err)[0])
            if err.min() >= 1000000:
                raise Warning('Possible problem with simulated spectrum. No good matches found')

            if self.isScaled == 1:
                finalScaleFac[i] = scaleFac[recID[i]]

            # Save the selected spectra
            sampleSmall[i, :] = np.log(np.exp(sampleBig[recID[i], :]) * finalScaleFac[i])

        # Apply Greedy subset modification procedure
        # Use njit to speed up the optimization algorithm
        @njit
        def find_rec(sampleSmall, scaleFac, mu_ln, sigma_ln, recIDs):

            def mean_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].mean())

                return np.array(res)

            def std_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].std())

                return np.array(res)

            minDev = 100000
            for j in range(nBig):
                # Add to the sample the scaled spectra
                temp = np.zeros((1, len(sampleBig[j, :])))
                temp[:, :] = sampleBig[j, :]
                tempSample = np.concatenate((sampleSmall, temp + np.log(scaleFac[j])), axis=0)
                devMean = mean_numba(tempSample) - mu_ln  # Compute deviations from target
                devSig = std_numba(tempSample) - sigma_ln
                devTotal = weights[0] * np.sum(devMean * devMean) + weights[1] * np.sum(devSig * devSig)

                # Check if we exceed the scaling limit
                if scaleFac[j] > maxScale or scaleFac[j] < 1 / maxScale or np.any(recIDs == j):
                    devTotal = devTotal + 1000000
                # Penalize bad spectra
                elif penalty > 0:
                    for m in range(nGM):
                        devTotal = devTotal + np.sum(
                            np.abs(np.exp(tempSample[m, :]) > np.exp(mu_ln + 3.0 * sigma_ln))) * penalty
                        devTotal = devTotal + np.sum(
                            np.abs(np.exp(tempSample[m, :]) < np.exp(mu_ln - 3.0 * sigma_ln))) * penalty

                # Should cause improvement and record should not be repeated
                if devTotal < minDev:
                    minID = j
                    minDev = devTotal

            return minID

        for k in range(self.nLoop):  # Number of passes

            for i in range(self.nGM):  # Loop for nGM
                sampleSmall = np.delete(sampleSmall, i, 0)
                recID = np.delete(recID, i)

                # Calculate the scaling factor
                if self.isScaled == 1:
                    # using conditioning IML
                    if self.cond == 1:
                        scaleFac = self.im_Tstar / sampleBig_imls
                    # using error minimization
                    elif self.cond == 0:
                        scaleFac = np.sum(np.exp(sampleBig) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(
                            np.exp(sampleBig) ** 2, axis=1)
                else:
                    scaleFac = np.ones(nBig)

                # Try to add a new spectra to the subset list
                minID = find_rec(sampleSmall, scaleFac, self.mu_ln, self.sigma_ln, recID)

                # Add new element in the right slot
                if self.isScaled == 1:
                    finalScaleFac[i] = scaleFac[minID]
                else:
                    finalScaleFac[i] = 1
                sampleSmall = np.concatenate(
                    (sampleSmall[:i, :], sampleBig[minID, :].reshape(1, sampleBig.shape[1]) + np.log(scaleFac[minID]),
                     sampleSmall[i:, :]), axis=0)
                recID = np.concatenate((recID[:i], np.array([minID]), recID[i:]))

            # Lets check if the selected ground motions are good enough, if the errors are sufficiently small stop!
            if self.cond == 1 and len(self.Tstar) == 1:  # if conditioned on SaT, ignore error at T*
                medianErr = np.max(
                    np.abs(np.exp(np.mean(sampleSmall[:, ind2], axis=0)) - np.exp(self.mu_ln[ind2])) / np.exp(
                        self.mu_ln[ind2])) * 100
                stdErr = np.max(
                    np.abs(np.std(sampleSmall[:, ind2], axis=0) - self.sigma_ln[ind2]) / self.sigma_ln[ind2]) * 100
            else:
                medianErr = np.max(
                    np.abs(np.exp(np.mean(sampleSmall, axis=0)) - np.exp(self.mu_ln)) / np.exp(self.mu_ln)) * 100
                stdErr = np.max(np.abs(np.std(sampleSmall, axis=0) - self.sigma_ln) / self.sigma_ln) * 100

            if medianErr < self.tol and stdErr < self.tol:
                break
        print('Ground motion selection is finished.')
        print(f'For T ∈ [{self.T[0]:.2f} - {self.T[-1]:.2f}]')
        print(f'Max error in median = {medianErr:.2f} %')
        print(f'Max error in standard deviation = {stdErr:.2f} %')
        if medianErr < self.tol and stdErr < self.tol:
            print(f'The errors are within the target {self.tol:d} percent %')

        recID = recID.tolist()
        # Add selected record information to self
        self.rec_scale = finalScaleFac
        self.rec_spec = sampleSmall
        self.rec_Vs30 = Vs30[recID]
        self.rec_Rjb = Rjb[recID]
        self.rec_Mw = Mw[recID]
        self.rec_fault = fault[recID]
        self.rec_eqID = eq_ID[recID]
        self.rec_h1 = Filename_1[recID]

        if self.selection == 2:
            self.rec_h2 = Filename_2[recID]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recID]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[recID]


class code_spectrum(_subclass_):
    """
    This class is used for
        1) Creating target spectrum based on various codes (TBEC 2018, ASCE 7-16, EC8-Part1)
        2) Selecting and scaling suitable ground motion sets for target spectrum in accordance with specified code
    """

    def __init__(self, database='NGA_W2', outdir='Outputs', target_path=None, nGM=11, selection=1,
                 Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, opt=1, maxScale=2, RecPerEvent=3):
        """
        Details
        -------
        Loads the record database to use, creates output folder, sets selection criteria.

        Parameters
        ----------
        database : str, optional
            Database to use: NGA_W2, ESM_2018
            The default is NGA_W2.
        outdir : str, optional
            Output directory
            The default is 'Outputs'
        target_path = str, optional, the default is None.
            Path for used defined target spectrum.
        nGM : int, optional, the default is 11.
            Number of records to be selected. 
        selection : int, optional, the default is 1.
            Number of ground motion components to select. 
        Mw_lim : list, optional, the default is None.
            The limiting values on magnitude. 
        Vs30_lim : list, optional, the default is None.
            The limiting values on Vs30. 
        Rjb_lim : list, optional, the default is None.
            The limiting values on Rjb. 
        fault_lim : list, optional, the default is None.
            The limiting fault mechanisms.
            For NGA_W2 database:
                0 for unspecified fault
                1 for strike-slip fault
                2 for normal fault
                3 for reverse fault
            For ESM_2018 database:
                'NF' for normal faulting
                'NS' for predominately normal with strike-slip component
                'O' for oblique
                'SS' for strike-slip faulting
                'TF' for thrust faulting
                'TS' for predominately thrust with strike-slip component
                'U' for unknown
        opt : int, optional, the default is 1.
            If equal to 1, the record set is selected using
            method of “least squares”, each record has individual scaling factor.
            If equal to 2, the record set selected such that each record has
            identical scale factor which is as close as possible to 1.
        maxScale : float, optional, the default is 2.
            Maximum allowed scaling factor, used with opt=2 case.
        RecPerEvent: int, the default is 3.
            The limit for the maximum number of records belong to the same event

        Returns
        -------
        None.
        """

        # Add the input the ground motion database to use
        super().__init__()
        matfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Meta_Data', database)
        self.database = loadmat(matfile, squeeze_me=True)

        # create the output directory and add the path to self
        cwd = os.getcwd()
        outdir_path = os.path.join(cwd, outdir)
        create_dir(outdir_path)

        # Add selection settings to self
        self.database['Name'] = database
        self.outdir = outdir_path
        self.nGM = nGM
        self.selection = selection
        self.Mw_lim = Mw_lim
        self.Vs30_lim = Vs30_lim
        self.Rjb_lim = Rjb_lim
        self.fault_lim = fault_lim
        self.opt = opt
        self.maxScale = maxScale
        self.target_path = target_path
        self.RecPerEvent = RecPerEvent

    @staticmethod
    @njit
    def _opt2(sampleSmall, scaleFac, target_spec, recIDs, eqIDs, minID, nBig, eq_ID, sampleBig, RecPerEvent):
        # Optimize based on scaling factor
        def mean_numba(a):

            res = []
            for i in range(a.shape[1]):
                res.append(a[:, i].mean())

            return np.array(res)

        for j in range(nBig):
            tmp = eq_ID[j]
            # record should not be repeated and number of eqs from the same event should not exceed 3
            if not np.any(recIDs == j) and np.sum(eqIDs == tmp) < RecPerEvent:
                # Add to the sample the scaled spectra
                temp = np.zeros((1, len(sampleBig[j, :])))
                temp[:, :] = sampleBig[j, :]  # get the trial spectra
                tempSample = np.concatenate((sampleSmall, temp), axis=0)  # add the trial spectra to subset list
                tempScale = np.max(target_spec / mean_numba(tempSample))  # compute new scaling factor

                # Should cause improvement
                if abs(tempScale - 1) <= abs(scaleFac - 1):
                    minID = j
                    scaleFac = tempScale

        return minID, scaleFac

    def tbec2018(self, Lat=41.0582, Long=29.00951, DD=2, SiteClass='ZC', Tp=1):
        """
        Details
        -------
        Selects the suitable ground motion set in accordance with TBEC 2018. 
        If user did not define any target spectrum, the design spectrum defined by the code is going to be used. 
        The latter requires the definition of site parameters


        References
        ----------
        TBEC. (2018). Turkish building earthquake code.

        Notes
        -----
        Rule 1: Mean of selected records should remain above the lower bound target spectra.
            For selection = 1: Sa_rec = (Sa_1 or Sa_2) - lower bound = 1.0 * SaTarget(0.2Tp-1.5Tp) 
            For Selection = 2: Sa_rec = (Sa_1**2+Sa_2**2)**0.5 - lower bound = 1.3 * SaTarget(0.2Tp-1.5Tp) 

        Rule 2: 
            No more than 3 records can be selected from the same event! In other words,
            rec_eqID cannot be the same for more than 3 of the selected records.      

        Rule 3: 
            At least 11 records (or pairs) must be selected.

        Parameters
        ----------
        Lat: float, optional, the default is 41.0582.
            Site latitude
        Long: float, optional, the default is 29.00951.
            Site longitude
        DD:  int, optional, the default is 2.
            Earthquake ground motion intensity level (1,2,3,4)
        SiteClass: str, optional, the default is 'ZC'.
            Site soil class ('ZA','ZB','ZC','ZD','ZE')
        Tp : float, optional, the default is 1.
            Predominant period of the structure. 
        
        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.Lat = Lat
        self.Long = Long
        self.DD = DD
        self.SiteClass = SiteClass
        self.Tp = Tp
        self.code = 'TBEC 2018'

        if self.nGM < 11:
            print('Warning! nGM must be at least 11 according to TBEC 2018. Changing...')
            self.nGM = 11

        if self.RecPerEvent > 3:
            print('Warning! Limit for Record Per Event must be at most 3 according to TBEC 2018. Changing...')
            self.RecPerEvent = 3

        # Set the period range
        self.Tlower = 0.2 * Tp
        self.Tupper = 1.5 * Tp

        # Match periods (periods for error computations)
        self.T = self.database['Periods']

        # Determine the elastic design spectrum from the user-defined spectrum
        if self.target_path:
            data = np.loadtxt(self.target_path)
            intfunc = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            target_spec = intfunc(self.T)

        # Determine the elastic design spectrum from code
        else:
            PGA, SDS, SD1, TL = SiteParam_tbec2018(Lat, Long, DD, SiteClass)
            target_spec = Sae_tbec2018(self.T, PGA, SDS, SD1, TL)

        # Consider the lower bound spectrum specified by the code as target spectrum
        if self.selection == 1:
            target_spec = 1.0 * target_spec
        elif self.selection == 2:
            target_spec = 1.3 * target_spec
            self.Sa_def = 'SRSS'

        # Search the database and filter
        sampleBig, Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num, eq_ID_, station_code = self._search_database()

        # Sample size of the filtered database
        nBig = sampleBig.shape[0]

        # Scale factors based on mse
        scaleFac = np.array(
            np.sum(np.matlib.repmat(target_spec, nBig, 1) * sampleBig, axis=1) / np.sum(sampleBig ** 2, axis=1))

        # Find best matches to the target spectrum from ground-motion database
        temp = (np.matlib.repmat(target_spec, nBig, 1) - sampleBig) ** 2
        mse = temp.mean(axis=1)

        if self.database['Name'].startswith('ESM'):
            d = {ni: indi for indi, ni in enumerate(set(eq_ID_.tolist()))}
            eq_ID = np.asarray([d[ni] for ni in eq_ID_.tolist()])
        else:
            eq_ID = eq_ID_.copy()

        recID_sorted = np.argsort(mse)
        recIDs = np.ones(self.nGM, dtype=int) * (-1)
        eqIDs = np.ones(self.nGM, dtype=int) * (-1)
        idx1 = 0
        idx2 = 0
        while idx1 < self.nGM:  # not more than 3 of the records should be from the same event
            tmp1 = recID_sorted[idx2]
            idx2 += 1
            tmp2 = eq_ID[tmp1]
            recIDs[idx1] = tmp1
            eqIDs[idx1] = tmp2
            if np.sum(eqIDs == tmp2) <= self.RecPerEvent:
                idx1 += 1

        # Initial selection results - based on MSE
        finalScaleFac = scaleFac[recIDs]
        sampleSmall = sampleBig[recIDs, :]

        # Must not be lower than target within the period range, find the indicies for this period range
        idxs = np.where((self.database['Periods'] >= self.Tlower) * (self.database['Periods'] <= self.Tupper))[0]

        if self.opt == 1:
            self.rec_scale = finalScaleFac * np.max(
                target_spec[idxs] / (finalScaleFac.reshape(-1, 1) * sampleSmall[:, idxs]).mean(axis=0))

        # try to optimize scaling factor to make it closest as possible to 1
        if self.opt == 2:
            finalScaleFac = np.max(target_spec[idxs] / sampleSmall[:, idxs].mean(axis=0))
            for i in range(self.nGM):  # Loop for nGM
                # note the ID of the record which is removed
                minID = recIDs[i]
                # remove the i'th record search for a candidate, and consider critical periods for error calculations only
                sampleSmall_reduced = np.delete(sampleSmall[:, idxs], i, 0)
                recIDs = np.delete(recIDs, i)
                eqIDs = np.delete(eqIDs, i)
                # Try to add a new spectra to the subset list
                minID, finalScaleFac = self._opt2(sampleSmall_reduced, finalScaleFac, target_spec[idxs], recIDs, eqIDs,
                                                  minID, nBig, eq_ID, sampleBig[:, idxs], self.RecPerEvent)
                # Add new element in the right slot
                sampleSmall = np.concatenate(
                    (sampleSmall[:i, :], sampleBig[minID, :].reshape(1, sampleBig.shape[1]), sampleSmall[i:, :]),
                    axis=0)
                recIDs = np.concatenate((recIDs[:i], np.array([minID]), recIDs[i:]))
                eqIDs = np.concatenate((eqIDs[:i], np.array([eq_ID[minID]]), eqIDs[i:]))
            self.rec_scale = np.ones(self.nGM) * float(finalScaleFac)

        # check the scaling
        if np.any(self.rec_scale > self.maxScale) or np.any(self.rec_scale < 1 / self.maxScale):
            raise ValueError('Scaling factor criteria is not satisfied',
                             'Please broaden your selection and scaling criteria or change the optimization scheme...')

        recIDs = recIDs.tolist()
        # Add selected record information to self
        self.rec_Vs30 = Vs30[recIDs]
        self.rec_Rjb = Rjb[recIDs]
        self.rec_Mw = Mw[recIDs]
        self.rec_fault = fault[recIDs]
        self.rec_eqID = eq_ID_[recIDs]
        self.rec_h1 = Filename_1[recIDs]

        if self.selection == 2:
            self.rec_h2 = Filename_2[recIDs]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recIDs]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[recIDs]

        rec_idxs = []
        if self.selection == 1:
            SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_h1:
                rec_idxs.append(np.where(Filename_1 == rec)[0][0])
            rec_spec = SaKnown[rec_idxs, :]
        elif self.selection == 2:
            for rec in self.rec_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            Sa_1 = self.database['Sa_1'][rec_idxs, :]
            Sa_2 = self.database['Sa_2'][rec_idxs, :]
            rec_spec = (Sa_1 ** 2 + Sa_2 ** 2) ** 0.5

        # Save the results for whole spectral range
        self.rec_spec = rec_spec
        self.T = self.database['Periods']

        if self.target_path:
            self.target = intfunc(self.T)
        else:
            self.target = Sae_tbec2018(self.T, PGA, SDS, SD1, TL)

        print('TBEC 2018 based ground motion record selection and amplitude scaling are finished...')

    def asce7_16(self, Lat=34, Long=-118, RiskCat='II', SiteClass='C', T1_small=1, T1_big=1, Tlower=None, Tupper=None):
        """
        Details
        -------
        Selects the suitable ground motion set in accordance with ASCE 7-16. 
        If user did not define any target spectrum, the MCE_R response spectrum defined by the code is going to be used. 
        The latter requires the definition of site parameters.


        References
        ----------
        American Society of Civil Engineers. (2017, June). Minimum design loads and associated criteria 
        for buildings and other structures. American Society of Civil Engineers.

        Notes
        -----
        Rule 1: Mean of selected records should remain above the lower bound target spectra.
            For selection = 1: Sa_rec = (Sa_1 or Sa_2) - lower bound = 0.9 * Sa_MCEr(Tlower-Tupper) 
            For Selection = 2: Sa_rec = RotD100 - lower bound = 0.9 * Sa_MCEr(Tlower-Tupper)     
            Tlower >= 0.2 * T1_small
            Tupper >= 1.5 * T1_big
            
        Rule 2: 
            At least 11 records (or pairs) must be selected.

        Parameters
        ----------
        Lat: float, optional, the default is 41.0582.
            Site latitude
        Long: float, optional, the default is 29.00951.
            Site longitude
        RiskCat:  str, the default is 'III'
            Risk category for structure ('I','II','III','IV')
        SiteClass: str, optional, the default is 'C'.
            Site soil class ('A','B','C','D','E')
        T1_small: float, the default is 1.
            The smallest of first-mode periods in principal horizontal directions
        T1_big: float, the default is 1.
            The largest of first-mode periods in principal horizontal directions        
        Tlower: float, the default is None.
            The lower bound for period range, if None equal to 0.2*T1_small
        Tupper: float, the default is None.
            The upper bound for period range, if None equal to 2.0*T1_big
        
        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.Lat = Lat
        self.Long = Long
        self.RiskCat = RiskCat
        self.SiteClass = SiteClass
        self.code = 'ASCE 7-16'

        # Section 16.2.3.1
        if not Tlower:
            Tlower = 0.2 * T1_small
        elif Tlower < 0.2 * T1_small:
            Tlower = 0.2 * T1_small
            print('Warning! Lower bound cannot be lower than 0.2 times the largest first-mode period according to '
                  'ASCE 7-16. Changing...')
        if not Tupper:
            Tupper = 2.0 * T1_big
        elif Tupper < 1.5 * T1_big:
            Tupper = 1.5 * T1_big
            print('Warning! Upper bound cannot be lower than 1.5 times the smallest first-mode period according to '
                  'ASCE 7-16. Changing...')
        self.Tlower = Tlower
        self.Tupper = Tupper

        # Section 16.2.2
        if self.nGM < 11:
            print('Warning! nGM must be at least 11 according to ASCE 7-16. Changing...')
            self.nGM = 11

        # Match periods (periods for error computations)
        self.T = self.database['Periods']

        # Determine the elastic design spectrum from the user-defined spectrum
        if self.target_path:
            data = np.loadtxt(self.target_path)
            intfunc = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            target_spec = intfunc(self.T)

        # Determine the elastic design spectrum from code, Section 16.2.1
        else:
            SDS, SD1, TL = SiteParam_asce7_16(Lat, Long, RiskCat, SiteClass)  # Retrieve site parameters
            target_spec = 1.5 * Sae_asce7_16(self.T, SDS, SD1,
                                             TL)  # Retrive the design spectrum and multiply by 1.5 to get MCER

        # Consider the lower bound spectrum specified by the code as target spectrum, Section 16.2.3.2
        target_spec = 0.9 * target_spec
        if self.selection == 2:
            self.Sa_def = 'RotD100'

            # Search the database and filter
        sampleBig, Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num, eq_ID_, station_code = self._search_database()

        # Sample size of the filtered database
        nBig = sampleBig.shape[0]

        # Scale factors based on mse
        scaleFac = np.array(
            np.sum(np.matlib.repmat(target_spec, nBig, 1) * sampleBig, axis=1) / np.sum(sampleBig ** 2, axis=1))

        # Find best matches to the target spectrum from ground-motion database
        temp = (np.matlib.repmat(target_spec, nBig, 1) - sampleBig) ** 2
        mse = temp.mean(axis=1)

        if self.database['Name'].startswith('ESM'):
            d = {ni: indi for indi, ni in enumerate(set(eq_ID_.tolist()))}
            eq_ID = np.asarray([d[ni] for ni in eq_ID_.tolist()])
        else:
            eq_ID = eq_ID_.copy()

        recID_sorted = np.argsort(mse)
        recIDs = np.ones(self.nGM, dtype=int) * (-1)
        eqIDs = np.ones(self.nGM, dtype=int) * (-1)
        idx1 = 0
        idx2 = 0
        while idx1 < self.nGM:  # not more than 3 of the records should be from the same event
            tmp1 = recID_sorted[idx2]
            idx2 += 1
            tmp2 = eq_ID[tmp1]
            recIDs[idx1] = tmp1
            eqIDs[idx1] = tmp2
            if np.sum(eqIDs == tmp2) <= self.RecPerEvent:
                idx1 += 1

        # Initial selection results - based on MSE
        finalScaleFac = scaleFac[recIDs]
        sampleSmall = sampleBig[recIDs, :]

        # Must not be lower than target within the period range, find the indicies for this period range
        idxs = np.where((self.database['Periods'] >= self.Tlower) * (self.database['Periods'] <= self.Tupper))[0]

        if self.opt == 1:
            self.rec_scale = finalScaleFac * np.max(
                target_spec[idxs] / (finalScaleFac.reshape(-1, 1) * sampleSmall[:, idxs]).mean(axis=0))

        # try to optimize scaling factor to make it closest as possible to 1
        if self.opt == 2:
            finalScaleFac = np.max(target_spec[idxs] / sampleSmall[:, idxs].mean(axis=0))
            for i in range(self.nGM):  # Loop for nGM
                # note the ID of the record which is removed
                minID = recIDs[i]
                # remove the i'th record search for a candidate, and consider critical periods for error calculations only
                sampleSmall_reduced = np.delete(sampleSmall[:, idxs], i, 0)
                recIDs = np.delete(recIDs, i)
                eqIDs = np.delete(eqIDs, i)
                # Try to add a new spectra to the subset list
                minID, finalScaleFac = self._opt2(sampleSmall_reduced, finalScaleFac, target_spec[idxs], recIDs, eqIDs,
                                                  minID, nBig, eq_ID, sampleBig[:, idxs], self.RecPerEvent)
                # Add new element in the right slot
                sampleSmall = np.concatenate(
                    (sampleSmall[:i, :], sampleBig[minID, :].reshape(1, sampleBig.shape[1]), sampleSmall[i:, :]),
                    axis=0)
                recIDs = np.concatenate((recIDs[:i], np.array([minID]), recIDs[i:]))
                eqIDs = np.concatenate((eqIDs[:i], np.array([eq_ID[minID]]), eqIDs[i:]))
            self.rec_scale = np.ones(self.nGM) * float(finalScaleFac)

        # check the scaling
        if np.any(self.rec_scale > self.maxScale) or np.any(self.rec_scale < 1 / self.maxScale):
            raise ValueError('Scaling factor criteria is not satisfied',
                             'Please broaden your selection and scaling criteria or change the optimization scheme...')
        recIDs = recIDs.tolist()
        # Add selected record information to self
        self.rec_Vs30 = Vs30[recIDs]
        self.rec_Rjb = Rjb[recIDs]
        self.rec_Mw = Mw[recIDs]
        self.rec_fault = fault[recIDs]
        self.rec_eqID = eq_ID_[recIDs]
        self.rec_h1 = Filename_1[recIDs]

        if self.selection == 2:
            self.rec_h2 = Filename_2[recIDs]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recIDs]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[recIDs]

        rec_idxs = []
        if self.selection == 1:
            SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_h1:
                rec_idxs.append(np.where(Filename_1 == rec)[0][0])
            rec_spec = SaKnown[rec_idxs, :]
        elif self.selection == 2:
            for rec in self.rec_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            rec_spec = self.database['Sa_RotD100'][rec_idxs, :]

        # Save the results for whole spectral range
        self.rec_spec = rec_spec
        self.T = self.database['Periods']

        if self.target_path:
            self.target = intfunc(self.T)
        else:
            self.target = 1.5 * Sae_asce7_16(self.T, SDS, SD1, TL)

        print('ASCE 7-16 based ground motion record selection and amplitude scaling are finished...')

    def ec8_part1(self, ag=0.2, xi=0.05, ImpClass='II', Type='Type1', SiteClass='C', Tp=1):
        """
        Details
        -------
        Select the suitable ground motion set in accordance with EC8 - PART 1.
        If user did not define any target spectrum, the design spectrum defined by the code is going to be used. 
        The latter requires the definition of site parameters

        References
        ----------
        CEN. Eurocode 8: Design of Structures for Earthquake Resistance -  Part 1: General Rules, Seismic Actions and Rules 
        for Buildings (EN 1998-1:2004). Brussels, Belgium: 2004.

        Notes
        -----
        Section 3.2.3.1.
        
        Rule 1 (a): 
            At least 3 records (or pairs) must be selected.

        Rule 2 (b): 
            mean(PGA_rec) >= PGA_target
            Here we assume SA(T[0])=PGA, where T[0] is 0.01 for both record databases.
            Not a bad assumption since it is very close to PGA.

        Rule 3 (c): Mean of selected records should remain above the lower bound target spectrum.
            For selection = 1: Sa_rec = (Sa_1 or Sa_2) - lower bound = 0.9 * SaTarget(0.2Tp-2.0Tp)
            For Selection = 2: Sa_rec = (Sa_1+Sa_2)*0.5 - lower bound = 0.9 * SaTarget(0.2Tp-2.0Tp)

        Parameters
        ----------
        ag:  float, optional, the default is 0.25.
            Peak ground acceleration [g]
        xi: float, optional, the default is 0.05.
            Damping
        ImpClass: str, the default is 'II'.
            Importance class ('I','II','III','IV')
        Type: str, optional, the default is 'Type1'
            Type of spectrum (Option: 'Type1' or 'Type2')
        SiteClass: str, optional, the default is 'B'
            Soil Class (Options: 'A', 'B', 'C', 'D' or 'E')
        Tp : float, optional, the default is 1.
            Predominant period of the structure. 
        
        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.Tp = Tp
        self.ag = ag
        self.ImpClass = ImpClass
        self.Type = Type
        self.SiteClass = SiteClass
        self.code = 'EC8-Part1'

        # Set the period range
        self.Tlower = 0.2 * Tp
        self.Tupper = 2.0 * Tp

        # Match periods (periods for error computations)
        self.T = self.database['Periods']

        # Determine the elastic design spectrum from the user-defined spectrum
        if self.target_path:
            data = np.loadtxt(self.target_path)
            func = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            target_spec = func(self.T)

        # Determine the elastic design spectrum from code
        else:
            target_spec = Sae_ec8_part1(ag, xi, self.T, ImpClass, Type, SiteClass)

        # Consider the lower bound spectrum specified by the code as target spectrum
        target_spec = 0.9 * target_spec  # scale down except for Sa(T[0]) or PGA
        if self.selection == 2:
            self.Sa_def = 'ArithmeticMean'

        # Search the database and filter
        sampleBig, Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num, eq_ID_, station_code = self._search_database()

        # Sample size of the filtered database
        nBig = sampleBig.shape[0]

        # Scale factors based on mse
        scaleFac = np.array(
            np.sum(np.matlib.repmat(target_spec, nBig, 1) * sampleBig, axis=1) / np.sum(sampleBig ** 2, axis=1))

        # Find best matches to the target spectrum from ground-motion database
        temp = (np.matlib.repmat(target_spec, nBig, 1) - sampleBig) ** 2
        mse = temp.mean(axis=1)

        if self.database['Name'].startswith('ESM'):
            d = {ni: indi for indi, ni in enumerate(set(eq_ID_.tolist()))}
            eq_ID = np.asarray([d[ni] for ni in eq_ID_.tolist()])
        else:
            eq_ID = eq_ID_.copy()

        recID_sorted = np.argsort(mse)
        recIDs = np.ones(self.nGM, dtype=int) * (-1)
        eqIDs = np.ones(self.nGM, dtype=int) * (-1)
        idx1 = 0
        idx2 = 0
        while idx1 < self.nGM:  # not more than 3 of the records should be from the same event
            tmp1 = recID_sorted[idx2]
            idx2 += 1
            tmp2 = eq_ID[tmp1]
            recIDs[idx1] = tmp1
            eqIDs[idx1] = tmp2
            if np.sum(eqIDs == tmp2) <= self.RecPerEvent:
                idx1 += 1

        # Initial selection results - based on MSE
        finalScaleFac = scaleFac[recIDs]
        sampleSmall = sampleBig[recIDs, :]
        target_spec[0] = target_spec[0] / 0.9  # scale up for Sa(T[0]) or PGA

        # Must not be lower than target within the period range, find the indicies for this period range
        idxs = np.where((self.database['Periods'] >= self.Tlower) * (self.database['Periods'] <= self.Tupper))[0]
        idxs = np.append(0, idxs)  # Add Sa(T=0) or PGA, approximated as Sa(T=0.01)

        if self.opt == 1:
            self.rec_scale = finalScaleFac * np.max(
                target_spec[idxs] / (finalScaleFac.reshape(-1, 1) * sampleSmall[:, idxs]).mean(axis=0))

        # try to optimize scaling factor to make it closest as possible to 1
        if self.opt == 2:
            finalScaleFac = np.max(target_spec[idxs] / sampleSmall[:, idxs].mean(axis=0))
            for i in range(self.nGM):  # Loop for nGM
                # note the ID of the record which is removed
                minID = recIDs[i]
                # remove the i'th record search for a candidate, and consider critical periods for error calculations only
                sampleSmall_reduced = np.delete(sampleSmall[:, idxs], i, 0)
                recIDs = np.delete(recIDs, i)
                eqIDs = np.delete(eqIDs, i)
                # Try to add a new spectra to the subset list
                minID, finalScaleFac = self._opt2(sampleSmall_reduced, finalScaleFac, target_spec[idxs], recIDs, eqIDs,
                                                  minID, nBig, eq_ID, sampleBig[:, idxs], self.RecPerEvent)
                # Add new element in the right slot
                sampleSmall = np.concatenate(
                    (sampleSmall[:i, :], sampleBig[minID, :].reshape(1, sampleBig.shape[1]), sampleSmall[i:, :]),
                    axis=0)
                recIDs = np.concatenate((recIDs[:i], np.array([minID]), recIDs[i:]))
                eqIDs = np.concatenate((eqIDs[:i], np.array([eq_ID[minID]]), eqIDs[i:]))
            self.rec_scale = np.ones(self.nGM) * float(finalScaleFac)

        # check the scaling
        if np.any(self.rec_scale > self.maxScale) or np.any(self.rec_scale < 1 / self.maxScale):
            raise ValueError('Scaling factor criteria is not satisfied',
                             'Please broaden your selection and scaling criteria or change the optimization scheme...')

        recIDs = recIDs.tolist()
        # Add selected record information to self
        self.rec_Vs30 = Vs30[recIDs]
        self.rec_Rjb = Rjb[recIDs]
        self.rec_Mw = Mw[recIDs]
        self.rec_fault = fault[recIDs]
        self.rec_eqID = eq_ID[recIDs]
        self.rec_h1 = Filename_1[recIDs]

        if self.selection == 2:
            self.rec_h2 = Filename_2[recIDs]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recIDs]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[recIDs]

        rec_idxs = []
        if self.selection == 1:
            SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_h1:
                rec_idxs.append(np.where(Filename_1 == rec)[0][0])
            self.rec_spec = SaKnown[rec_idxs, :]

        elif self.selection == 2:
            for rec in self.rec_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            self.rec_spec = 0.5 * (self.database['Sa_1'][rec_idxs, :] + self.database['Sa_2'][rec_idxs, :])

        # Save the results for whole spectral range
        self.T = self.database['Periods']
        if self.target_path:
            self.target = func(self.T)
        else:
            self.target = Sae_ec8_part1(ag, xi, self.T, ImpClass, Type, SiteClass)

        print('EC8 - Part 1 based ground motion record selection and amplitude scaling are finished...')
