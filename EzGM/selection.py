"""
Ground motion record selection toolbox
"""
# Possible future developments
# TODO: Seems that exact CS computation via OpenQuake engine is possible. Add separate class for exact CS, and name
#  the previous one as approximate. Nonetheless, what need to be figured out is how to get correlation matrix directly.
# TODO: Add 3 component selection: https://github.com/bakerjw/CS_Selection
# TODO: Add spectral matching methods for record selection (e.g., REQPY, https://github.com/LuisMontejo/REQPY)
# TODO: Add generalized conditional intensity measure approach (GCIM) to select ground motion records

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
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from .webdriverdownloader import ChromeDriverDownloader, GeckoDriverDownloader
from numba import njit
from openquake.hazardlib import gsim, imt, const
from .utility import make_dir, content_from_zip, read_nga, read_esm, get_esm_token
from .utility import site_parameters_tbec2018, sae_tbec2018, site_parameters_asce7_16, sae_asce7_16, sae_ec8_part1
from .utility import random_multivariate_normal


SMALL_SIZE = 15
MEDIUM_SIZE = 16
BIG_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class _SubClass_:
    """
    Details
    -------
    This sub-class contains common methods inherited by the two parent classes:
    ConditionalSpectrum and CodeSpectrum.
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
        sample_big : numpy.ndarray
            An array which contains the IMLs from filtered database.
        vs30 : numpy.ndarray
            An array which contains the Vs30s from filtered database.
        magnitude : numpy.ndarray
            An array which contains the magnitudes from filtered database.
        rjb : numpy.ndarray
            An array which contains the Rjbs from filtered database.
        mechanism : numpy.ndarray
            An array which contains the fault mechanism info from filtered database.
        filename1 : numpy.ndarray
            An array which contains the filename of 1st gm component from filtered database.
            If selection is set to 1, it will include filenames of both components.
        filename2 : numpy.ndarray
            An array which contains the filename of 2nd gm component filtered database.
            If selection is set to 1, it will be None value.
        nga_num : numpy.ndarray
            If NGA_W2 is used as record database, record sequence numbers from filtered
            database will be saved, for other databases this variable is None.
        eq_id : numpy.ndarray
            An array which contains event ids from filtered database.
        station_code : numpy.ndarray
            If ESM_2018 is used as record database, station codes from filtered
            database will be saved, for other databases this variable is None.
        """

        if self.num_components == 1:  # sa_known is from arbitrary ground motion component

            sa_known = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            vs30 = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
            magnitude = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)
            rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
            mechanism = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
            filename1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            eq_id = np.append(self.database['EQID'], self.database['EQID'], axis=0)

            if self.database['Name'] == "NGA_W2":
                nga_num = np.append(self.database['NGA_num'], self.database['NGA_num'], axis=0)

            elif self.database['Name'] == "ESM_2018":
                station_code = np.append(self.database['station_code'], self.database['station_code'], axis=0)

        elif self.num_components == 2:

            if self.spectrum_definition == 'GeoMean':
                sa_known = np.sqrt(self.database['Sa_1'] * self.database['Sa_2'])
            elif self.spectrum_definition == 'SRSS':
                sa_known = np.sqrt(self.database['Sa_1'] ** 2 + self.database['Sa_2'] ** 2)
            elif self.spectrum_definition == 'ArithmeticMean':
                sa_known = (self.database['Sa_1'] + self.database['Sa_2']) / 2
            elif self.spectrum_definition == 'RotD50':  # sa_known = Sa_RotD50.
                sa_known = self.database['Sa_RotD50']
            elif self.spectrum_definition == 'RotD100':  # sa_known = Sa_RotD100.
                sa_known = self.database['Sa_RotD100']
            else:
                raise ValueError('Unexpected Sa definition, exiting...')

            vs30 = self.database['soil_Vs30']
            magnitude = self.database['magnitude']
            rjb = self.database['Rjb']
            mechanism = self.database['mechanism']
            filename1 = self.database['Filename_1']
            filename2 = self.database['Filename_2']
            eq_id = self.database['EQID']

            if self.database['Name'] == "NGA_W2":
                nga_num = self.database['NGA_num']

            elif self.database['Name'] == "ESM_2018":
                station_code = self.database['station_code']

        else:
            raise ValueError('Selection can only be performed for one or two components at the moment, exiting...')

        # Limiting the records to be considered using the `not_allowed' variable
        # Sa cannot be negative or zero, remove these.
        not_allowed = np.unique(np.where(sa_known <= 0)[0]).tolist()

        if self.vs30_limits is not None:  # limiting values on soil exist
            mask = (vs30 > min(self.vs30_limits)) * (vs30 < max(self.vs30_limits))
            temp = [i for i, x in enumerate(mask) if not x]
            not_allowed.extend(temp)

        if self.mag_limits is not None:  # limiting values on magnitude exist
            mask = (magnitude > min(self.mag_limits)) * (magnitude < max(self.mag_limits))
            temp = [i for i, x in enumerate(mask) if not x]
            not_allowed.extend(temp)

        if self.rjb_limits is not None:  # limiting values on Rjb exist
            mask = (rjb > min(self.rjb_limits)) * (rjb < max(self.rjb_limits))
            temp = [i for i, x in enumerate(mask) if not x]
            not_allowed.extend(temp)

        if self.mech_limits is not None:  # limiting values on mechanism exist
            for mechanism_i in range(len(self.mech_limits)):
                if mechanism_i == 0:
                    mask = mechanism == self.mech_limits[mechanism_i]
                else:
                    mask = np.logical_or(mask, mechanism == self.mech_limits[mechanism_i])
            temp = [i for i, x in enumerate(mask) if not x]
            not_allowed.extend(temp)

        # get the unique values
        not_allowed = (list(set(not_allowed)))
        allowed = [i for i in range(sa_known.shape[0])]
        for i in not_allowed:
            allowed.remove(i)

        # Use only allowed records
        sa_known = sa_known[allowed, :]
        vs30 = vs30[allowed]
        magnitude = magnitude[allowed]
        rjb = rjb[allowed]
        mechanism = mechanism[allowed]
        eq_id = eq_id[allowed]
        filename1 = filename1[allowed]

        if self.num_components == 1:
            filename2 = None
        else:
            filename2 = filename2[allowed]

        if self.database['Name'] == "NGA_W2":
            nga_num = nga_num[allowed]
            station_code = None
        elif self.database['Name'] == "ESM_2018":
            nga_num = None
            station_code = station_code[allowed]

        # Arrange the available spectra in a usable format and check for invalid input
        # Match periods (known periods and periods for error computations)
        record_periods = []
        for i in range(len(self.periods)):
            record_periods.append(np.where(self.database['Periods'] == self.periods[i])[0][0])

        # Check for invalid input
        sample_big = sa_known[:, record_periods]
        if np.any(np.isnan(sample_big)):
            raise ValueError('NaNs found in input response spectra')

        if self.num_records > len(eq_id):
            raise ValueError('There are not enough records which satisfy',
                             'the given record selection criteria...',
                             'Please use broaden your selection criteria...')

        return sample_big, vs30, magnitude, rjb, mechanism, filename1, filename2, nga_num, eq_id, station_code

    def write(self, object=0, records=1, record_type='acc', zip_parent_path=''):
        """
        Details
        -------
        Writes the object as pickle, selected and scaled records as .txt files.

        Parameters
        ----------
        object : int, optional
            flag to write the object into the pickle file.
            The default is 0.
        records : int, optional
            flag to write the selected and scaled time histories
            time-steps, filenames, scaling factors.
            The default is 1.
        record_type : str, optional
            option to choose the type of time history to be written.
            'acc' : for the acceleration series, units: g
            'vel' : for the velocity series, units: g * sec
            'disp': for the displacement series: units: g * sec2
        zip_parent_path : str, optional
            This is option could be used if the user already has all the
            records in database. This is the folder path which contains
            "database.zip" file (e.g., database could be NGA_W2 or ESM_2018). 
            The records must be placed inside zip_parent_path/database.zip/database/
            The default is ''.

        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """

        def save_signal(path, unscaled_acc, sf, dt):
            """
            Details
            -------
            Saves the final signal to the specified path.

            Parameters
            ----------
            path : str
                path of the file to save
            unscaled_acc : numpy.ndarray
                unscaled acceleration series
            sf : float
                scaling factor
            dt : float
                time step 

            Returns
            -------
            None.
            """

            if record_type == 'vel':  # integrate once if velocity
                signal = integrate.cumtrapz(unscaled_acc * sf, dx=dt, initial=0)

            elif record_type == 'disp':  # integrate twice if displacement
                signal = integrate.cumtrapz(integrate.cumtrapz(unscaled_acc * sf, dx=dt, initial=0), dx=dt, initial=0)

            else:
                signal = unscaled_acc * sf

            np.savetxt(path, signal, fmt='%1.5e')

        if records == 1:
            # set the directories and file names
            try:  # this will work if records are downloaded
                zip_name = self.unscaled_rec_file
            except AttributeError:
                zip_name = os.path.join(zip_parent_path, self.database['Name'] + '.zip')
            size = len(self.rec_file_h1)
            dts = np.zeros(size)
            path_h1 = os.path.join(self.output_directory_path, 'GMR_names.txt')
            if self.num_components == 2:
                path_h1 = os.path.join(self.output_directory_path, 'GMR_H1_names.txt')
                path_h2 = os.path.join(self.output_directory_path, 'GMR_H2_names.txt')
                h2s = open(path_h2, 'w')
            h1s = open(path_h1, 'w')

            # Get record paths for # NGA_W2 or ESM_2018
            if zip_name != os.path.join(zip_parent_path, self.database['Name'] + '.zip'):
                rec_paths1 = self.rec_file_h1
                if self.num_components == 2:
                    rec_paths2 = self.rec_file_h2
            else:
                rec_paths1 = [self.database['Name'] + '/' + self.rec_file_h1[i] for i in range(size)]
                if self.num_components == 2:
                    rec_paths2 = [self.database['Name'] + '/' + self.rec_file_h2[i] for i in range(size)]

            # Read contents from zipfile
            contents1 = content_from_zip(rec_paths1, zip_name)  # H1 gm components
            if self.num_components == 2:
                contents2 = content_from_zip(rec_paths2, zip_name)  # H2 gm components

            # Start saving records
            for i in range(size):

                # Read the record files
                if self.database['Name'].startswith('NGA'):  # NGA
                    dts[i], npts1, _, _, inp_acc1 = read_nga(in_filename=self.rec_file_h1[i], content=contents1[i])
                    gmr_file1 = self.rec_file_h1[i].replace('/', '_')[:-4] + '_' + record_type.upper() + '.txt'

                    if self.num_components == 2:  # H2 component
                        _, npts2, _, _, inp_acc2 = read_nga(in_filename=self.rec_file_h2[i], content=contents2[i])
                        gmr_file2 = self.rec_file_h2[i].replace('/', '_')[:-4] + '_' + record_type.upper() + '.txt'

                elif self.database['Name'].startswith('ESM'):  # ESM
                    dts[i], npts1, _, _, inp_acc1 = read_esm(in_filename=self.rec_file_h1[i], content=contents1[i])
                    gmr_file1 = self.rec_file_h1[i].replace('/', '_')[:-11] + '_' + record_type.upper() + '.txt'
                    if self.num_components == 2:  # H2 component
                        _, npts2, _, _, inp_acc2 = read_esm(in_filename=self.rec_file_h2[i], content=contents2[i])
                        gmr_file2 = self.rec_file_h2[i].replace('/', '_')[:-11] + '_' + record_type.upper() + '.txt'

                # Write the record files
                if self.num_components == 2:
                    # ensure that two acceleration signals have the same length, if not add zeros.
                    npts = max(npts1, npts2)
                    temp1 = np.zeros(npts)
                    temp1[:npts1] = inp_acc1
                    inp_acc1 = temp1.copy()
                    temp2 = np.zeros(npts)
                    temp2[:npts2] = inp_acc2
                    inp_acc2 = temp2.copy()

                    # H2 component
                    save_signal(os.path.join(self.output_directory_path, gmr_file2), inp_acc2, self.rec_scale_factors[i], dts[i])
                    h2s.write(gmr_file2 + '\n')

                # H1 component
                save_signal(os.path.join(self.output_directory_path, gmr_file1), inp_acc1, self.rec_scale_factors[i], dts[i])
                h1s.write(gmr_file1 + '\n')

            # Time steps
            np.savetxt(os.path.join(self.output_directory_path, 'GMR_dts.txt'), dts, fmt='%.5f')
            # Scale factors
            np.savetxt(os.path.join(self.output_directory_path, 'GMR_sf_used.txt'), np.array([self.rec_scale_factors]).T, fmt='%1.5f')
            # Close the files
            h1s.close()
            if self.num_components == 2:
                h2s.close()

        if object == 1:
            # save some info as pickle obj
            object = vars(copy.deepcopy(self))  # use copy.deepcopy to create independent obj
            object['database'] = self.database['Name']
            del object['output_directory_path']

            if 'bgmpe' in object:
                del object['bgmpe']

            with open(os.path.join(self.output_directory_path, 'obj.pkl'), 'wb') as file:
                pickle.dump(object, file)

        print(f"Finished writing process, the files are located in\n{self.output_directory_path}")

    def plot(self, target=0, simulations=0, records=1, save=0, show=1):
        """
        Details
        -------
        Plots the spectra of selected and simulated records,
        and/or target spectrum.

        Parameters
        ----------
        target : int, optional for ConditionalSpectrum
            Flag to plot target spectrum.
            The default is 1.
        simulations : int, optional for ConditionalSpectrum
            Flag to plot simulated response spectra vs. target spectrum.
            The default is 0.
        records : int, optional for ConditionalSpectrum
            Flag to plot Selected response spectra of selected records
            vs. target spectrum.
            The default is 1.
        save : int, optional for all selection options
            Flag to save plotted figures in pdf format.
            The default is 0.
        show : int, optional for all selection options
            Flag to show figures
            The default is 0.

        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """

        plt.ioff()

        if type(self).__name__ == 'ConditionalSpectrum':

            # xticks and yticks to use for plotting
            xticks = [self.periods[0]]
            for x in [0.01, 0.1, 0.2, 0.5, 1, 5, 10]:
                if self.periods[0] < x < self.periods[-1]:
                    xticks.append(x)
            xticks.append(self.periods[-1])
            yticks = [0.01, 0.1, 0.2, 0.5, 1, 2, 3, 5]

            if self.is_conditioned == 1:
                if len(self.Tstar) == 1:
                    hatch = [float(self.Tstar * 0.98), float(self.Tstar * 1.02)]
                else:
                    hatch = [float(self.Tstar.min()), float(self.Tstar.max())]

            if target == 1:
                # Plot Target spectrum vs. Simulated response spectra
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                plt.suptitle('Target Spectrum', y=0.95)
                ax[0].loglog(self.periods, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
                if self.use_variance == 1:
                    ax[0].loglog(self.periods, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.periods, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].set_xlim([self.periods[0], self.periods[-1]])
                ax[0].set_xticks(xticks)
                ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
                ax[0].get_xaxis().set_minor_formatter(NullFormatter())
                ax[0].set_yticks(yticks)
                ax[0].get_yaxis().set_major_formatter(ScalarFormatter())
                ax[0].get_yaxis().set_minor_formatter(NullFormatter())
                ax[0].set_xlabel('Period [sec]')
                ax[0].set_ylabel('Spectral Acceleration [g]')
                ax[0].grid(True)

                handles, labels = ax[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
                if self.is_conditioned == 1:
                    ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                # Sample and target standard deviations
                if self.use_variance == 1:
                    ax[1].semilogx(self.periods, self.sigma_ln, color='red', linestyle='--', lw=2,
                                   label='Target - $\sigma_{ln}$')
                    ax[1].set_xlabel('Period [sec]')
                    ax[1].set_ylabel('Dispersion')
                    ax[1].grid(True)
                    ax[1].legend(frameon=False)
                    ax[1].set_xlim([self.periods[0], self.periods[-1]])
                    ax[1].set_xticks(xticks)
                    ax[1].get_xaxis().set_major_formatter(ScalarFormatter())
                    ax[1].get_xaxis().set_minor_formatter(NullFormatter())
                    ax[1].set_ylim(bottom=0)
                    ax[1].get_yaxis().set_major_formatter(ScalarFormatter())
                    ax[1].get_yaxis().set_minor_formatter(NullFormatter())
                    if self.is_conditioned == 1:
                        ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if save == 1:
                    plt.savefig(os.path.join(self.output_directory_path, 'Targeted.pdf'))

            if simulations == 1:
                # Plot Target spectrum vs. Simulated response spectra
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                plt.suptitle('Target Spectrum vs. Simulated Spectra', y=0.95)

                for i in range(self.num_records):
                    ax[0].loglog(self.periods, np.exp(self.sim_spec[i, :]), color='gray', lw=1, label='Selected')

                ax[0].loglog(self.periods, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
                if self.use_variance == 1:
                    ax[0].loglog(self.periods, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.periods, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].loglog(self.periods, np.exp(np.mean(self.sim_spec, axis=0)), color='blue', lw=2,
                             label='Selected - $e^{\mu_{ln}}$')
                if self.use_variance == 1:
                    ax[0].loglog(self.periods, np.exp(np.mean(self.sim_spec, axis=0) + 2 * np.std(self.sim_spec, axis=0)),
                                 color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.periods, np.exp(np.mean(self.sim_spec, axis=0) - 2 * np.std(self.sim_spec, axis=0)),
                                 color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].set_xlim([self.periods[0], self.periods[-1]])
                ax[0].set_xticks(xticks)
                ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
                ax[0].get_xaxis().set_minor_formatter(NullFormatter())
                ax[0].set_yticks(yticks)
                ax[0].get_yaxis().set_major_formatter(ScalarFormatter())
                ax[0].get_yaxis().set_minor_formatter(NullFormatter())
                ax[0].set_xlabel('Period [sec]')
                ax[0].set_ylabel('Spectral Acceleration [g]')
                ax[0].grid(True)
                handles, labels = ax[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
                if self.is_conditioned == 1:
                    ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if self.use_variance == 1:
                    # Sample and target standard deviations
                    ax[1].semilogx(self.periods, self.sigma_ln, color='red', linestyle='--', lw=2,
                                   label='Target - $\sigma_{ln}$')
                    ax[1].semilogx(self.periods, np.std(self.sim_spec, axis=0), color='black', linestyle='--', lw=2,
                                   label='Selected - $\sigma_{ln}$')
                    ax[1].set_xlabel('Period [sec]')
                    ax[1].set_ylabel('Dispersion')
                    ax[1].grid(True)
                    ax[1].legend(frameon=False)
                    ax[1].set_xlim([self.periods[0], self.periods[-1]])
                    ax[1].set_xticks(xticks)
                    ax[1].get_xaxis().set_major_formatter(ScalarFormatter())
                    ax[1].get_xaxis().set_minor_formatter(NullFormatter())
                    ax[1].set_ylim(bottom=0)
                    ax[1].get_yaxis().set_major_formatter(ScalarFormatter())
                    ax[1].get_yaxis().set_minor_formatter(NullFormatter())
                    if self.is_conditioned == 1:
                        ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if save == 1:
                    plt.savefig(os.path.join(self.output_directory_path, 'Simulated.pdf'))

            if records == 1:
                # Plot Target spectrum vs. Selected response spectra
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                plt.suptitle('Target Spectrum vs. Spectra of Selected Records', y=0.95)

                for i in range(self.num_records):
                    ax[0].loglog(self.periods, np.exp(self.rec_sa_ln[i, :]), color='gray', lw=1, label='Selected')

                ax[0].loglog(self.periods, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
                if self.use_variance == 1:
                    ax[0].loglog(self.periods, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                    ax[0].loglog(self.periods, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                                 label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].loglog(self.periods, np.exp(np.mean(self.rec_sa_ln, axis=0)), color='blue', lw=2,
                             label='Selected - $e^{\mu_{ln}}$')
                ax[0].loglog(self.periods, np.exp(np.mean(self.rec_sa_ln, axis=0) + 2 * np.std(self.rec_sa_ln, axis=0)),
                             color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.periods, np.exp(np.mean(self.rec_sa_ln, axis=0) - 2 * np.std(self.rec_sa_ln, axis=0)),
                             color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

                ax[0].set_xlim([self.periods[0], self.periods[-1]])
                ax[0].set_xticks(xticks)
                ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
                ax[0].get_xaxis().set_minor_formatter(NullFormatter())
                ax[0].set_yticks(yticks)
                ax[0].get_yaxis().set_major_formatter(ScalarFormatter())
                ax[0].get_yaxis().set_minor_formatter(NullFormatter())
                ax[0].set_xlabel('Period [sec]')
                ax[0].set_ylabel('Spectral Acceleration [g]')
                ax[0].grid(True)
                handles, labels = ax[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
                if self.is_conditioned == 1:
                    ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                # Sample and target standard deviations
                ax[1].semilogx(self.periods, self.sigma_ln, color='red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
                ax[1].semilogx(self.periods, np.std(self.rec_sa_ln, axis=0), color='black', linestyle='--', lw=2,
                               label='Selected - $\sigma_{ln}$')
                ax[1].set_xlabel('Period [sec]')
                ax[1].set_ylabel('Dispersion')
                ax[1].grid(True)
                ax[1].legend(frameon=False)
                ax[1].set_xlim([self.periods[0], self.periods[-1]])
                ax[1].set_xticks(xticks)
                ax[1].get_xaxis().set_major_formatter(ScalarFormatter())
                ax[1].get_xaxis().set_minor_formatter(NullFormatter())
                ax[1].set_ylim(bottom=0)
                ax[1].get_yaxis().set_major_formatter(ScalarFormatter())
                ax[1].get_yaxis().set_minor_formatter(NullFormatter())
                if self.is_conditioned == 1:
                    ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

                if save == 1:
                    plt.savefig(os.path.join(self.output_directory_path, 'Selected.pdf'))

        if type(self).__name__ == 'CodeSpectrum':

            hatch = [self.lower_bound_period, self.upper_bound_period]
            # Plot Target spectrum vs. Selected response spectra
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            for i in range(self.rec_sa_ln.shape[0]):
                ax.plot(self.periods, self.rec_sa_ln[i, :] * self.rec_scale_factors[i], color='gray', lw=1, label='Selected')
            ax.plot(self.periods, np.mean(self.rec_sa_ln * self.rec_scale_factors.reshape(-1, 1), axis=0), color='black', lw=2,
                    label='Selected Mean')

            if self.code == 'TBEC 2018':
                ax.plot(self.periods, self.target, color='red', lw=2, label='Design Response Spectrum')
                if self.num_components == 2:
                    ax.plot(self.periods, 1.3 * self.target, color='red', ls='--', lw=2,
                            label='1.3 x Design Response Spectrum')

            if self.code == 'ASCE 7-16':
                ax.plot(self.periods, self.target, color='red', lw=2, label='$MCE_{R}$ Response Spectrum')
                ax.plot(self.periods, 0.9 * self.target, color='red', ls='--', lw=2,
                        label='0.9 x $MCE_{R}$ Response Spectrum')

            if self.code == 'EC8-Part1':
                ax.plot(self.periods, self.target, color='red', lw=2, label='Design Response Spectrum')
                ax.plot(self.periods, 0.9 * self.target, color='red', lw=2, ls='--', label='0.9 x Design Response Spectrum')

            ax.axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
            ax.set_xlabel('Period [sec]')
            ax.set_ylabel('Spectral Acceleration [g]')
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), frameon=False)
            ax.set_xlim([self.periods[0], self.upper_bound_period * 2])
            plt.suptitle(f'Spectra of Selected Records ({self.code})', y=0.95)

            if save == 1:
                plt.savefig(os.path.join(self.output_directory_path, 'Selected.pdf'))

        # Show the figure
        if show == 1:
            plt.show()

        plt.close('all')

    def download(self, username=None, password=None, token_path=None, sleeptime=2, browser='chrome'):
        """
        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from either
        NGA-West2 (https://ngawest2.berkeley.edu/) or ESM databases (https://esm-db.eu/).

        Notes
        -----
        Either of google-chrome or mozilla-firefox should have been installed priorly to download from NGA-West2.

        Parameters
        ----------
        username : str
            Account username (e-mail),  e.g. 'example_username@email.com'.
        password : str
            Account password, e.g. 'example_password123456'.
        sleeptime : int, optional
            Time (sec) spent between each browser operation. This can be increased or decreased depending on the internet speed.
            Used in the case of database='NGA_W2'
            The default is 2
        browser : str, optional
            The browser to use for download purposes. Valid entries are: 'chrome' or 'firefox'. 
            Used in the case of database='NGA_W2'
            The default is 'chrome'.

        Returns
        -------
        None
        """
        
        if self.database['Name'] == 'ESM_2018':

            if token_path is None:
                # In order to access token file must be retrieved initially.
                # copy paste the readily available token.txt into EzGM or generate new one using get_esm_token method.
                if username is None or password is None:
                    raise ValueError('You have to enter either credentials or path to the token to download records from ESM database')
                else:
                    get_esm_token(username, password)
                    token_path = 'token.txt'

            self._esm2018_download(token_path)

        elif self.database['Name'] == 'NGA_W2':

            if username is None or password is None:
                raise ValueError('You have to enter either credentials  to download records from NGA-West2 database')

            self._ngaw2_download(username, password, sleeptime, browser)

        else:
            raise NotImplementedError('You have to use either of ESM_2018 or NGA_W2 databases to use download method.')

    def _esm2018_download(self, token_path=None):
        """

        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from ESM database
        (https://esm-db.eu/) based on their event ID and station_codes.

        Parameters
        ----------
        username : str
            Account username (e-mail),  e.g. 'example_username@email.com'.
        password : str
            Account password, e.g. 'example_password123456'.
        

        Returns
        -------
        None.

        """

        print('\nStarted executing download method to retrieve selected records from https://esm-db.eu')

        # temporary zipfile name
        zip_temp = os.path.join(self.output_directory_path, 'output_temp.zip')
        # temporary folder to extract files
        folder_temp = os.path.join(self.output_directory_path, 'output_temp')

        for i in range(self.num_records):
            print('Downloading %d/%d...' % (i + 1, self.num_records))
            event = self.rec_eq_id[i]
            station = self.rec_station_code[i]
            params = (
                ('eventid', event),
                ('data-type', 'ACC'),
                ('station', station),
                ('format', 'ascii'),
            )
            files = {
                'message': ('path/to/token.txt', open(token_path, 'rb')),
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
        file_name = os.path.join(self.output_directory_path, f'unscaled_records_{time_tag_str}.zip')
        with zipfile.ZipFile(file_name, 'w', zipfile.ZIP_DEFLATED) as zipObj:
            len_dir_path = len(folder_temp)
            for root, _, files in os.walk(folder_temp):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipObj.write(file_path, file_path[len_dir_path:])

        shutil.rmtree(folder_temp)
        self.unscaled_rec_file = file_name
        print(f'Downloaded files are located in\n{self.unscaled_rec_file}')

    def _ngaw2_download(self, username, password, sleeptime, browser):
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
        username : str
            Account username (e-mail),  e.g. 'example_username@email.com'.
        password : str
            Account password, e.g. 'example_password123456'.
        sleeptime : int, default is 3
            Time (sec) spent between each browser operation. This can be increased or decreased depending on the internet speed.
        browser : str, default is 'chrome'
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
            download_dir : str
                Directory for the output time histories to be downloaded

            Returns
            -------
            total_size : float
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
            download_dir : str
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
            download_dir : str
                Directory for the output time histories to be downloaded.
            browser : str, default is 'chrome'
                The browser to use for download purposes. Valid entries are: 'chrome' or 'firefox'

            Returns
            -------
            driver : selenium webdriver object
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
                    driver_path = gdd.download_and_install(version = 'compatible')
                    options = webdriver.ChromeOptions()
                    prefs = {"download.default_directory": download_dir}
                    options.add_experimental_option("prefs", prefs)
                    options.headless = True
                    driver = webdriver.Chrome(executable_path=driver_path[1], options=options)

                print('Webdriver is obtained successfully.')

                return driver

            except RuntimeError:
                print('Failed to get webdriver.')
                raise

        def sign_in(driver, username, password):
            """

            Details
            -------
            This function signs in to 'https://ngawest2.berkeley.edu/' with
            given account credentials.

            Parameters
            ----------
            driver : selenium webdriver object
                Driver object used to download NGA_W2 records.
            username : str
                Account username (e-mail), e.g.: 'username@mail.com'.
            password : str
                Account password, e.g.: 'password!12345'.

            Returns
            -------
            driver : selenium webdriver object
                Driver object used to download NGA_W2 records.

            """
            # TODO: For Selenium >= 4.3.0
            #  Deprecated find_element_by_* and find_elements_by_* are now removed (#10712)
            #  https://stackoverflow.com/questions/72773206/selenium-python-attributeerror-webdriver-object-has-no-attribute-find-el
            #  Modify the ngaw2_download method to use greater versions of Selenium than 4.2.0
            print("Signing in with credentials...")
            driver.get('https://ngawest2.berkeley.edu/users/sign_in')
            driver.find_element_by_id('user_email').send_keys(username)
            driver.find_element_by_id('user_password').send_keys(password)
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

        def download(rsn, download_dir, driver):
            """

            Details
            -------
            This function dowloads the timehistories which have been indicated with their record sequence numbers (rsn)
            from 'https://ngawest2.berkeley.edu/'.

            Parameters
            ----------
            rsn : str
                A string variable contains RSNs to be downloaded which uses ',' as delimiter
                between RNSs, e.g.: '1,5,91,35,468'.
            download_dir : str
                Directory for the output time histories to be downloaded.
            driver : class object, (selenium webdriver)
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
            driver.find_element_by_id('search_search_nga_number').send_keys(rsn)
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
                driver.save_screenshot(os.path.join(self.output_directory_path, 'download_error.png'))
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

        print('\nStarted executing download method to retrieve selected records from https://ngawest2.berkeley.edu')

        self.username = username
        self.pwd = password
        driver = set_driver(self.output_directory_path, browser)
        driver = sign_in(driver, self.username, self.pwd)
        rsn = ''
        for i in self.rec_rsn:
            rsn += str(int(i)) + ','
        rsn = rsn[:-1:]
        files_before_download = set(os.listdir(self.output_directory_path))
        download(rsn, self.output_directory_path, driver)
        files_after_download = set(os.listdir(self.output_directory_path))
        downloaded_file = str(list(files_after_download.difference(files_before_download))[0])
        file_extension = downloaded_file[downloaded_file.find('.')::]
        time_tag = gmtime()
        time_tag_str = f'{time_tag[0]}'
        for i in range(1, len(time_tag)):
            time_tag_str += f'_{time_tag[i]}'
        new_file_name = f'unscaled_records_{time_tag_str}{file_extension}'
        downloaded_file = os.path.join(self.output_directory_path, downloaded_file)
        downloaded_file_rename = os.path.join(self.output_directory_path, new_file_name)
        os.rename(downloaded_file, downloaded_file_rename)
        self.unscaled_rec_file = downloaded_file_rename
        print(f'Downloaded files are located in\n{self.unscaled_rec_file}')


class ConditionalSpectrum(_SubClass_):
    """
    This class is used to
        1) Create target spectrum
            Unconditional spectrum using specified gmpe
            Conditional spectrum using average spectral acceleration
            Conditional spectrum using spectral acceleration
            with and without considering variance
        2) Selecting suitable ground motion sets for target spectrum
        3) Scaling and processing of selected ground motion records

    Kohrangi et al. 2017 verified that the mean and standard deviation of AvgSA estimates
    based on the indirect method are indeed robust and can be used in real life applications.
    However, pairing GMPE-SAs and SA correlation models based
    on different ground motion databases should be avoided because it may result in standard
    deviations of AvgSA that are biased compared to those of the direct method.
    Reference:
    Kohrangi, M., Kotha, S. R., & Bazzurro, P. (2017). Ground-motion models for average spectral acceleration in a
    period range: direct and indirect methods. In Bulletin of Earthquake Engineering, 16(1): 45–65.
    Springer Science and Business Media LLC. https://doi.org/10.1007/s10518-017-0216-5

    Baker and Bradley 2017 described that IM correlations they have studied are stable across a range of conditions,
    and as a result, that existing correlation models are generally appropriate for continued use
    in engineering calculations. The results they present is based on NGA-West2 ground motion record database.
    Reference:
    Baker, J. W., & Bradley, B. A. (2017). Intensity Measure Correlations Observed in the NGA-West2 Database,
    and Dependence of Correlations on Rupture and Site Parameters. In Earthquake Spectra, 33(1): 145–156.
    SAGE Publications. https://doi.org/10.1193/060716eqs095m

    """

    def __init__(self, database='NGA_W2', output_directory='Outputs', obj_path=None):
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
        output_directory : str, optional.
            output directory to create.
            The default is 'Outputs'
        obj_path : str, optional
            This is the path to the previously saved obj.pkl file by EzGM.
            One can use the previously saved instance and use rest of the methods.
            The default is None.

        Returns
        -------
        None.
        """

        # Inherit the subclass
        super().__init__()

        # Read the old EzGM obj
        if obj_path:
            with open('obj.pkl', 'rb') as file:
                obj = pickle.load(file)
            self.__dict__.update(obj)
            database = self.database

        # Add the input the ground motion database to use
        # TODO: Combine all metadata into single jason file. Not essential, but makes the code more elegant.
        matfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Meta_Data', database)
        self.database = loadmat(matfile, squeeze_me=True)
        self.database['Name'] = database

        # create the output directory and add the path to self
        cwd = os.getcwd()
        outdir_path = os.path.join(cwd, output_directory)
        self.output_directory_path = outdir_path
        make_dir(self.output_directory_path)

    @staticmethod
    def _baker_jayaram_correlation_model(period1, period2):
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
        period1 : float
            First period
        period2 : float
            Second period
    
        Returns
        -------
        rho: float
             Predicted correlation coefficient
        """

        period_min = min(period1, period2)
        period_max = max(period1, period2)

        c1 = 1.0 - np.cos(np.pi / 2.0 - np.log(period_max / max(period_min, 0.109)) * 0.366)

        if period_max < 0.2:
            c2 = 1.0 - 0.105 * (1.0 - 1.0 / (1.0 + np.exp(100.0 * period_max - 5.0))) * (period_max - period_min) / (period_max - 0.0099)
        else:
            c2 = 0

        if period_max < 0.109:
            c3 = c2
        else:
            c3 = c1

        c4 = c1 + 0.5 * (np.sqrt(c3) - c3) * (1.0 + np.cos(np.pi * period_min / 0.109))

        if period_max <= 0.109:
            rho = c2
        elif period_min > 0.109:
            rho = c1
        elif period_max < 0.2:
            rho = min(c2, c4)
        else:
            rho = c4

        return rho

    @staticmethod
    def _akkar_correlation_model(period1, period2):
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
        period1 : float
            First period
        period2 : float
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

        if np.any([period1, period2] < periods[0]) or \
                np.any([period1, period2] > periods[-1]):
            raise ValueError("Period array contains values outside of the "
                             "range supported by the Akkar et al. (2014) "
                             "correlation model")

        if period1 == period2:
            rho = 1.0
        else:
            rho = interpolate.interp2d(periods, periods, coeff_table, kind='linear')(period1, period2)[0]

        return rho

    def _get_correlation(self, period1, period2):
        """
        Details
        -------
        Compute the inter-period correlation for any two Sa(T) values.
        
        Parameters
        ----------
        period1 : float
            First period
        period2 : float
            Second period
                
        Returns
        -------
        rho : float
             Predicted correlation coefficient
        """
        # TODO: Add alternative correlation models: https://github.com/bakerjw/NGAW2_correlations

        correlation_function_handles = {
            'baker_jayaram': self._baker_jayaram_correlation_model,
            'akkar': self._akkar_correlation_model,
        }

        # Check for existing correlation function
        if self.correlation_model not in correlation_function_handles:
            raise ValueError('Not a valid correlation function')
        else:
            rho = \
                correlation_function_handles[self.correlation_model](period1, period2)

        return rho

    def _gmpe_sb_2014_ratios(self, periods):
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
        periods : numpy.ndarray
            Period(s) of interest (sec)

        Returns
        -------
        ratio : float
             geometric mean of Sa_RotD100/Sa_RotD50
        sigma : float
            standard deviation of log(Sa_RotD100/Sa_RotD50)
        """

        # Model coefficient values from Table 1 of the above-reference paper
        periods_orig = np.array(
            [0.0100000000000000, 0.0200000000000000, 0.0300000000000000, 0.0500000000000000, 0.0750000000000000,
             0.100000000000000, 0.150000000000000, 0.200000000000000, 0.250000000000000, 0.300000000000000,
             0.400000000000000, 0.500000000000000, 0.750000000000000, 1, 1.50000000000000, 2, 3, 4, 5,
             7.50000000000000, 10])
        mu_ratios_orig = np.array(
            [1.19243805900000, 1.19124621700000, 1.18767783300000, 1.18649074900000, 1.18767783300000,
             1.18767783300000, 1.19961419400000, 1.20562728500000, 1.21652690500000, 1.21896239400000,
             1.22875320400000, 1.22875320400000, 1.23738465100000, 1.24110237900000, 1.24234410200000,
             1.24358706800000, 1.24732343100000, 1.25985923900000, 1.264908769000, 1.28531008400000,
             1.29433881900000])
        sigma_orig = np.array(
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
             0.08, 0.08, 0.08, 0.08])

        # Interpolate to compute values for the user-specified periods
        mu_ratio = interpolate.interp1d(np.log(periods_orig), mu_ratios_orig)(np.log(periods))
        sigma = interpolate.interp1d(np.log(periods_orig), sigma_orig)(np.log(periods))

        return mu_ratio, sigma

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
        mu_lnAvgsaTstar : float
            Logarithmic mean of intensity measure according to the selected GMPE.
        sigma_lnAvgsaTstar : float
           Logarithmic standard deviation of intensity measure according to the selected GMPE.
        rho_lnSaT_lnAvgsaTstar : numpy.ndarray
            Correlation coefficients.
        """

        len_Tstar = len(self.Tstar)
        mu_lnSaTstar = np.zeros(len_Tstar)
        sigma_lnSaTstar = np.zeros(len_Tstar)
        MoC = np.zeros((len_Tstar, len_Tstar))

        # Get the raw GMPE output (SAs only)
        for i in range(len_Tstar):
            params = self.bgmpe.get_mean_and_stddevs(sctx, rctx, dctx, imt.SA(period=self.Tstar[i]), [const.StdDev.TOTAL])
            mu_lnSaTstar[i] = params[0]
            sigma_lnSaTstar[i] = params[1][0]

            # Modify spectral targets if RotD100 values were specified for two-component selection
            if self.spectrum_definition == 'RotD100' and not 'RotD100' in self.bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT and self.num_components == 2:
                rotD100_mu_ratio, rotD100_sigma = self._gmpe_sb_2014_ratios(self.Tstar[i])
                mu_lnSaTstar[i] = mu_lnSaTstar[i] + np.log(rotD100_mu_ratio)
                sigma_lnSaTstar[i] = (sigma_lnSaTstar[i] ** 2 + rotD100_sigma ** 2) ** 0.5

            # Compute correlations at AvgSA periods
            for j in range(len_Tstar):
                rho_lnSaTstar_lnSaTstar = self._get_correlation(self.Tstar[i], self.Tstar[j])
                MoC[i, j] = rho_lnSaTstar_lnSaTstar

        # Determine logarithmic mean and standard from selected gmpe for IM=AvgSA
        # In case of IM=SA len_period_star becomes 1, thus, also works in this case
        mu_lnAvgsaTstar = (1 / len_Tstar) * sum(mu_lnSaTstar)
        sigma_lnAvgsaTstar = 0
        for i in range(len_Tstar):
            for j in range(len_Tstar):
                sigma_lnAvgsaTstar = sigma_lnAvgsaTstar + (MoC[i, j] * sigma_lnSaTstar[i] * sigma_lnSaTstar[j])
        sigma_lnAvgsaTstar = np.sqrt(sigma_lnAvgsaTstar * (1 / len_Tstar) ** 2)

        # Compute correlation coefficients for IM=AvgSA
        # In case of IM=SA len_period_star becomes 1, thus, also works in this case
        rho_lnSaT_lnAvgsaTstar = np.zeros(len(self.periods))
        for i in range(len(self.periods)):
            for j in range(len_Tstar):
                rho = self._get_correlation(self.periods[i], self.Tstar[j])
                rho_lnSaT_lnAvgsaTstar[i] = rho_lnSaT_lnAvgsaTstar[i] + rho * sigma_lnSaTstar[j]

            rho_lnSaT_lnAvgsaTstar[i] = rho_lnSaT_lnAvgsaTstar[i] / (len_Tstar * sigma_lnAvgsaTstar)

        return mu_lnAvgsaTstar, sigma_lnAvgsaTstar, rho_lnSaT_lnAvgsaTstar

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
        index : int
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
                            rx = rjb * np.tan(np.radians(azimuth)) * np.cos(np.radians(azimuth) - np.arcsin(width * np.cos(np.radians(dip)) * np.cos(np.radians(azimuth)) / rjb))
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
            if ztor * np.tan(np.radians(dip)) <= rx <= ztor * np.tan(np.radians(dip)) + width * 1. / np.cos(np.radians(dip)):
                rrup1 = rx * np.sin(np.radians(dip)) + ztor * np.cos(np.radians(dip))
            if rx > ztor * np.tan(np.radians(dip)) + width * 1. / np.cos(np.radians(dip)):
                rrup1 = np.sqrt(np.square(rx - width * np.cos(np.radians(dip))) + np.square(ztor + width * np.sin(np.radians(dip))))
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

    @staticmethod
    @njit
    def _find_rec_greedy(sample_small, scaling_factors, mu_ln, sigma_ln, rec_id, sample_big, error_weights, max_scale_factor, num_records, penalty):
        """
        Details
        -------
        Greedy subset modification algorithm
        The method is defined separately so that njit can be used as wrapper and the routine can be run faster

        Parameters
        ----------
        sample_small : numpy.ndarray (2-D)
            Spectra of the reduced candidate record set (num_records - 1)
        scaling_factors : numpy.ndarray (1-D)
            Scaling factors for all records in the filtered database
        mu_ln : numpy.ndarray (1-D)
            Logarthmic mean of the target spectrum (conditional or unconditional)
        sigma_ln : numpy.ndarray (1-D)
            Logarthmic standard deviation of the target spectrum (conditional or unconditional)
        rec_id : numpy.ndarray (1-D)
            Record IDs of the reduced candidate set records in the database (num_records - 1)
        sample_big : numpy.ndarray (2-D)
            Spectra of the records in the filtered database
        error_weights : numpy.ndarray (1-D) or list 
            Weights for error in mean, standard deviation and skewness
        max_scale_factor : float
            The maximum allowable scale factor
        num_records : int
            Number of ground motions to be selected.
        penalty : int
            > 0 to penalize selected spectra more than 3 sigma from the target at any period, 0 otherwise.

        Returns
        -------
        min_id : int
            ID of the new selected record with the scale factor closest to 1
        """
        def mean_numba(arr):
            """
            Computes the mean of a 2-D array along axis=0.
            Required for computations since njit is used as wrapper.
            """

            res = []
            for i in range(arr.shape[1]):
                res.append(arr[:, i].mean())

            return np.array(res)

        def std_numba(arr):
            """
            Computes the standard deviation of a 2-D array along axis=0.
            Required for computations since njit is used as wrapper.
            """

            res = []
            for i in range(arr.shape[1]):
                res.append(arr[:, i].std())

            return np.array(res)

        min_dev = 100000
        for j in range(sample_big.shape[0]):
            # Add to the sample the scaled spectrum
            temp = np.zeros((1, len(sample_big[j, :])))
            temp[:, :] = sample_big[j, :]
            sample_small_trial = np.concatenate((sample_small, temp + np.log(scaling_factors[j])), axis=0)
            dev_mean = mean_numba(sample_small_trial) - mu_ln  # Compute deviations from target
            dev_sig = std_numba(sample_small_trial) - sigma_ln
            dev_total = error_weights[0] * np.sum(dev_mean * dev_mean) + error_weights[1] * np.sum(dev_sig * dev_sig)

            # Check if we exceed the scaling limit
            if scaling_factors[j] > max_scale_factor or scaling_factors[j] < 1 / max_scale_factor or np.any(rec_id == j):
                dev_total = dev_total + 1000000
            # Penalize bad spectra
            elif penalty > 0:
                for m in range(num_records):
                    dev_total = dev_total + np.sum(np.abs(np.exp(sample_small_trial[m, :]) > np.exp(mu_ln + 3.0 * sigma_ln))) * penalty
                    dev_total = dev_total + np.sum(np.abs(np.exp(sample_small_trial[m, :]) < np.exp(mu_ln - 3.0 * sigma_ln))) * penalty

            # Should cause improvement and record should not be repeated
            if dev_total < min_dev:
                min_id = j
                min_dev = dev_total

        return min_id

    def create(self, Tstar=None, gmpe='BooreEtAl2014', num_components=None, spectrum_definition='RotD50',
               site_param={'vs30': 520}, rup_param={'rake': [0.0, 45.0], 'mag': [7.2, 6.5]},
               dist_param={'rjb': [20, 5]}, hz_cont=[0.6, 0.4], period_range=[0.01, 4],
               im_Tstar=1.0, epsilon=None, use_variance=1, correlation_model='baker_jayaram'):
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
        in accordance with Kaklamanos et al. 2011 within ConditionalSpectrum._set_contexts method. They are not
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
        Tstar : int, float, numpy.ndarray
            Conditioning period or periods in case of AvgSa [sec].
            If None the target is an unconditional spectrum.
            The default is None.
        gmpe : str, optional
            GMPE model (see OpenQuake library).
            The default is 'BooreEtAl2014'.
        num_components : int, optional, the default is None.
            1 for single-component selection and arbitrary component sigma.
            2 for two-component selection and average component sigma.
            if None determined based on spectrum_definition
        spectrum_definition : str, optional
            The spectra definition of horizontal component, 'Arbitrary', 'GeoMean', 'RotD50', 'RotD100'.
            The default is 'RotD50'.
        site_param : dictionary
            Contains site parameters to define target spectrum.
            Dictionary keys (parameters) are not list type. Same parameters are used for each scenario.
            Some parameters are:
            'vs30': Average shear-wave velocity of the site
            'vs30measured': vs30 type, True (measured) or False (inferred)
            'z1pt0': Depth to Vs=1 km/sec from the site
            'z2pt5': Depth to Vs=2.5 km/sec from the site
            The default is {'vs30': 520}
        rup_param : dictionary
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
            The default is {'rake': [0.0, 45.0], 'mag': [7.2, 6.5]}
        dist_param : dictionary
            Contains distance parameters to define target spectrum.
            Dictionary keys (parameters) are list type. Each item in the list corresponds to a scenario.
            Some parameters are:
            'rjb': Closest distance to surface projection of coseismic rupture (km)
            'rrup': Closest distance to coseismic rupture (km)
            'repi': Epicentral distance (km)
            'rhypo': Hypocentral distance (km)
            'rx': Horizontal distance from top of rupture measured perpendicular to fault strike (km)
            'ry0': The horizontal distance off the end of the rupture measured parallel to strike (km)
            The default is {'rjb': [20, 5]}
        hz_cont : list, optional
            Hazard contribution for considered scenarios. 
            If None hazard contribution is the same for all scenarios.
            The default is None.
        im_Tstar : int, float, optional
            Conditioning intensity measure level [g] (conditional selection)
            the default is 1.
        epsilon : list, optional
            Epsilon values for considered scenarios (conditional selection)
            The default is None.
        period_range : list, optional
            Lower and upper bound values for the period range of target spectrum.
            The default is [0.01,4].
        use_variance : int, optional
            0 not to use variance in target spectrum
            1 to use variance in target spectrum
            The default is 1.
        correlation_model : str, optional
            correlation model to use "baker_jayaram","akkar"
            The default is baker_jayaram

        Returns
        -------
        None.                    
        """
        # TODO: gsim.get_mean_and_stddevs is deprecated, use ContextMaker.get_mean_stds in the future.
        # https://docs.openquake.org/oq-engine/advanced/developing.html#working-with-gmpes-directly-the-contextmaker

        # TODO:  Make the step size equal in period array. This will result in more realistic matching.
        # However, this is not essential. Moreover, this will require generation of new meta_data files
        if Tstar is None:
            # runing unconditional spectrum based record selection
            self.is_conditioned = 0

        else:
            # runing conditional-spectrum based record selection
            self.is_conditioned = 1

            # add Tstar to self
            if isinstance(Tstar, int) or isinstance(Tstar, float):
                self.Tstar = np.array([Tstar])
            elif isinstance(Tstar, numpy.ndarray):
                self.Tstar = Tstar

            # check if AvgSa or Sa is used as IM, then in case of Sa(T*) add T* and Sa(T*) if not present
            if not self.Tstar[0] in self.database['Periods'] and len(self.Tstar) == 1:
                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_1'], axis=1)
                sa_in = f(self.Tstar[0])
                sa_in.shape = (len(sa_in), 1)
                sa = np.append(self.database['Sa_1'], sa_in, axis=1)
                periods = np.append(self.database['Periods'], self.Tstar[0])
                self.database['Sa_1'] = sa[:, np.argsort(periods)]

                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_2'], axis=1)
                sa_in = f(self.Tstar[0])
                sa_in.shape = (len(sa_in), 1)
                sa = np.append(self.database['Sa_2'], sa_in, axis=1)
                self.database['Sa_2'] = sa[:, np.argsort(periods)]

                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_RotD50'], axis=1)
                sa_in = f(self.Tstar[0])
                sa_in.shape = (len(sa_in), 1)
                sa = np.append(self.database['Sa_RotD50'], sa_in, axis=1)
                self.database['Sa_RotD50'] = sa[:, np.argsort(periods)]

                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_RotD100'], axis=1)
                sa_in = f(self.Tstar[0])
                sa_in.shape = (len(sa_in), 1)
                sa = np.append(self.database['Sa_RotD100'], sa_in, axis=1)
                self.database['Sa_RotD100'] = sa[:, np.argsort(periods)]

                self.database['Periods'] = periods[np.argsort(periods)]

        try:  # this is smth like self.bgmpe = gsim.boore_2014.BooreEtAl2014()
            self.bgmpe = gsim.get_available_gsims()[gmpe]()
            self.gmpe = gmpe

        except KeyError:
            print(f'{gmpe} is not a valid gmpe name')
            raise
        
        # Keep num_components for now, the parameter will be required in the case of vertical spectrum
        if num_components is None:
            num_components = {'Arbitrary': 1, 'GeoMean': 2, 'RotD50': 2, 'RotD100': 2}[spectrum_definition]

        elif spectrum_definition in ['GeoMean', 'RotD50', 'RotD100'] and num_components < 2:
            num_components = 2
            print(f'Changing number of components to 2 for this horizontal spectra component definition: {spectrum_definition}...')

        elif spectrum_definition == 'Arbitrary' and num_components > 1:
            num_components = 1
            print(f'Changing number of components to 1 for this horizontal spectra component definition: {spectrum_definition}...')

        # add target spectrum settings to self
        self.num_components = num_components
        self.spectrum_definition = spectrum_definition
        self.use_variance = use_variance
        self.correlation_model = correlation_model
        self.site_param = site_param
        self.rup_param = rup_param
        self.dist_param = dist_param

        num_scenarios = len(rup_param['mag'])  # number of scenarios
        if hz_cont is None:  # equal for all
            self.hz_cont = [1 / num_scenarios for _ in range(num_scenarios)]
        else:
            self.hz_cont = hz_cont

        # Period range of the target spectrum
        temp = np.abs(self.database['Periods'] - np.min(period_range))
        idx1 = np.where(temp == np.min(temp))[0][0]
        temp = np.abs(self.database['Periods'] - np.max(period_range))
        idx2 = np.where(temp == np.min(temp))[0][0]
        self.periods = self.database['Periods'][idx1:idx2 + 1]

        # Hazard contribution of all rupture scenarios
        hz_cont_rups = np.matlib.repmat(np.asarray(self.hz_cont), len(self.periods), 1)
        # Conditional mean spectra (in logartihm) for all rupture scenarios
        mu_ln_rups = np.zeros((len(self.periods), num_scenarios))
        # Covariance matrices for all rupture scenarios
        cov_rups = np.zeros((num_scenarios, len(self.periods), len(self.periods)))

        for n in range(num_scenarios):

            # gmpe spectral values
            mu_lnSaT = np.zeros(len(self.periods))
            sigma_lnSaT = np.zeros(len(self.periods))

            # correlation coefficients
            rho_lnSaT_lnAvgsaTstar = np.zeros(len(self.periods))

            # Covariance
            cov = np.zeros((len(self.periods), len(self.periods)))

            # Set the contexts for the scenario
            sctx, rctx, dctx = self._set_contexts(n)

            for i in range(len(self.periods)):
                # Get the GMPE output for a rupture scenario
                params = self.bgmpe.get_mean_and_stddevs(sctx, rctx, dctx, imt.SA(period=self.periods[i]), [const.StdDev.TOTAL])
                mu_lnSaT[i] = params[0]
                sigma_lnSaT[i] = params[1][0]
                # modify spectral targets if RotD100 values were specified for two-component selection:
                if self.spectrum_definition == 'RotD100' and not 'RotD100' in self.bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT.name and self.num_components == 2:
                    rotd100_mu_ratio, rotd100_sigma = self._gmpe_sb_2014_ratios(self.periods[i])
                    mu_lnSaT[i] = mu_lnSaT[i] + np.log(rotd100_mu_ratio)
                    sigma_lnSaT[i] = (sigma_lnSaT[i] ** 2 + rotd100_sigma ** 2) ** 0.5

            if self.is_conditioned == 1:
                # Get the GMPE output and calculate for IM=AvgSA and associated dispersion
                mu_lnAvgsaTstar, sigma_lnAvgsaTstar, rho_lnSaT_lnAvgsaTstar = self._get_cond_param(sctx, rctx, dctx)

                if epsilon is None:
                    # Back calculate epsilon
                    epsilon_rup = (np.log(im_Tstar) - mu_lnAvgsaTstar) / sigma_lnAvgsaTstar
                else:
                    epsilon_rup = epsilon[n]

                # Get the value of the ln(CMS), conditioned on T_star
                mu_ln_rups[:, n] = mu_lnSaT + rho_lnSaT_lnAvgsaTstar * epsilon_rup * sigma_lnSaT

            elif self.is_conditioned == 0:
                mu_ln_rups[:, n] = mu_lnSaT

            for i in range(len(self.periods)):
                for j in range(len(self.periods)):

                    var1 = sigma_lnSaT[i] ** 2
                    var2 = sigma_lnSaT[j] ** 2

                    rho = self._get_correlation(self.periods[i], self.periods[j])
                    sigma_corr = rho * np.sqrt(var1 * var2)

                    if self.is_conditioned == 1:
                        varTstar = sigma_lnAvgsaTstar ** 2
                        sigma11 = np.matrix([[var1, sigma_corr], [sigma_corr, var2]])
                        sigma22 = np.array([varTstar])
                        sigma12 = np.array([rho_lnSaT_lnAvgsaTstar[i] * np.sqrt(var1 * varTstar), rho_lnSaT_lnAvgsaTstar[j] * np.sqrt(varTstar * var2)])
                        sigma12.shape = (2, 1)
                        sigma22.shape = (1, 1)
                        sigma_cond = sigma11 - sigma12 * 1. / sigma22 * sigma12.T
                        cov[i, j] = sigma_cond[0, 1]

                    elif self.is_conditioned == 0:
                        cov[i, j] = sigma_corr

            # Get the value of standard deviation of target spectrum
            cov_rups[n, :, :] = cov

        # over-write covariance matrix with zeros if no variance is desired in the ground motion selection
        if self.use_variance == 0:
            cov_rups = np.zeros(cov_rups.shape)

        mu_ln_target = np.sum(mu_ln_rups * hz_cont_rups, 1)
        # all 2D matrices are the same for each kk scenario, since sigma is only T dependent
        cov_target = cov_rups[0, :, :]
        cov_elms = np.zeros((len(self.periods), num_scenarios))
        for ii in range(len(self.periods)):
            for kk in range(num_scenarios):
                # Hcont[kk] is hazard contribution of the k-th scenario
                cov_elms[ii, kk] = (cov_rups[kk, ii, ii] + (mu_ln_rups[ii, kk] - mu_ln_target[ii]) ** 2) * self.hz_cont[kk]

        # Compute the final covariance matrix
        cov_diag = np.sum(cov_elms, 1)
        cov_target[np.eye(len(self.periods)) == 1] = cov_diag
        # Avoid positive semi-definite covariance matrix with several eigenvalues being exactly zero.
        # See: https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning!
        min_eig = np.min(np.real(np.linalg.eigvals(cov_target)))
        if min_eig < 0:
            cov_target -= 2 * min_eig * np.eye(*cov_target.shape)
        sigma_ln_target = np.sqrt(np.diagonal(cov_target))

        # Add target spectrum to self
        self.mu_ln = mu_ln_target
        self.sigma_ln = sigma_ln_target
        self.cov = cov_target

        if self.is_conditioned == 1:
            # add intensity measure level to self
            if epsilon is None:
                self.im_Tstar = im_Tstar
            else:
                f = interpolate.interp1d(self.periods, np.exp(self.mu_ln))
                sa_in = f(self.Tstar)
                self.im_Tstar = np.exp(np.sum(np.log(sa_in)) / len(self.Tstar))
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
        if self.seed_value:
            np.random.seed(self.seed_value)
        else:
            np.random.seed(sum(gmtime()[:6]))

        dev_total_sim = np.zeros((self.num_simulations, 1))
        spectra = {}
        # Generate simulated response spectra with best matches to the target values
        for j in range(self.num_simulations):
            # It might be better to use the second function if cov_rank = np.linalg.matrix_rank(self.cov) < len(mu_ln)
            spectra[j] = np.exp(random_multivariate_normal(self.mu_ln, self.cov, self.num_records, 'LHS'))
            # specDict[j] = np.exp(np.random.multivariate_normal(self.mu_ln, self.cov, size=self.num_records))

            # how close is the mean of the spectra to the target
            dev_mean_sim = np.mean(np.log(spectra[j]), axis=0) - self.mu_ln
            # how close is the mean of the spectra to the target
            dev_sig_sim = np.std(np.log(spectra[j]), axis=0) - self.sigma_ln
            # how close is the skewness of the spectra to zero (i.e., the target)  
            dev_skew_sim = skew(np.log(spectra[j]), axis=0)
            # combine the three error metrics to compute a total error
            dev_total_sim[j] = self.error_weights[0] * np.sum(dev_mean_sim ** 2) + self.error_weights[1] * np.sum(dev_sig_sim ** 2) + 0.1 * (self.error_weights[2]) * np.sum(dev_skew_sim ** 2)

        recUse = np.argmin(np.abs(dev_total_sim))  # find the simulated spectra that best match the targets
        self.sim_spec = np.log(spectra[recUse])  # return the best set of simulations

    def select(self, num_records=30, is_scaled=1, max_scale_factor=4, mag_limits=None, vs30_limits=None, rjb_limits=None,
               mech_limits=None, num_simulations=20, seed_value=None, error_weights=[1, 2, 0.3], num_greedy_loops=2, penalty=0, 
               tolerance=10):
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
        num_records : int, optional
            Number of ground motions to be selected.
            The default is 30.
        max_scale_factor : float, optional
            The maximum allowable scale factor
            If None use of amplitude scaling for spectral matching is not allowed.
            If other than None use of amplitude scaling for spectral matching is allowed.
            The default is 4.
        is_scaled : int, optional
            If 1 use of amplitude scaling for spectral matching is not allowed.
            If 0 None use of amplitude scaling for spectral matching is allowed.
            The default is 1.        
        mag_limits : list, optional
            The limiting values on magnitude.
            The default is None.
        vs30_limits : list, optional
            The limiting values on Vs30. 
            The default is None.
        rjb_limits : list, optional
            The limiting values on Rjb.
            The default is None.
        mech_limits : list, optional
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
            The default is None.
        num_simulations : int, optional
            num_simulations sets of response spectra are simulated and the best set (in terms of
            matching means, variances and skewness is chosen as the seed). The user
            can also optionally rerun this segment multiple times before deciding to
            proceed with the rest of the algorithm. It is to be noted, however, that
            the greedy improvement technique significantly improves the match between
            the means and the variances subsequently.
            The default is 20.
        seed_value : int, optional
            For repeatability. For a particular seed value not equal to
            zero, the code will output the same set of ground motions.
            The set will change when the seed value changes. If set to
            zero, the code randomizes the algorithm and different sets of
            ground motions (satisfying the target mean and variance) are
            generated each time.
            The default is None.
        error_weights : numpy.ndarray or list, optional
            Weights for error in mean, standard deviation and skewness
            the default is [1, 2, 0.3].
        num_greedy_loops : int, optional
            Number of loops of optimization to perform.
            The default is 2.
        penalty : int, optional
            > 0 to penalize selected spectra more than 3 sigma from the target at any period, 
            0 otherwise.
            The default is 0.
        tolerance : int, optional
            Tolerable percent error to skip optimization
            The default is 10.

        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.num_records = num_records
        self.mag_limits = mag_limits
        self.vs30_limits = vs30_limits
        self.rjb_limits = rjb_limits
        self.mech_limits = mech_limits
        self.seed_value = seed_value
        self.error_weights = error_weights
        self.num_simulations = num_simulations
        self.max_scale_factor = max_scale_factor
        self.is_scaled = is_scaled
        self.num_greedy_loops = num_greedy_loops
        self.tolerance = tolerance
        self.penalty = penalty

        # Simulate response spectra
        self._simulate_spectra()

        # Search the database and filter
        sample_big, vs30, mag, rjb, mechanism, filename1, filename2, rsn, eq_id, station_code = self._search_database()

        # Processing available spectra
        sample_big = np.log(sample_big)
        len_big = sample_big.shape[0]

        # Find best matches to the simulated spectra from ground-motion database
        rec_id = np.ones(self.num_records, dtype=int) * (-1)
        final_scale_factors = np.ones(self.num_records)
        sample_small = np.ones((self.num_records, sample_big.shape[1]))
        error_weights = np.array(error_weights)

        # Check if max_scale_factor is None
        if max_scale_factor is None:
            max_scale_factor = 10 # use the default value

        if self.is_conditioned == 1 and self.is_scaled == 1:
            # Calculate IMLs for the sample
            f = interpolate.interp1d(self.periods, np.exp(sample_big), axis=1)
            sample_big_imls = np.exp(np.sum(np.log(f(self.Tstar)), axis=1) / len(self.Tstar))

        if self.is_conditioned == 1 and len(self.Tstar) == 1:
            # These indices are required in case IM = Sa(T) to break the loop
            ind2 = (np.where(self.periods != self.Tstar[0])[0][0]).tolist()

        # Find num_records ground motions, initial subset
        for i in range(self.num_records):
            error = np.zeros(len_big)
            scaling_factors = np.ones(len_big)

            # Calculate the scaling factor
            if self.is_scaled == 1:
                # using conditioning IML
                if self.is_conditioned == 1:
                    scaling_factors = self.im_Tstar / sample_big_imls
                # using error minimization
                elif self.is_conditioned == 0:
                    scaling_factors = np.sum(np.exp(sample_big) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(np.exp(sample_big) ** 2, axis=1)

            else:
                scaling_factors = np.ones(len_big)

            # check if enough records are found
            mask = (1 / max_scale_factor < scaling_factors) * (scaling_factors < max_scale_factor)
            idxs = np.where(mask)[0]
            error[~mask] = 1000000
            error[mask] = np.sum((np.log(np.exp(sample_big[idxs, :]) * scaling_factors[mask].reshape(len(scaling_factors[mask]), 1)) -self.sim_spec[i, :]) ** 2, axis=1)
            rec_id[i] = int(np.argsort(error)[0])
            if error.min() >= 1000000:
                raise Warning('Possible problem with simulated spectrum. No good matches found')

            if self.is_scaled == 1:
                final_scale_factors[i] = scaling_factors[rec_id[i]]

            # Save the selected spectra
            sample_small[i, :] = np.log(np.exp(sample_big[rec_id[i], :]) * final_scale_factors[i])

        # Apply greedy subset modification procedure
        for _ in range(self.num_greedy_loops):  # Number of passes

            for i in range(self.num_records):  # Loop for num_records
                sample_small = np.delete(sample_small, i, 0)
                rec_id = np.delete(rec_id, i)

                # Calculate the scaling factor
                if self.is_scaled == 1:
                    # using conditioning IML
                    if self.is_conditioned == 1:
                        scaling_factors = self.im_Tstar / sample_big_imls
                    # using error minimization
                    elif self.is_conditioned == 0:
                        scaling_factors = np.sum(np.exp(sample_big) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(np.exp(sample_big) ** 2, axis=1)
                else:
                    scaling_factors = np.ones(len_big)

                # Try to add a new spectrum to the subset list
                min_id = self._find_rec_greedy(sample_small, scaling_factors, self.mu_ln, self.sigma_ln, rec_id, sample_big, error_weights, max_scale_factor, num_records, penalty)

                # Add new element in the right slot
                if self.is_scaled == 1:
                    final_scale_factors[i] = scaling_factors[min_id]
                else:
                    final_scale_factors[i] = 1
                sample_small = np.concatenate((sample_small[:i, :], sample_big[min_id, :].reshape(1, sample_big.shape[1]) + np.log(scaling_factors[min_id]), sample_small[i:, :]), axis=0)
                rec_id = np.concatenate((rec_id[:i], np.array([min_id]), rec_id[i:]))

            # Lets check if the selected ground motions are good enough, if the errors are sufficiently small stop!
            if self.is_conditioned == 1 and len(self.Tstar) == 1:  # if conditioned on SaT, ignore error at T*
                median_error = np.max(np.abs(np.exp(np.mean(sample_small[:, ind2], axis=0)) - np.exp(self.mu_ln[ind2])) / np.exp(self.mu_ln[ind2])) * 100
                std_error = np.max(np.abs(np.std(sample_small[:, ind2], axis=0) - self.sigma_ln[ind2]) / self.sigma_ln[ind2]) * 100
            else:
                median_error = np.max(np.abs(np.exp(np.mean(sample_small, axis=0)) - np.exp(self.mu_ln)) / np.exp(self.mu_ln)) * 100
                std_error = np.max(np.abs(np.std(sample_small, axis=0) - self.sigma_ln) / self.sigma_ln) * 100

            if median_error < self.tolerance and std_error < self.tolerance:
                break

        print('Ground motion selection is finished.')
        print(f'For T ∈ [{self.periods[0]:.2f} - {self.periods[-1]:.2f}]')
        print(f'Max error in median = {median_error:.2f} %')
        print(f'Max error in standard deviation = {std_error:.2f} %')
        if median_error < self.tolerance and std_error < self.tolerance:
            print(f'The errors are within the target {self.tolerance:d} percent %')

        rec_id = rec_id.tolist()
        # Add selected record information to self
        self.rec_scale_factors = final_scale_factors
        self.rec_sa_ln = sample_small
        self.rec_vs30 = vs30[rec_id]
        self.rec_rjb = rjb[rec_id]
        self.rec_mag = mag[rec_id]
        self.rec_mech = mechanism[rec_id]
        self.rec_eq_id = eq_id[rec_id]
        self.rec_file_h1 = filename1[rec_id]

        if self.num_components == 2:
            self.rec_file_h2 = filename2[rec_id]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = rsn[rec_id]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[rec_id]


class CodeSpectrum(_SubClass_):
    """
    This class is used for
        1) Creating target spectrum based on various codes (TBEC 2018, ASCE 7-16, EC8-Part1)
        2) Selecting and scaling suitable ground motion sets for target spectrum in accordance with specified code
    """

    def __init__(self, database='NGA_W2', output_directory='Outputs', target_path=None, num_records=11, num_components=1, selection_algorithm=1,
                 mag_limits=None, vs30_limits=None, rjb_limits=None, mech_limits=None, max_scale_factor=2, max_rec_per_event=3, obj_path=None):
        """
        Details
        -------
        Loads the record database to use, creates output folder, sets selection criteria.

        Parameters
        ----------
        database : str, optional
            Database to use: NGA_W2, ESM_2018
            The default is NGA_W2.
        output_directory : str, optional
            Output directory
            The default is 'Outputs'
        target_path : str, optional, the default is None.
            Path for used defined target spectrum.
        num_records : int, optional, the default is 11.
            Number of records to be selected. 
        num_components : int, optional, the default is 1.
            Number of ground motion components to select.
        selection_algorithm : int, optional, the default is 1.
            If equal to 1, the record set is selected using
            method of “least squares”, each record has individual scaling factor.
            If equal to 2, the record set selected such that each record has
            identical scale factor which is as close as possible to 1.
        mag_limits : list, optional, the default is None.
            The limiting values on magnitude. 
        vs30_limits : list, optional, the default is None.
            The limiting values on Vs30. 
        rjb_limits : list, optional, the default is None.
            The limiting values on Rjb. 
        mech_limits : list, optional, the default is None.
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
        max_scale_factor : float, optional, the default is 2.
            Maximum allowed scaling factor, used with opt=2 case.
        max_rec_per_event : int, the default is 3.
            The limit for the maximum number of records belong to the same event
        obj_path : str, optional, the default is None.
            This is the path to the previously saved obj.pkl file by EzGM.
            One can use the previously saved instance and use rest of the methods.

        Returns
        -------
        None.
        """

        # Inherit the subclass
        super().__init__()

        # Read the old EzGM obj
        if obj_path:
            with open('obj.pkl', 'rb') as file:
                obj = pickle.load(file)
            self.__dict__.update(obj)
            database = self.database

        # Add new selection settings to self
        else:
            self.num_records = num_records
            self.num_components = num_components
            self.mag_limits = mag_limits
            self.vs30_limits = vs30_limits
            self.rjb_limits = rjb_limits
            self.mech_limits = mech_limits
            self.selection_algorithm = selection_algorithm
            self.max_scale_factor = max_scale_factor
            self.target_path = target_path
            self.max_rec_per_event = max_rec_per_event

        # Add the input the ground motion database to use
        matfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Meta_Data', database)
        self.database = loadmat(matfile, squeeze_me=True)
        if 'Name' in self.database:
            pass
        else:
            self.database['Name'] = database

        # create the output directory and add the path to self
        cwd = os.getcwd()
        outdir_path = os.path.join(cwd, output_directory)
        make_dir(outdir_path)
        self.output_directory_path = outdir_path

    @staticmethod
    @njit
    def _find_rec_smallest_sf(sample_small, scale_factors_small, target_spectrum, rec_ids_small, eq_ids_small, min_id, eq_ids_big, sample_big, max_rec_per_event):
        """
        Details
        -------
        Greedy subset modification to obtain set of records the scaling factors closest to 1
        The method is defined separately so that njit can be used as wrapper and the routine can be run faster

        Parameters
        ----------
        sample_small : numpy.ndarray (2-D)
            Spectra of the reduced candidate record set (num_records - 1)
        scale_factors_small : numpy.ndarray (2-D)
            Scale factors for the reduced candidate record set (num_records - 1)
        target_spectrum : numpy.ndarray (1-D)
            Target spectrum (elastic spectrum from the code)
        rec_ids_small : numpy.ndarray (1-D)
            Record IDs of the reduced candidate set records in the database
        eq_ids_small : numpy.ndarray (1-D)
            Event IDs of the reduced candidate set records in the database
        min_id : int
            ID of the eliminated record from candidate record set
        eq_ids_big : numpy.ndarray (2-D)
            Event IDs of the records in the filtered database
        sample_big : numpy.ndarray (2-D)
            Spectra of the records in the filtered database
        max_rec_per_event : int
            The limit for the maximum number of records belong to the same event

        Returns
        -------
        min_id : int
            ID of the new selected record with the scale factor closest to 1
        scale_factors_small : numpy.ndarray (2-D)
            Scale factors for the candidate record set (num_records) with the addition of new record
        """

        def mean_numba(a):
            """
            Computes the mean of a 2-D array along axis=0.
            Required for computations since njit is used as wrapper.
            """

            res = []
            for i in range(a.shape[1]):
                res.append(a[:, i].mean())

            return np.array(res)

        for j in range(sample_big.shape[0]):
            tmp = eq_ids_big[j]
            # record should not be repeated and number of eqs from the same event should not exceed 3
            if not np.any(rec_ids_small == j) and np.sum(eq_ids_small == tmp) < max_rec_per_event:
                # Add to the sample the scaled spectra
                temp = np.zeros((1, len(sample_big[j, :])))
                temp[:, :] = sample_big[j, :]  # get the trial spectra
                sample_small_trial = np.concatenate((sample_small, temp), axis=0)  # add the trial spectra to subset list
                temp_scale = np.max(target_spectrum / mean_numba(sample_small_trial))  # compute new scaling factor

                # Should cause improvement
                if abs(temp_scale - 1) <= abs(scale_factors_small - 1):
                    min_id = j
                    scale_factors_small = temp_scale

        return min_id, scale_factors_small

    def select_tbec2018(self, lat=41.0582, long=29.00951, dd_level=2, site_class='ZC', predominant_period=1):
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
            For num_components = 1: Sa_rec = (Sa_1 or Sa_2) - lower bound = 1.0 * SaTarget(0.2Tp-1.5Tp) 
            For num_components = 2: Sa_rec = (Sa_1**2+Sa_2**2)**0.5 - lower bound = 1.3 * SaTarget(0.2Tp-1.5Tp) 

        Rule 2: 
            No more than 3 records can be selected from the same event! In other words,
            rec_eq_id cannot be the same for more than 3 of the selected records.      

        Rule 3: 
            At least 11 records (or pairs) must be selected.

        Parameters
        ----------
        lat: float, optional, the default is 41.0582.
            Site latitude
        long: float, optional, the default is 29.00951.
            Site longitude
        dd_level:  int, optional, the default is 2.
            Earthquake ground motion intensity level (1,2,3,4)
        site_class: str, optional, the default is 'ZC'.
            Site soil class ('ZA','ZB','ZC','ZD','ZE')
        predominant_period : float, optional, the default is 1.
            Predominant period of the structure. 
        
        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.lat = lat
        self.long = long
        self.dd_level = dd_level
        self.site_class = site_class
        self.predominant_period = predominant_period
        self.code = 'TBEC 2018'

        if self.num_records < 11:
            print('Warning! Number of requested records must be at least 11 according to TBEC 2018. Changing...')
            self.num_records = 11

        if self.max_rec_per_event > 3:
            print('Warning! Limit for record per event must be at most 3 according to TBEC 2018. Changing...')
            self.max_rec_per_event = 3

        # Set the period range
        self.lower_bound_period = 0.2 * predominant_period
        self.upper_bound_period = 1.5 * predominant_period

        # Match periods (periods for error computations)
        self.periods = self.database['Periods']

        # Determine the elastic design spectrum from the user-defined spectrum
        if self.target_path:
            data = np.loadtxt(self.target_path)
            interp_func = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            target_spectrum = interp_func(self.periods)

        # Determine the elastic design spectrum from code
        else:
            _, sds, sd1, tl = site_parameters_tbec2018(lat, long, dd_level, site_class)
            target_spectrum, _ = sae_tbec2018(self.periods, sds, sd1, tl)

        # Consider the lower bound spectrum specified by the code as target spectrum
        if self.num_components == 1:
            target_spectrum = 1.0 * target_spectrum
        elif self.num_components == 2:
            target_spectrum = 1.3 * target_spectrum
            self.spectrum_definition = 'SRSS'

        # Search the database and filter
        sample_big, vs30, mag, rjb, mechanism, filename1, filename2, rsn, eq_ids, station_code = self._search_database()

        # Sample size of the filtered database
        len_big = sample_big.shape[0]

        # Scale factors based on mse
        scale_factors_big = np.array(np.sum(np.matlib.repmat(target_spectrum, len_big, 1) * sample_big, axis=1) / np.sum(sample_big ** 2, axis=1))

        # Find best matches to the target spectrum from ground-motion database
        temp = (np.matlib.repmat(target_spectrum, len_big, 1) - sample_big) ** 2
        mse = temp.mean(axis=1)

        if self.database['Name'].startswith('ESM'):
            d = {ni: indi for indi, ni in enumerate(set(eq_ids.tolist()))}
            eq_ids_big = np.asarray([d[ni] for ni in eq_ids.tolist()])
        else:
            eq_ids_big = eq_ids.copy()

        rec_id_sorted = np.argsort(mse)
        rec_ids_small = np.ones(self.num_records, dtype=int) * (-1)
        eq_ids_small = np.ones(self.num_records, dtype=int) * (-1)
        idx1 = 0
        idx2 = 0
        while idx1 < self.num_records:  # not more than 3 of the records should be from the same event
            tmp1 = rec_id_sorted[idx2]
            idx2 += 1
            tmp2 = eq_ids_big[tmp1]
            rec_ids_small[idx1] = tmp1
            eq_ids_small[idx1] = tmp2
            if np.sum(eq_ids_small == tmp2) <= self.max_rec_per_event:
                idx1 += 1

        # Initial selection results - based on MSE
        scale_factors_small = scale_factors_big[rec_ids_small]
        sample_small = sample_big[rec_ids_small, :]

        # Must not be lower than target within the period range, find the indicies for this period range
        idxs = np.where((self.database['Periods'] >= self.lower_bound_period) * (self.database['Periods'] <= self.upper_bound_period))[0]

        if self.selection_algorithm == 1:
            self.rec_scale_factors = scale_factors_small * np.max(target_spectrum[idxs] / (scale_factors_small.reshape(-1, 1) * sample_small[:, idxs]).mean(axis=0))

        # try to optimize scaling factor to make it closest as possible to 1
        if self.selection_algorithm == 2:
            scale_factors_small = np.max(target_spectrum[idxs] / sample_small[:, idxs].mean(axis=0))
            for i in range(self.num_records):  # Loop for num_records
                # note the ID of the record which is removed
                min_id = rec_ids_small[i]
                # remove the i'th record search for a candidate
                sample_small = np.delete(sample_small, i, 0)
                rec_ids_small = np.delete(rec_ids_small, i)
                eq_ids_small = np.delete(eq_ids_small, i)
                # Try to add a new spectra to the subset list and consider critical periods for error calculations only (idxs)
                min_id, scale_factors_small = self._find_rec_smallest_sf(sample_small[:, idxs], scale_factors_small, target_spectrum[idxs], rec_ids_small, eq_ids_small, min_id, eq_ids_big, sample_big[:, idxs], self.max_rec_per_event)
                # Add new element in the right slot
                sample_small = np.concatenate((sample_small[:i, :], sample_big[min_id, :].reshape(1, sample_big.shape[1]), sample_small[i:, :]), axis=0)
                rec_ids_small = np.concatenate((rec_ids_small[:i], np.array([min_id]), rec_ids_small[i:]))
                eq_ids_small = np.concatenate((eq_ids_small[:i], np.array([eq_ids_big[min_id]]), eq_ids_small[i:]))
            self.rec_scale_factors = np.ones(self.num_records) * float(scale_factors_small)

        # check the scaling
        if np.any(self.rec_scale_factors > self.max_scale_factor) or np.any(self.rec_scale_factors< 1 / self.max_scale_factor):
            raise ValueError('Scaling factor criteria is not satisfied',
                             'Please broaden your selection and scaling criteria or change the optimization scheme...')

        rec_ids_small = rec_ids_small.tolist()
        # Add selected record information to self
        self.rec_vs30 = vs30[rec_ids_small]
        self.rec_rjb = rjb[rec_ids_small]
        self.rec_mag = mag[rec_ids_small]
        self.rec_mech = mechanism[rec_ids_small]
        self.rec_eq_id = eq_ids[rec_ids_small]
        self.rec_file_h1 = filename1[rec_ids_small]

        if self.num_components == 2:
            self.rec_file_h2 = filename2[rec_ids_small]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = rsn[rec_ids_small]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[rec_ids_small]

        rec_idxs = []
        if self.num_components == 1:
            sa_known = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            filename1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_file_h1:
                rec_idxs.append(np.where(filename1 == rec)[0][0])
            rec_spec = sa_known[rec_idxs, :]
        elif self.num_components == 2:
            for rec in self.rec_file_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            sa1 = self.database['Sa_1'][rec_idxs, :]
            sa2 = self.database['Sa_2'][rec_idxs, :]
            rec_spec = (sa1 ** 2 + sa2 ** 2) ** 0.5

        # Save the results for whole spectral range
        self.rec_sa_ln = rec_spec
        self.periods = self.database['Periods']

        if self.target_path:
            self.target = interp_func(self.periods)
        else:
            self.target, _ = sae_tbec2018(self.periods, sds, sd1, tl)

        print('TBEC 2018 based ground motion record selection and amplitude scaling are finished...')

    def select_asce7_16(self, lat=34, long=-118, risk_cat='II', site_class='C', fundamental_periods = [1, 1], lower_bound_period=None, upper_bound_period=None):
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
            For num_components = 1: Sa_rec = (Sa1 or Sa2) - lower bound = 0.9 * Sa_MCEr(Tlower-Tupper)
            For num_components = 2: Sa_rec = RotD100 - lower bound = 0.9 * Sa_MCEr(Tlower-Tupper)
            lower_bound_period >= 0.2 * min(fundamental_periods)
            upper_bound_period >= 1.5 * max(fundamental_periods)
            
        Rule 2: 
            At least 11 records (or pairs) must be selected.

        Parameters
        ----------
        lat : float, optional, the default is 41.0582.
            Site latitude
        long : float, optional, the default is 29.00951.
            Site longitude
        risk_category :  str, the default is 'III'
            Risk category for structure ('I','II','III','IV')
        site_class : str, optional, the default is 'C'.
            Site soil class ('A','B','C','D','E')
        fundamental_periods : list, the default is [1, 1].
            The first-mode periods in principal horizontal directions     
        lower_bound_period: float, the default is None.
            The lower bound for matching period range, if None equal to 0.2*min(fundamental_periods)
        upper_bound_period: float, the default is None.
            The upper bound for matching period range, if None equal to 2.0*max(fundamental_periods)
        
        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.lat = lat
        self.long = long
        self.risk_category = risk_cat
        self.site_class = site_class
        self.code = 'ASCE 7-16'

        # Section 16.2.3.1
        if not lower_bound_period:
            lower_bound_period = 0.2 * min(fundamental_periods)
        elif lower_bound_period < 0.2 * min(fundamental_periods):
            lower_bound_period = 0.2 * min(fundamental_periods)
            print('Warning! Lower bound cannot be lower than 0.2 times the largest first-mode period according to '
                  'ASCE 7-16. Changing...')
        if not upper_bound_period:
            upper_bound_period = 2.0 * max(fundamental_periods)
        elif upper_bound_period < 1.5 * max(fundamental_periods):
            upper_bound_period = 1.5 * max(fundamental_periods)
            print('Warning! Upper bound cannot be lower than 1.5 times the smallest first-mode period according to '
                  'ASCE 7-16. Changing...')
        self.lower_bound_period = lower_bound_period
        self.upper_bound_period = upper_bound_period

        # Section 16.2.2
        if self.num_records < 11:
            print('Warning! Number of records must be at least 11 according to ASCE 7-16. Changing...')
            self.num_records = 11

        # Match periods (periods for error computations)
        self.periods = self.database['Periods']

        # Determine the elastic design spectrum from the user-defined spectrum
        if self.target_path:
            data = np.loadtxt(self.target_path)
            interp_func = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            target_spectrum = interp_func(self.periods)

        # Determine the elastic design spectrum from code, Section 16.2.1
        else:
            sds, sd1, tl = site_parameters_asce7_16(lat, long, risk_cat, site_class)  # Retrieve site parameters
            target_spectrum = 1.5 * sae_asce7_16(self.periods, sds, sd1, tl)  # Retrive the design spectrum and multiply by 1.5 to get MCER

        # Consider the lower bound spectrum specified by the code as target spectrum, Section 16.2.3.2
        target_spectrum = 0.9 * target_spectrum
        if self.num_components == 2:
            self.spectrum_definition = 'RotD100'

            # Search the database and filter
        sample_big, vs30, mag, rjb, mechanism, filename1, filename2, rsn, eq_ids, station_code = self._search_database()

        # Sample size of the filtered database
        len_big = sample_big.shape[0]

        # Scale factors based on mse
        scale_factors_big = np.array(np.sum(np.matlib.repmat(target_spectrum, len_big, 1) * sample_big, axis=1) / np.sum(sample_big ** 2, axis=1))

        # Find best matches to the target spectrum from ground-motion database
        temp = (np.matlib.repmat(target_spectrum, len_big, 1) - sample_big) ** 2
        mse = temp.mean(axis=1)

        if self.database['Name'].startswith('ESM'):
            d = {ni: indi for indi, ni in enumerate(set(eq_ids.tolist()))}
            eq_ids_big = np.asarray([d[ni] for ni in eq_ids.tolist()])
        else:
            eq_ids_big = eq_ids.copy()

        rec_id_sorted = np.argsort(mse)
        rec_ids_small = np.ones(self.num_records, dtype=int) * (-1)
        eq_ids_small = np.ones(self.num_records, dtype=int) * (-1)
        idx1 = 0
        idx2 = 0
        while idx1 < self.num_records:  # not more than 3 of the records should be from the same event
            tmp1 = rec_id_sorted[idx2]
            idx2 += 1
            tmp2 = eq_ids_big[tmp1]
            rec_ids_small[idx1] = tmp1
            eq_ids_small[idx1] = tmp2
            if np.sum(eq_ids_small == tmp2) <= self.max_rec_per_event:
                idx1 += 1

        # Initial selection results - based on MSE
        scale_factors_small = scale_factors_big[rec_ids_small]
        sample_small = sample_big[rec_ids_small, :]

        # Must not be lower than target within the period range, find the indicies for this period range
        idxs = np.where((self.database['Periods'] >= self.lower_bound_period) * (self.database['Periods'] <= self.upper_bound_period))[0]

        if self.selection_algorithm == 1:
            self.rec_scale_factors = scale_factors_small * np.max(target_spectrum[idxs] / (scale_factors_small.reshape(-1, 1) * sample_small[:, idxs]).mean(axis=0))

        # try to optimize scaling factor to make it closest as possible to 1
        if self.selection_algorithm == 2:
            scale_factors_small = np.max(target_spectrum[idxs] / sample_small[:, idxs].mean(axis=0))
            for i in range(self.num_records):  # Loop for num_records
                # note the ID of the record which is removed
                min_id = rec_ids_small[i]
                # remove the i'th record search for a candidate 
                sample_small = np.delete(sample_small, i, 0)
                rec_ids_small = np.delete(rec_ids_small, i)
                eq_ids_small = np.delete(eq_ids_small, i)
                # Try to add a new spectra to the subset list and consider critical periods for error calculations only (idxs)
                min_id, scale_factors_small = self._find_rec_smallest_sf(sample_small[:, idxs], scale_factors_small, target_spectrum[idxs], rec_ids_small, eq_ids_small, min_id, eq_ids_big, sample_big[:, idxs], self.max_rec_per_event)
                # Add new element in the right slot
                sample_small = np.concatenate((sample_small[:i, :], sample_big[min_id, :].reshape(1, sample_big.shape[1]), sample_small[i:, :]), axis=0)
                rec_ids_small = np.concatenate((rec_ids_small[:i], np.array([min_id]), rec_ids_small[i:]))
                eq_ids_small = np.concatenate((eq_ids_small[:i], np.array([eq_ids_big[min_id]]), eq_ids_small[i:]))
            self.rec_scale_factors = np.ones(self.num_records) * float(scale_factors_small)

        # check the scaling
        if np.any(self.rec_scale_factors > self.max_scale_factor) or np.any(self.rec_scale_factors < 1 / self.max_scale_factor):
            raise ValueError('Scaling factor criteria is not satisfied',
                             'Please broaden your selection and scaling criteria or change the optimization scheme...')
        rec_ids_small = rec_ids_small.tolist()
        # Add selected record information to self
        self.rec_vs30 = vs30[rec_ids_small]
        self.rec_rjb = rjb[rec_ids_small]
        self.rec_mag = mag[rec_ids_small]
        self.rec_mech = mechanism[rec_ids_small]
        self.rec_eq_id = eq_ids[rec_ids_small]
        self.rec_file_h1 = filename1[rec_ids_small]

        if self.num_components == 2:
            self.rec_file_h2 = filename2[rec_ids_small]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = rsn[rec_ids_small]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[rec_ids_small]

        rec_idxs = []
        if self.num_components == 1:
            sa_known = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            filename1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_file_h1:
                rec_idxs.append(np.where(filename1 == rec)[0][0])
            rec_spec = sa_known[rec_idxs, :]
        elif self.num_components == 2:
            for rec in self.rec_file_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            rec_spec = self.database['Sa_RotD100'][rec_idxs, :]

        # Save the results for whole spectral range
        self.rec_sa_ln = rec_spec
        self.periods = self.database['Periods']

        if self.target_path:
            self.target = interp_func(self.periods)
        else:
            self.target = 1.5 * sae_asce7_16(self.periods, sds, sd1, tl)

        print('ASCE 7-16 based ground motion record selection and amplitude scaling are finished...')

    def select_ec8_part1(self, ag=0.2, xi=0.05, importance_class='II', target_type='Type1', site_class='C', predominant_period=1):
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
            For num_components = 1: Sa_rec = (Sa_1 or Sa_2) - lower bound = 0.9 * SaTarget(0.2Tp-2.0Tp)
            For num_components = 2: Sa_rec = (Sa_1 + Sa_2) * 0.5 - lower bound = 0.9 * SaTarget(0.2Tp-2.0Tp)

        Parameters
        ----------
        ag:  float, optional, the default is 0.25.
            Peak ground acceleration [g]
        xi: float, optional, the default is 0.05.
            Damping
        importance_class: str, the default is 'II'.
            Importance class ('I','II','III','IV')
        target_type: str, optional, the default is 'Type1'
            Type of spectrum (Option: 'Type1' or 'Type2')
        site_class: str, optional, the default is 'B'
            Soil Class (Options: 'A', 'B', 'C', 'D' or 'E')
        predominant_period : float, optional, the default is 1.
            Predominant period of the structure. 
        
        Returns
        -------
        None.
        """

        # Add selection settings to self
        self.predominant_period = predominant_period
        self.ag = ag
        self.importance_class = importance_class
        self.target_type = target_type
        self.site_class = site_class
        self.code = 'EC8-Part1'

        # Set the period range
        self.lower_bound_period = 0.2 * predominant_period
        self.upper_bound_period = 2.0 * predominant_period

        # Match periods (periods for error computations)
        self.periods = self.database['Periods']

        # Determine the elastic design spectrum from the user-defined spectrum
        if self.target_path:
            data = np.loadtxt(self.target_path)
            interp_funct = interpolate.interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            target_spectrum = interp_funct(self.periods)

        # Determine the elastic design spectrum from code
        else:
            target_spectrum = sae_ec8_part1(ag, xi, self.periods, importance_class, target_type, site_class)

        # Consider the lower bound spectrum specified by the code as target spectrum
        target_spectrum = 0.9 * target_spectrum  # scale down except for Sa(T[0]) or PGA
        if self.num_components == 2:
            self.spectrum_definition = 'ArithmeticMean'

        # Search the database and filter
        sample_big, vs30, mag, rjb, mechanism, filename1, filename2, rsn, eq_ids, station_code = self._search_database()

        # Sample size of the filtered database
        len_big = sample_big.shape[0]

        # Scale factors based on mse
        scale_factors_big = np.array(np.sum(np.matlib.repmat(target_spectrum, len_big, 1) * sample_big, axis=1) / np.sum(sample_big ** 2, axis=1))

        # Find best matches to the target spectrum from ground-motion database
        temp = (np.matlib.repmat(target_spectrum, len_big, 1) - sample_big) ** 2
        mse = temp.mean(axis=1)

        if self.database['Name'].startswith('ESM'):
            d = {ni: indi for indi, ni in enumerate(set(eq_ids.tolist()))}
            eq_ids_big = np.asarray([d[ni] for ni in eq_ids.tolist()])
        else:
            eq_ids_big = eq_ids.copy()

        rec_id_sorted = np.argsort(mse)
        rec_ids_small = np.ones(self.num_records, dtype=int) * (-1)
        eq_ids_small = np.ones(self.num_records, dtype=int) * (-1)
        idx1 = 0
        idx2 = 0
        while idx1 < self.num_records:  # not more than 3 of the records should be from the same event
            tmp1 = rec_id_sorted[idx2]
            idx2 += 1
            tmp2 = eq_ids_big[tmp1]
            rec_ids_small[idx1] = tmp1
            eq_ids_small[idx1] = tmp2
            if np.sum(eq_ids_small == tmp2) <= self.max_rec_per_event:
                idx1 += 1

        # Initial selection results - based on MSE
        scale_factors_small = scale_factors_big[rec_ids_small]
        sample_small = sample_big[rec_ids_small, :]
        target_spectrum[0] = target_spectrum[0] / 0.9  # scale up for Sa(T[0]) or PGA

        # Must not be lower than target within the period range, find the indicies for this period range
        idxs = np.where((self.database['Periods'] >= self.lower_bound_period) * (self.database['Periods'] <= self.upper_bound_period))[0]
        idxs = np.append(0, idxs)  # Add Sa(T=0) or PGA, approximated as Sa(T=0.01)

        if self.selection_algorithm == 1:
            self.rec_scale_factors = scale_factors_small * np.max(target_spectrum[idxs] / (scale_factors_small.reshape(-1, 1) * sample_small[:, idxs]).mean(axis=0))

        # try to optimize scaling factor to make it closest as possible to 1
        if self.selection_algorithm == 2:
            scale_factors_small = np.max(target_spectrum[idxs] / sample_small[:, idxs].mean(axis=0))
            for i in range(self.num_records):  # Loop for num_records
                # note the ID of the record which is removed
                min_id = rec_ids_small[i]
                # remove the i'th record search for a candidate
                sample_small = np.delete(sample_small, i, 0)
                rec_ids_small = np.delete(rec_ids_small, i)
                eq_ids_small = np.delete(eq_ids_small, i)
                # Try to add a new spectra to the subset list and consider critical periods for error calculations only (idxs)
                min_id, scale_factors_small = self._find_rec_smallest_sf(sample_small[:, idxs], scale_factors_small, target_spectrum[idxs], rec_ids_small, eq_ids_small, min_id, eq_ids_big, sample_big[:, idxs], self.max_rec_per_event)
                # Add new element in the right slot
                sample_small = np.concatenate((sample_small[:i, :], sample_big[min_id, :].reshape(1, sample_big.shape[1]), sample_small[i:, :]), axis=0)
                rec_ids_small = np.concatenate((rec_ids_small[:i], np.array([min_id]), rec_ids_small[i:]))
                eq_ids_small = np.concatenate((eq_ids_small[:i], np.array([eq_ids_big[min_id]]), eq_ids_small[i:]))
            self.rec_scale_factors = np.ones(self.num_records) * float(scale_factors_small)

        # check the scaling
        if np.any(self.rec_scale_factors > self.max_scale_factor) or np.any(self.rec_scale_factors< 1 / self.max_scale_factor):
            raise ValueError('Scaling factor criteria is not satisfied',
                             'Please broaden your selection and scaling criteria or change the optimization scheme...')

        rec_ids_small = rec_ids_small.tolist()
        # Add selected record information to self
        self.rec_vs30 = vs30[rec_ids_small]
        self.rec_rjb = rjb[rec_ids_small]
        self.rec_mag = mag[rec_ids_small]
        self.rec_mech = mechanism[rec_ids_small]
        self.rec_eq_id = eq_ids_big[rec_ids_small]
        self.rec_file_h1 = filename1[rec_ids_small]

        if self.num_components == 2:
            self.rec_file_h2 = filename2[rec_ids_small]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = rsn[rec_ids_small]

        if self.database['Name'] == 'ESM_2018':
            self.rec_station_code = station_code[rec_ids_small]

        rec_idxs = []
        if self.num_components == 1:
            sa_known = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            filename1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_file_h1:
                rec_idxs.append(np.where(filename1 == rec)[0][0])
            self.rec_sa_ln = sa_known[rec_idxs, :]

        elif self.num_components == 2:
            for rec in self.rec_file_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            self.rec_sa_ln = 0.5 * (self.database['Sa_1'][rec_idxs, :] + self.database['Sa_2'][rec_idxs, :])

        # Save the results for whole spectral range
        self.periods = self.database['Periods']
        if self.target_path:
            self.target = interp_funct(self.periods)
        else:
            self.target = sae_ec8_part1(ag, xi, self.periods, importance_class, target_type, site_class)

        print('EC8 - Part 1 based ground motion record selection and amplitude scaling are finished...')
