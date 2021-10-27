"""
Includes utilities used within Selection.py    
"""

# Import python libraries
import copy
import errno
import os
import pickle
import shutil
import stat
import sys
import zipfile
import difflib
from time import gmtime, time, sleep
import numpy as np
import requests
from selenium import webdriver
from webdriverdownloader import ChromeDriverDownloader, GeckoDriverDownloader


def RunTime(startTime):
    """

    Details
    -------
    Prints the time passed between startTime and Finishtime (now)
    in hours, minutes, seconds. startTime is a global variable.

    Parameters
    ----------
    startTime : The initial time obtained via time()

    Returns
    -------

    None.
    """
    finishTime = time()
    # Procedure to obtained elapsed time in Hr, Min, and Sec
    timeSeconds = finishTime - startTime
    timeMinutes = int(timeSeconds / 60)
    timeHours = int(timeSeconds / 3600)
    timeMinutes = int(timeMinutes - timeHours * 60)
    timeSeconds = timeSeconds - timeMinutes * 60 - timeHours * 3600
    print("Run time: %d hours: %d minutes: %.2f seconds" % (timeHours, timeMinutes, timeSeconds))


#############################################################################################
#############################################################################################

class database_manager:

    def __init__(self):

        """

        Details
        -------
        This class object contains methods to manipulate used databases.

        """

    def search_database(self):
        """

        Details
        -------
        Search the database and does the filtering.
        
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

            if self.database['Name'].startswith("EXSIM"):
                SaKnown = self.database['Sa_1']
                soil_Vs30 = self.database['soil_Vs30']
                Mw = self.database['magnitude']
                Rjb = self.database['Rjb']
                fault = self.database['mechanism']
                Filename_1 = self.database['Filename_1']
                eq_ID = self.database['EQID']

            else:
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

        elif self.selection == 2:  # SaKnown = Sa_g.m. or RotD50

            if self.Sa_def == 'GeoMean':
                SaKnown = np.sqrt(self.database['Sa_1'] * self.database['Sa_2'])
            elif self.Sa_def == 'SRSS':
                SaKnown = np.sqrt(self.database['Sa_1'] ** 2 + self.database['Sa_2'] ** 2)
            elif self.Sa_def == 'ArithmeticMean':
                SaKnown = (self.database['Sa_1'] + self.database['Sa_2'])/2
            elif self.Sa_def == 'RotD50':  # SaKnown = Sa_RotD50.
                SaKnown = self.database['Sa_RotD50']
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
        # Remove if Sa is nan
        temp = np.unique(np.where(np.isnan(SaKnown))[0]).tolist()
        notAllowed.extend(temp)

        if self.Vs30_lim is not None:  # limiting values on soil exist
            mask = (soil_Vs30 > min(self.Vs30_lim)) * (soil_Vs30 < max(self.Vs30_lim) * np.invert(np.isnan(soil_Vs30)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Mw_lim is not None:  # limiting values on magnitude exist
            mask = (Mw > min(self.Mw_lim)) * (Mw < max(self.Mw_lim) * np.invert(np.isnan(Mw)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Rjb_lim is not None:  # limiting values on Rjb exist
            mask = (Rjb > min(self.Rjb_lim)) * (Rjb < max(self.Rjb_lim) * np.invert(np.isnan(Rjb)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.fault_lim is not None:  # limiting values on mechanism exist
            for fault_i in range(len(self.fault_lim)):
                if fault_i == 0:
                    mask = (fault == self.fault_lim[fault_i] * np.invert(np.isnan(fault)))
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
        elif self.database['Name'].startswith("EXSIM"):
            NGA_num = None
            station_code = None           

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


#############################################################################################
#############################################################################################

class file_manager:

    def __init__(self):

        """

        Details
        -------
        This class object contains methods being used to read and write 
        ground motion record files, and create folders etc.

        """

    @staticmethod
    def create_dir(dir_path):
        """  
        Parameters
        ----------
        dir_path : str
            name of directory to create.

        None.
        """

        def handleRemoveReadonly(func, path, exc):
            excvalue = exc[1]
            if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
                os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
                func(path)
            else:
                raise Warning("Path is being used by at the moment.",
                              "It cannot be recreated.")

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=False, onerror=handleRemoveReadonly)
        os.makedirs(dir_path)

    @staticmethod
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

    @staticmethod
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
        At least one of the two variables must be defined: inFilename, content
    
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
                float(temp[0])
                flag = 1
            except:  # description is in the beginning
                flag = 0

            counter = 0
            desc, row4Val, acc_data = "", "", []

            if flag == 1:
                for x in content:
                    if counter == len(content) - 3:
                        desc = x
                    elif counter == len(content) - 1:
                        row4Val = x
                        if row4Val[0][0] == 'N':
                            val = row4Val.split()
                            npts = float(val[(val.index('NPTS=')) + 1].rstrip(','))
                            try:
                                dt = float(val[(val.index('DT=')) + 1])
                            except:
                                dt = float(val[(val.index('DT=')) + 1].replace('SEC,', ''))
                        else:
                            val = row4Val.split()
                            npts = float(val[0])
                            dt = float(val[1])

                    elif counter < len(content) - 4:
                        data = str(x).split()
                        for value in data:
                            a = float(value)
                            acc_data.append(a)
                        acc = np.asarray(acc_data)
                    counter = counter + 1

            if flag == 0:
                for x in content:
                    if counter == 1:
                        desc = x
                    elif counter == 3:
                        row4Val = x
                        if row4Val[0][0] == 'N':
                            val = row4Val.split()
                            npts = float(val[(val.index('NPTS=')) + 1].rstrip(','))
                            try:
                                dt = float(val[(val.index('DT=')) + 1])
                            except:
                                dt = float(val[(val.index('DT=')) + 1].replace('SEC,', ''))
                        else:
                            val = row4Val.split()
                            npts = float(val[0])
                            dt = float(val[1])

                    elif counter > 3:
                        data = str(x).split()
                        for value in data:
                            a = float(value)
                            acc_data.append(a)
                        acc = np.asarray(acc_data)
                    counter = counter + 1

            t = []  # save time history
            for i in range(0, len(acc_data)):
                ti = i * dt
                t.append(ti)

            if outFilename is not None:
                np.savetxt(outFilename, acc, fmt='%1.4e')

            npts = int(npts)
            return dt, npts, desc, t, acc

        except:
            print("processMotion FAILED!: The record file is not in the directory")
            print(inFilename)

    @staticmethod
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
            Raw content of the exsim record file.
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
            acc = []

            for i in range(64, len(content)):
                acc.append(float(content[i]))

            acc = np.asarray(acc)
            dur = len(acc) * dt
            t = np.arange(0, dur, dt)
            acc = acc / 980.655  # cm/s**2 to g

            if outFilename is not None:
                np.savetxt(outFilename, acc, fmt='%1.4e')

            return dt, npts, desc, t, acc

        except:
            print("processMotion FAILED!: The record file is not in the directory")
            print(inFilename)

    @staticmethod
    def ReadEXSIM(inFilename=None, content=None, outFilename=None):
        """

        Details
        -------
        This function process acceleration history for EXSIM data file.
        
        Parameters
        ----------
        inFilename : str, optional
            Location and name of the input file.
            The default is None
        content    : str, optional
            Raw content of the exsim record file.
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

            desc = content[:12]
            dt = float(content[6].split()[1])
            npts = int(content[5].split()[0])
            acc = []

            for i in range(12, len(content)):
                temp = content[i].split()
                acc.append(float(temp[1]))

            acc = np.asarray(acc)
            if len(acc) < 20000:
                acc = acc[2500:10800]  # get rid of zeros
            dur = len(acc) * dt
            t = np.arange(0, dur, dt)
            acc = acc / 980.655  # cm/s**2 to g

            if outFilename is not None:
                np.savetxt(outFilename, acc, fmt='%1.4e')

            return dt, npts, desc, t, acc

        except:
            print("processMotion FAILED!: The record file is not in the directory")
            print(inFilename)

    def write(self, obj=0, recs=1, recs_f=''):
        """
        
        Details
        -------
        Writes the cs_master object, selected and scaled records
        
        Parameters
        ----------
        obj : int, optional
            flag to write the object into the pickle file. The default is 0.
        recs : int, optional
            flag to write the selected and scaled time histories. 
            The default is 1.
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

        if recs == 1:
            # set the directories and file names
            try:  # this will work if records are downloaded
                zipName = self.Unscaled_rec_file
            except:
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

            # Get record paths
            if self.database['Name'].startswith('EXSIM'):  # EXSIM
                rec_paths1 = [self.database['Name'] + '/' + self.rec_h1[i].split('_acc')[0] + '/'
                              + self.rec_h1[i] for i in range(n)]
            else:  # NGA_W2 or ESM
                if zipName != os.path.join(recs_f, self.database['Name'] + '.zip'):
                    rec_paths1 = self.rec_h1
                    if self.selection == 2:
                        rec_paths2 = self.rec_h2
                else:
                    rec_paths1 = [self.database['Name'] + '/' + self.rec_h1[i] for i in range(n)]
                    if self.selection == 2:
                        rec_paths2 = [self.database['Name'] + '/' + self.rec_h2[i] for i in range(n)]

            # Read contents from zipfile
            contents1 = self.ContentFromZip(rec_paths1, zipName)  # H1 gm components
            if self.selection == 2:
                contents2 = self.ContentFromZip(rec_paths2, zipName)  # H2 gm components

            # Start saving records
            for i in range(n):
                # This is necessary, since scale factor is a single value for code-based selection
                if type(self.rec_scale) is float:
                    SF = self.rec_scale
                else:
                    SF = self.rec_scale[i]

                # Read the record files
                if self.database['Name'].startswith('EXSIM'):  # EXSIM
                    dts[i], _, _, _, inp_acc1 = self.ReadEXSIM(inFilename=self.rec_h1[i], content=contents1[i])
                    gmr_file1 = self.rec_h1[i][:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'

                elif self.database['Name'].startswith('NGA'):  # NGA
                    dts[i], _, _, _, inp_acc1 = self.ReadNGA(inFilename=self.rec_h1[i], content=contents1[i])
                    gmr_file1 = self.rec_h1[i].replace('/', '_')[:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'
                    if self.selection == 2:  # H2 component
                        _, _, _, _, inp_acc2 = self.ReadNGA(inFilename=self.rec_h2[i], content=contents2[i])
                        gmr_file1 = self.rec_h2[i].replace('/', '_')[:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'

                elif self.database['Name'].startswith('ESM'):  # ESM
                    dts[i], _, _, _, inp_acc1 = self.ReadESM(inFilename=self.rec_h1[i], content=contents1[i])
                    gmr_file1 = self.rec_h1[i][:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'
                    if self.selection == 2:  # H2 component
                        _, _, _, _, inp_acc2 = self.ReadESM(inFilename=self.rec_h2[i], content=contents2[i])
                        gmr_file2 = self.rec_h2[i][:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'

                # Write the record files - H1 component
                path = os.path.join(self.outdir, gmr_file1)
                acc_Sc = SF * inp_acc1
                np.savetxt(path, acc_Sc, fmt='%1.4e')
                h1s.write(gmr_file1 + '\n')

                # Write the record files - H2 component
                if self.selection == 2:
                    path = os.path.join(self.outdir, gmr_file2)
                    acc_Sc = SF * inp_acc2
                    np.savetxt(path, acc_Sc, fmt='%1.4e')
                    h2s.write(gmr_file2 + '\n')

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
                obj['gmpe'] = str(obj['bgmpe']).replace('[', '', ).replace(']', '')
                del obj['bgmpe']

            with open(path_obj, 'wb') as file:
                pickle.dump(obj, file)

        print('Finished writing process, the files are located in\n%s' % self.outdir)


#############################################################################################
#############################################################################################

class downloader:

    def __init__(self):
        """

        Details
        -------
        This object contains the methods to download records from 
        record databases available online. 
        For now there are built-in methods is ngaw2_download and esm2018_download which is being used to
        download records from PEER NGA West2 record database.

        """
        pass

    @staticmethod
    def get_esm_token(username, pwd):
        """

        Details
        -------
        This function get ESM database token.

        Notes
        -------
        Data must be obtained using any program supporting the HTTP-POST method, e.g. CURL.
        see: https://esm-db.eu/esmws/generate-signed-message/1/query-options.html
        Credentials must have been retrieved from https://esm-db.eu/#/home

        Parameters
        ----------
        username     : str
            Account username (e-mail),  e.g. 'username@mail.com'
        pwd          : str
            Account password, e.g. 'password!12345'

        Returns
        -------
        None

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

    def esm2018_download(self):
        """

        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from ESM database
        (https://esm-db.eu/) based on their event ID and station_codes.

        Parameters
        ----------
        None

        Returns
        -------
        None

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
            print('Finished esm2018_download method successfully.')

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
            Account username (e-mail),  e.g. 'username@mail.com'
        pwd          : str
            Account password, e.g. 'password!12345'

        sleeptime    : int, default is 3
            Time (sec) spent between each browser operation
            This can be increased or decreased depending on the internet speed
        browser       : str, default is 'chrome'
            The browser to use for download purposes. Valid entries are:
            'chrome' or 'firefox'

        Returns
        -------
        None

        """

        def dir_size(Download_Dir):
            """

            Details
            -------
            Measures download directory size

            Parameters
            ----------
            Download_Dir     : str
                Directory for the output time histories to be downloaded

            Returns
            -------
            total_size      : float
                Measured size of the download directory

            """
            total_size = 0
            for path, dirs, files in os.walk(Download_Dir):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            return total_size

        def download_wait(Download_Dir):
            """

            Details
            -------
            Waits for download to finish, and an additional amount of time based on
            the predefined sleeptime variable.

            Parameters
            ----------
            Download_Dir     : str
                Directory for the output time histories to be downloaded

            Returns
            -------
            None

            """
            delta_size = 100
            flag = 0
            flag_lim = 5
            while delta_size > 0 and flag < flag_lim:
                size_0 = dir_size(Download_Dir)
                sleep(1.5 * sleeptime)
                size_1 = dir_size(Download_Dir)
                if size_1 - size_0 > 0:
                    delta_size = size_1 - size_0
                else:
                    flag += 1
                    print('Finishing in ', flag_lim - flag, '...')
            print(f'Downloaded files are located in\n{Download_Dir}')

        def set_driver(Download_Dir, browser):
            """
            
            Details
            -------
            
            This function starts the webdriver in headless mode

            Parameters
            ----------
            Download_Dir     : str
                Directory for the output time histories to be downloaded
            browser       : str, default is 'chrome'
                The browser to use for download purposes. Valid entries are:
                'chrome' or 'firefox'

            Returns
            -------
            driver      : selenium webdriver object
                Driver object used to download NGA_W2 records
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
                    prefs = {"download.default_directory": Download_Dir}
                    options.add_experimental_option("prefs", prefs)
                    driver = webdriver.Chrome('chromedriver', options=options)

                # Running on Binder or Running on personal computer (PC) using firefox
                elif (_in_ipython_session and 'jovyan' in os.getcwd()) or browser == 'firefox':
                    gdd = GeckoDriverDownloader()
                    driver_path = gdd.download_and_install("v0.26.0")
                    options = webdriver.firefox.options.Options()
                    options.headless = True
                    options.set_preference("browser.download.folderList", 2)
                    options.set_preference("browser.download.dir", Download_Dir)
                    options.set_preference('browser.download.useDownloadDir', True)
                    options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/zip')
                    driver = webdriver.Firefox(executable_path=driver_path[1], options=options)

                # Running on personal computer (PC) using chrome
                elif browser == 'chrome':
                    gdd = ChromeDriverDownloader()
                    driver_path = gdd.download_and_install()
                    ChromeOptions = webdriver.ChromeOptions()
                    prefs = {"download.default_directory": Download_Dir}
                    ChromeOptions.add_experimental_option("prefs", prefs)
                    ChromeOptions.headless = True
                    driver = webdriver.Chrome(executable_path=driver_path[1], options=ChromeOptions)

                print('Webdriver is obtained successfully.')

            except:
                raise RuntimeError('Failed to get webdriver.')

            return driver

        def sign_in(driver, USERNAME, PASSWORD):
            """
            
            Details
            -------
            This function signs in to 'https://ngawest2.berkeley.edu/' with
            given account credentials

            Parameters
            ----------
            driver     : selenium webdriver object
                Driver object used to download NGA_W2 records
            USERNAME   : str
                Account username (e-mail), e.g.: 'username@mail.com'
            PASSWORD   : str
                Account password, e.g.: 'password!12345'

            Returns
            -------
            driver      : selenium webdriver object
                Driver object used to download NGA_W2 records

            """
            print("Signing in with credentials...")
            driver.get('https://ngawest2.berkeley.edu/users/sign_in')
            driver.find_element_by_id('user_email').send_keys(USERNAME)
            driver.find_element_by_id('user_password').send_keys(PASSWORD)
            driver.find_element_by_id('user_submit').click()

            try:
                alert = driver.find_element_by_css_selector('p.alert')
                warn = alert.text
            except:
                warn = None

            if str(warn) == 'Invalid email or password.':
                driver.quit()
                raise Warning('Invalid email or password.')
            else:
                print('Signed in successfully.')

            return driver

        def download(RSNs, Download_Dir, driver):
            """
            
            Details
            -------
            This function dowloads the timehistories which have been indicated with their RSNs
            from 'https://ngawest2.berkeley.edu/'.

            Parameters
            ----------
            RSNs     : str
                A string variable contains RSNs to be downloaded which uses ',' as delimiter
                between RNSs, e.g.: '1,5,91,35,468'
            Download_Dir     : str
                Directory for the output time histories to be downloaded
            driver     : selenium webdriver object
                Driver object used to download NGA_W2 records

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
            except:
                note = 'NO'

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
                download_wait(Download_Dir)
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
        else:
            raise ValueError('You have to use NGA_W2 database to use ngaw2_download method.')
