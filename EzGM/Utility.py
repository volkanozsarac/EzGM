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
from time import gmtime, time, sleep
import numpy as np
import numpy.matlib
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
            zipName = os.path.join(recs_f, self.database['Name'] + '.zip')
            if self.database['Name'] == 'NGA_W2':
                try:
                    zipName = self.Unscaled_rec_file
                except:
                    pass

            if type(self.rec_scale) is float:
                SF = self.rec_scale

            n = len(self.rec_h1)
            path_dts = os.path.join(self.outdir, 'GMR_dts.txt')
            dts = np.zeros(n)

            if self.rec_h2 is not None:
                path_H1 = os.path.join(self.outdir, 'GMR_H1_names.txt')
                path_H2 = os.path.join(self.outdir, 'GMR_H2_names.txt')
                h2s = open(path_H2, 'w')

            else:
                path_H1 = os.path.join(self.outdir, 'GMR_names.txt')

            h1s = open(path_H1, 'w')

            if self.database['Name'] == 'NGA_W2':

                if zipName != os.path.join(recs_f, self.database['Name'] + '.zip'):
                    rec_paths = self.rec_h1
                else:
                    rec_paths = [self.database['Name'] + '/' + self.rec_h1[i] for i in range(n)]
                contents = self.ContentFromZip(rec_paths, zipName)

                # Save the H1 gm components
                for i in range(n):
                    if type(self.rec_scale) is not float:
                        SF = self.rec_scale[i]
                    dts[i], _, _, _, inp_acc = self.ReadNGA(inFilename=self.rec_h1[i], content=contents[i])
                    gmr_file = self.rec_h1[i].replace('/', '_')[:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'
                    path = os.path.join(self.outdir, gmr_file)
                    acc_Sc = SF * inp_acc
                    np.savetxt(path, acc_Sc, fmt='%1.4e')
                    h1s.write(gmr_file + '\n')

                # Save the H2 gm components
                if self.rec_h2 is not None:

                    if zipName != os.path.join(recs_f, self.database['Name'] + '.zip'):
                        rec_paths = self.rec_h2
                    else:
                        rec_paths = [self.database['Name'] + '/' + self.rec_h2[i] for i in range(n)]

                    contents = self.ContentFromZip(rec_paths, zipName)
                    for i in range(n):
                        if type(self.rec_scale) is not float:
                            SF = self.rec_scale[i]
                        _, _, _, _, inp_acc = self.ReadNGA(inFilename=self.rec_h2[i], content=contents[i])
                        gmr_file = self.rec_h2[i].replace('/', '_')[:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'
                        path = os.path.join(self.outdir, gmr_file)
                        acc_Sc = SF * inp_acc
                        np.savetxt(path, acc_Sc, fmt='%1.4e')
                        h2s.write(gmr_file + '\n')

                    h2s.close()

            if self.database['Name'].startswith('EXSIM'):
                sf = 1 / 981  # cm/s**2 to g
                rec_paths = [self.database['Name'] + '/' + self.rec_h1[i].split('_acc')[0] + '/'
                             + self.rec_h1[i] for i in range(n)]
                contents = self.ContentFromZip(rec_paths, zipName)

                for i in range(n):
                    dts[i], _, _, t, inp_acc = self.ReadEXSIM(inFilename=self.rec_h1[i], content=contents[i])
                    gmr_file = self.rec_h1[i][:-4] + '_SF_' + "{:.3f}".format(SF) + '.txt'
                    path = os.path.join(self.outdir, gmr_file)
                    acc_Sc = SF * inp_acc * sf
                    np.savetxt(path, acc_Sc, fmt='%1.4e')
                    h1s.write(gmr_file + '\n')

            h1s.close()
            np.savetxt(path_dts, dts, fmt='%.5f')

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
        For now only built-in method is ngaw2_download, which is being used to
        download records from PEER NGA West2 record database.

        """
        pass

    def ngaw2_download(self, username, pwd, sleeptime = 3, browser = 'chrome'):
        """
        
        Details
        -------
        
        This function has been created as a web automation tool in order to 
        download unscaled record time histories from NGA-West2 Database 
        (https://ngawest2.berkeley.edu/) by Record Sequence Numbers (RSNs).

        Parameters
        ----------
        username     : str
                Account username (e-mail)
                            e.g.: 'username@mail.com'
        pwd          : str
                Account password
                            e.g.: 'password!12345'

        sleeptime    : int, default is 3
                Time (sec) spent between each browser operation
                This can be increased or decreased depending on the internet speed

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

            """
            total_size = 0
            for path, dirs, files in os.walk(Download_Dir):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            return total_size

        def download_wait(Download_Dir):
            delta_size = 100
            flag = 0
            flag_lim = 5
            while delta_size > 0 and flag < flag_lim:
                size_0 = dir_size(Download_Dir)
                sleep(1.5*sleeptime)
                size_1 = dir_size(Download_Dir)
                if size_1 - size_0 > 0:
                    delta_size = size_1 - size_0
                else:
                    flag += 1
                    print('Ending in ', flag_lim - flag, '...')
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
                    driver = webdriver.Chrome('chromedriver',options=options)

                # Running on Binder or Running on personal computer (PC) using firefox
                elif (_in_ipython_session and 'jovyan' in os.getcwd()) or browser == 'firefox':
                    gdd = GeckoDriverDownloader()
                    driver_path = gdd.download_and_install("v0.26.0")
                    options = webdriver.firefox.options.Options()
                    options.headless = True
                    options.set_preference("browser.download.folderList", 2)
                    options.set_preference("browser.download.dir", Download_Dir)
                    options.set_preference('browser.download.useDownloadDir', True)
                    options.set_preference('browser.helperApps.neverAsk.saveToDisk','application/zip')
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
                raise  RuntimeError('Failed to get webdriver.')

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
                    Please use the driver have been generated as output of 
                    'go_to_sign_in_page' function
            USERNAME   : str
                    Account username (e-mail)
                                e.g.: 'username@mail.com'
            PASSWORD   : str
                    Account password
                                e.g.: 'password!12345' 
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
                    between RNSs
                                e.g.: '1,5,91,35,468'
            Download_Dir     : str
                    Directory for the output timehistories to be downloaded
            driver     : selenium webdriver object
                    Please use the driver have been generated as output of 
                    sign_in_with_given_creds' function

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
            sleep(1.5*sleeptime)
            try:
                note = driver.find_element_by_id('notice').text
                print(note)
            except:
                note = 'NO'

            if 'NO' in note:
                driver.set_window_size(800, 800)
                driver.save_screenshot(os.path.join(self.outdir,'download_error.png'))
                driver.quit()
                raise Warning("Could not be able to download records!"
                              "Either they no longer exist in database"
                              "or you have exceeded the download limit")
            else:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                sleep(sleeptime)
                driver.find_element_by_xpath("//button[@type='button' and @onclick='getSelectedResult(true)']").click()
                # print("Downloading the records...")
                obj = driver.switch_to.alert
                msg = obj.text
                print(msg)
                sleep(sleeptime)
                obj.accept()
                # driver.switch_to.alert.accept()
                sleep(sleeptime)
                obj = driver.switch_to.alert
                msg = obj.text
                print(msg)
                sleep(sleeptime)
                obj.accept()
                # driver.switch_to.alert.accept()
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
            print('You have to use NGA_W2 database to use ngaw2_download method.')