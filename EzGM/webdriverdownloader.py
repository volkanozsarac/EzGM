"""
Module for managing the download of Selenium webdriver binaries.

This code is released under the MIT license.
It is retrieved from: https://github.com/leonidessaguisagjr/webdriverdownloader
Modified by Volkan Ozsarac on 21/04/2022
"""

# Import python libraries
import abc
import logging
import os
import re
from pathlib import Path
import platform
import shutil
import stat
import tarfile
from urllib.parse import urlparse, urlsplit
import zipfile
from bs4 import BeautifulSoup
import requests
import tqdm
import sys

logger = logging.getLogger(__name__)


class WebDriverDownloaderBase:
    """Abstract Base Class for the different web driver downloaders
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, download_root=None, link_path=None, os_name=None):
        """
        Initializer for the class.  Accepts three optional parameters.

        :param download_root: Path where the web driver binaries will be downloaded.  If running as root in macOS or
                              Linux, the default will be '/usr/local/webdriver', otherwise will be '$HOME/webdriver'.
        :param link_path: Path where the link to the web driver binaries will be created.  If running as root in macOS
                          or Linux, the default will be 'usr/local/bin', otherwise will be '$HOME/bin'.  On macOS and
                          Linux, a symlink will be created.
        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        """
        if os_name is None:
            os_name = platform.system()
        if os_name in ['Darwin', 'Linux'] and os.geteuid() == 0:
            base_path = "/usr/local"
        else:
            base_path = os.path.expanduser("~")

        if download_root is None:
            self.download_root = os.path.join(base_path, "webdriver")
        else:
            self.download_root = download_root

        if link_path is None:
            self.link_path = os.path.join(base_path, "bin")
        else:
            self.link_path = link_path

        if not os.path.isdir(self.download_root):
            os.makedirs(self.download_root)
            logger.info("Created download root directory: {0}".format(self.download_root))
        if not os.path.isdir(self.link_path):
            os.makedirs(self.link_path)
            logger.info("Created symlink directory: {0}".format(self.link_path))

    @abc.abstractmethod
    def get_driver_filename(self, os_name=None):
        """
        Method for getting the filename of the web driver binary.

        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :returns: The filename of the web driver binary.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_download_path(self, version="latest"):
        """
        Method for getting the target download path for a web driver binary.

        :param version: String representing the version of the web driver binary to download.  For example, "2.38".
                        Default if no version is specified is "latest".  The version string should match the version
                        as specified on the download page of the webdriver binary.

        :returns: The target download path of the web driver binary.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_download_url(self, version="latest", os_name=None, bitness=None):
        """
        Method for getting the source download URL for a web driver binary.

        :param version: String representing the version of the web driver binary to download.  For example, "2.38".
                        Default if no version is specified is "latest".  The version string should match the version
                        as specified on the download page of the webdriver binary.
        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :param bitness: Bitness of the web driver binary to download, as a str e.g. "32", "64".  If not specified, we
                        will try to guess the bitness by using util.get_architecture_bitness().
        :returns: The source download URL for the web driver binary.
        """
        raise NotImplementedError

    def download(self, version="latest", os_name=None, bitness=None, show_progress_bar=True):
        """
        Method for downloading a web driver binary.

        :param version: String representing the version of the web driver binary to download.  For example, "2.38".
                        Default if no version is specified is "latest".  The version string should match the version
                        as specified on the download page of the webdriver binary.  Prior to downloading, the method
                        will check the local filesystem to see if the driver has been downloaded already and will
                        skip downloading if the file is already present locally.
        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :param bitness: Bitness of the web driver binary to download, as a str e.g. "32", "64".  If not specified, we
                        will try to guess the bitness by using util.get_architecture_bitness().
        :param show_progress_bar: Boolean (default=True) indicating if a progress bar should be shown in the console.
        :returns: The path + filename to the downloaded web driver binary.
        """
        download_url = self.get_download_url(version, os_name=os_name, bitness=bitness)
        filename = os.path.split(urlparse(download_url).path)[1]
        filename_with_path = os.path.join(self.get_download_path(version), filename)
        if not os.path.isdir(self.get_download_path(version)):
            os.makedirs(self.get_download_path(version))
        if os.path.isfile(filename_with_path):
            logger.info("Skipping download. File {0} already on filesystem.".format(filename_with_path))
            return filename_with_path
        data = requests.get(download_url, stream=True)
        if data.status_code == 200:
            logger.debug("Starting download of {0} to {1}".format(download_url, filename_with_path))
            with open(filename_with_path, mode="wb") as fileobj:
                chunk_size = 1024
                if show_progress_bar:
                    expected_size = int(data.headers['Content-Length'])
                    for chunk in tqdm.tqdm(data.iter_content(chunk_size),
                                           total=int(expected_size / chunk_size),
                                           unit='kb'):
                        fileobj.write(chunk)
                else:
                    for chunk in data.iter_content(chunk_size):
                        fileobj.write(chunk)
            logger.debug("Finished downloading {0} to {1}".format(download_url, filename_with_path))
            return filename_with_path
        else:
            error_message = "Error downloading file {0}, got status code: {1}".format(filename, data.status_code)
            logger.error(error_message)
            raise RuntimeError(error_message)

    def download_and_install(self, version="latest", os_name=None, bitness=None, show_progress_bar=True):
        """
        Method for downloading a web driver binary, extracting it into the download directory and creating a symlink
        to the binary in the link directory.

        :param version: String representing the version of the web driver binary to download.  For example, "2.38".
                        Default if no version is specified is "latest".  The version string should match the version
                        as specified on the download page of the webdriver binary.
        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :param bitness: Bitness of the web driver binary to download, as a str e.g. "32", "64".  If not specified, we
                        will try to guess the bitness by using util.get_architecture_bitness().
        :param show_progress_bar: Boolean (default=True) indicating if a progress bar should be shown in the console.
        :returns: Tuple containing the path + filename to [0] the extracted binary, and [1] the symlink to the
                  extracted binary.
        """
        filename_with_path = self.download(version,
                                           os_name=os_name,
                                           bitness=bitness,
                                           show_progress_bar=show_progress_bar)
        filename = os.path.split(filename_with_path)[1]
        if filename.lower().endswith(".tar.gz"):
            extract_dir = os.path.join(self.get_download_path(version), filename[:-7])
        elif filename.lower().endswith(".zip"):
            extract_dir = os.path.join(self.get_download_path(version), filename[:-4])
        else:
            error_message = "Unknown archive format: {0}".format(filename)
            logger.error(error_message)
            raise RuntimeError(error_message)
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir)
            logger.debug("Created directory: {0}".format(extract_dir))
        if filename.lower().endswith(".tar.gz"):
            with tarfile.open(os.path.join(self.get_download_path(version), filename), mode="r:*") as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, extract_dir)
                logger.debug("Extracted files: {0}".format(", ".join(tar.getnames())))
        elif filename.lower().endswith(".zip"):
            with zipfile.ZipFile(os.path.join(self.get_download_path(version), filename), mode="r") as driver_zipfile:
                driver_zipfile.extractall(extract_dir)
        driver_filename = self.get_driver_filename(os_name=os_name)
        for root, dirs, files in os.walk(extract_dir):
            for curr_file in files:
                if curr_file == driver_filename:
                    actual_driver_filename = os.path.join(root, curr_file)
                    break
        if os_name is None:
            os_name = platform.system()
        if os_name in ['Darwin', 'Linux']:
            symlink_src = actual_driver_filename
            symlink_target = os.path.join(self.link_path, driver_filename)
            if os.path.islink(symlink_target):
                if os.path.samefile(symlink_src, symlink_target):
                    logger.info("Symlink already exists: {0} -> {1}".format(symlink_target, symlink_src))
                    return tuple([symlink_src, symlink_target])
                else:
                    logger.warning("Symlink {0} already exists and will be overwritten.".format(symlink_target))
                    os.unlink(symlink_target)
            os.symlink(symlink_src, symlink_target)
            logger.info("Created symlink: {0} -> {1}".format(symlink_target, symlink_src))
            st = os.stat(symlink_src)
            os.chmod(symlink_src, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return tuple([symlink_src, symlink_target])
        elif os_name == "Windows":
            src_file = actual_driver_filename
            dest_file = os.path.join(self.link_path, driver_filename)
            if os.path.isfile(dest_file):
                logger.info("File {0} already exists and will be overwritten.".format(dest_file))
            shutil.copy2(src_file, dest_file)
            return tuple([src_file, dest_file])


class GeckoDriverDownloader(WebDriverDownloaderBase):
    """Class for downloading the Gecko (Mozilla Firefox) WebDriver.
    """

    gecko_driver_releases_api_url = "https://api.github.com/repos/mozilla/geckodriver/releases/"
    gecko_driver_releases_ui_url = "https://github.com/mozilla/geckodriver/releases/"

    def get_driver_filename(self, os_name=None):
        """
        Method for getting the filename of the web driver binary.

        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :returns: The filename of the web driver binary.
        """
        if os_name is None:
            os_name = platform.system()
        if os_name == "Windows":
            return "geckodriver.exe"
        else:
            return "geckodriver"

    def get_download_path(self, version="latest"):
        if version == "latest":
            info = requests.get(self.gecko_driver_releases_api_url + version)
            if info.status_code != 200:
                info_message = "Error attempting to get version info via API, got status code: {0}".format(
                    info.status_code)
                logger.info(info_message)
                resp = requests.get(self.gecko_driver_releases_ui_url + version)
                if resp.status_code == 200:
                    ver = Path(urlsplit(resp.url).path).name
            else:
                ver = info.json()['tag_name']
        else:
            ver = version
        return os.path.join(self.download_root, "gecko", ver)

    def get_download_url(self, version="latest", os_name=None, bitness=None):
        """
        Method for getting the download URL for the Gecko (Mozilla Firefox) driver binary.

        :param version: String representing the version of the web driver binary to download.  For example, "v0.20.1".
                        Default if no version is specified is "latest".  The version string should match the version
                        as specified on the download page of the webdriver binary.
        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :param bitness: Bitness of the web driver binary to download, as a str e.g. "32", "64".  If not specified, we
                        will try to guess the bitness by using util.get_architecture_bitness().
        :returns: The download URL for the Gecko (Mozilla Firefox) driver binary.
        """
        if version == "latest":
            gecko_driver_version_release_api_url = self.gecko_driver_releases_api_url + version
            gecko_driver_version_release_ui_url = self.gecko_driver_releases_ui_url + version
        else:
            gecko_driver_version_release_api_url = self.gecko_driver_releases_api_url + "tags/" + version
            gecko_driver_version_release_ui_url = self.gecko_driver_releases_ui_url + "tags/" + version
        logger.debug("Attempting to access URL: {0}".format(gecko_driver_version_release_api_url))
        info = requests.get(gecko_driver_version_release_api_url)
        if info.status_code != 200:
            info_message = "Error, unable to get info for gecko driver {0} release. Status code: {1}".format(
                version, info.status_code)
            logger.info(info_message)
            resp = requests.get(gecko_driver_version_release_ui_url, allow_redirects=True)
            if resp.status_code == 200:
                json_data = {"assets": []}
            soup = BeautifulSoup(resp.text, features="html.parser")
            urls = [resp.url + a['href'] for a in soup.find_all('a', href=True) if r"/download/" in a['href']]
            for url in urls:
                json_data["assets"].append({"name": Path(urlsplit(url).path).name, "browser_download_url": url})
        else:
            json_data = info.json()

        if os_name is None:
            os_name = platform.system()
            if os_name == "Darwin":
                os_name = "macos"
            elif os_name == "Windows":
                os_name = "win"
            elif os_name == "Linux":
                os_name = "linux"
        if bitness is None:
            bitness = get_architecture_bitness()
            logger.debug("Detected OS: {0}bit {1}".format(bitness, os_name))

        filenames = [asset['name'] for asset in json_data['assets']]
        filename = [name for name in filenames if os_name in name]
        if len(filename) == 0:
            info_message = "Error, unable to find a download for os: {0}".format(os_name)
            logger.error(info_message)
            raise RuntimeError(info_message)
        if len(filename) > 1:
            filename = [name for name in filenames if os_name + bitness in name]
            for file in filename:
                if '.asc' in file and os_name == 'linux':
                    filename.remove(file)
            if len(filename) != 1:
                info_message = "Error, unable to determine correct filename for {0}bit {1}".format(bitness, os_name)
                logger.error(info_message)
                raise RuntimeError(info_message)
        filename = filename[0]

        result = json_data['assets'][filenames.index(filename)]['browser_download_url']
        logger.info("Download URL: {0}".format(result))
        return result


class ChromeDriverDownloader(WebDriverDownloaderBase):
    """Class for downloading the Google Chrome WebDriver.
    """

    chrome_driver_base_url = "https://googlechromelabs.github.io/chrome-for-testing/latest-versions-per-milestone-with-downloads.json"

    def get_driver_filename(self, os_name=None):
        """
        Method for getting the filename of the web driver binary.

        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :returns: The filename of the web driver binary.
        """
        if os_name is None:
            os_name = platform.system()
        if os_name == "Windows":
            return "chromedriver.exe"
        else:
            return "chromedriver"

    def get_download_path(self, version="latest"):

        chrome_version = get_chrome_version()
        milestone = chrome_version.split('.')[0]
        data = requests.get(self.chrome_driver_base_url).json()
        driver_version = data['milestones'][milestone]['version']

        return os.path.join(self.download_root, "chrome", driver_version)

    def get_download_url(self, version="latest", os_name=None, bitness=None):
        """
        Method for getting the download URL for the Google Chome driver binary.

        :param version: String representing the version of the web driver binary to download.  For example, "2.39".
                        Default if no version is specified is "latest".  The version string should match the version
                        as specified on the download page of the webdriver binary.
        :param os_name: Name of the OS to download the web driver binary for, as a str.  If not specified, we will use
                        platform.system() to get the OS.
        :param bitness: Bitness of the web driver binary to download, as a str e.g. "32", "64".  If not specified, we
                        will try to guess the bitness by using util.get_architecture_bitness().
        :returns: The download URL for the Google Chrome driver binary.
        """

        if bitness is None:
            bitness = get_architecture_bitness()
            logger.debug("Detected OS: {0}bit {1}".format(bitness, os_name))

        if os_name is None:
            os_name = platform.system()
        if os_name == "Darwin":
            os_name = 'mac-x' + bitness 
        elif os_name == "Windows":
            os_name = "win" + bitness 
        elif os_name == "Linux":
            os_name = "linux" + bitness 

        chrome_version = get_chrome_version()
        milestone = chrome_version.split('.')[0]

        result = None
        data = requests.get(self.chrome_driver_base_url).json()
        for item in data['milestones'][milestone]['downloads']['chromedriver']:
            if item['platform'] == os_name:
                result = item['url']
                break
        
        resp = requests.get(result)
        if resp.status_code != 200:
            error_message = "Error, unable to get version number for the chrome compatible release, got code: {0}".format(
                resp.status_code)
            logger.error(error_message)
            raise RuntimeError(error_message)

        return result
    

def get_architecture_bitness() -> str:
    """
    Function for getting the "bitness" of the underlying architecture i.e. if it is 32-bit or 64-bit.  Code is based
    on the following:

     https://docs.python.org/3/library/platform.html#cross-platform

    :return: The bitness of the underlying architecture, as a str i.e. "32" or "64"
    """
    # TODO: Try using py-cpuinfo https://pypi.org/project/py-cpuinfo/
    return "64" if sys.maxsize > 2 ** 32 else "32"


def extract_version_registry(output):
    try:
        google_version = ''
        for letter in output[output.rindex('DisplayVersion    REG_SZ') + 24:]:
            if letter != '\n':
                google_version += letter
            else:
                break
        return(google_version.strip())
    except TypeError:
        return

def extract_version_folder():
    # Check if the Chrome folder exists in the x32 or x64 Program Files folders.
    for i in range(2):
        path = 'C:\\Program Files' + (' (x86)' if i else '') +'\\Google\\Chrome\\Application'
        if os.path.isdir(path):
            paths = [f.path for f in os.scandir(path) if f.is_dir()]
            for path in paths:
                filename = os.path.basename(path)
                pattern = '\d+\.\d+\.\d+\.\d+'
                match = re.search(pattern, filename)
                if match and match.group():
                    # Found a Chrome version.
                    return match.group(0)

    return None

def get_chrome_version():
    """
    Programmatically detect the version of the Chrome web browser installed on the PC.
    Compatible with Windows, Mac, Linux.
    Written in Python.
    Uses native OS detection. Does not require Selenium nor the Chrome web driver.
    """

    version = None
    install_path = None

    try:
        os_name = sys.platform.lower()
        if os_name == "linux" or os_name == "linux2":
            # linux
            install_path = "/usr/bin/google-chrome"
        elif os_name == "darwin":
            # OS X
            install_path = "/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
        elif os_name == "win32":
            # Windows...
            try:
                # Try registry key.
                stream = os.popen('reg query "HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Google Chrome"')
                output = stream.read()
                version = extract_version_registry(output)
            except Exception as ex:
                # Try folder path.
                version = extract_version_folder()
    except Exception as ex:
        print(ex)

    version = os.popen(f"{install_path} --version").read().strip('Google Chrome ').strip() if install_path else version

    return version
