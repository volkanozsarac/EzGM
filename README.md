# EzGM 

Toolbox for ground motion record selection and processing. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5878962.svg)](https://doi.org/10.5281/zenodo.5878962)

***
## Getting Started

[![YOUTUBE](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=A2gF4Sc2Sn0)

The package has three different modules: 
1. **EzGM.selection** deals with the record selection. 
   It can be used to perform unconditional or conditional spectrum based selection in which intensity measure can be chosen as Sa(T*) or AvgSa(T*). The tool makes use of 
   [OpenQuake hazard library](https://docs.openquake.org/oq-engine/3.16/reference/openquake.hazardlib.gsim.html) and 
   thus any available ground motion prediction equation available can directly be used (see Example 1). <br />
   It can also be used to perform the selection based on Turkish Building Earthquake Code (TBEC-2018), ASCE 7-16, and Eurocode 8 Part 1 (see Example 2). <br />
   Currently, the records can be selected from the two publicly available databases: *NGA_W2* and *ESM_2018*. 
   The original flat-files for these databases were modified by discarding the records which are not possible to download. <br />
   The database files which include features to perform record selection are stored as .mat files in path/to/EzGM/Meta_Data.
   Upon installation, during the use of this module for the first time, the default Meta_Data folder will be downloaded from: https://drive.google.com/file/d/15cfA8rVB6uLG7T85HOrar7u0AaCOUdxt/view?usp=sharing.
   If users desire to use/add another database such as ESM_2018.mat, they must stick to the same format in publicly available databases. <br />
   Upon performing ground motion record selection/scaling if users desire to get formatted records, for the given metadata, they should place the available records from metadata file into the Records.zip with the name of database, 
   e.g. *ESM_2018.zip* for database *ESM_2018*. 
   <br /> In case of publicly available databases, users can also download the records directly by using the associated methods since the records are not generally available beforehand.
   To use *ESM_2018* database, users must have access token (path/to/current/directory/token.txt) from https://esm-db.eu. The token
   can be retrieved directly using EzGM as well if the user credentials are provided. In order to use *NGA_W2* database, users must have account obtained from https://ngawest2.berkeley.edu.
2. **EzGM.utility** can be used to post-process results of probabilistic seismic hazard analysis (PSHA) from OpenQuake.Engine. Its methods can be used to read and visualize seismic hazard curves and seismic disaggregation results. The module can be particularly useful
while performing conditional spectrum (CS) based record selection for multiple-stripe analysis (MSA) (see Example 3).
3. **EzGM.signal** can be used to process ground motion records. It contains methods for filtering, baseline correction, and intensity measure calculations (see Example 4).

At the moment, no documentation is available for EzGM; hence, users are recommended to see the jupyter notebook examples to get familiar with EzGM.
These can be accessed and run through *binder* which is an online service to deploy interactive computational environments for online repositories. Likewise, the notebooks which are ready to use through google colaboratory can be accessed.
For EzGM examples, see:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/HEAD?filepath=Examples%2Fbinder)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/volkanozsarac/EzGM/blob/master/Examples/google%20colab/Tutorial.ipynb)

## Installation
[![YOUTUBE](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/19steSlknmo)

- EzGM downloads google-chrome or firefox webdriver while executing ngaw2_download method. Therefore, user-specified browser ('chrome' or 'firefox') must be readily available.
- EzGM requires: <br /> openquake.engine>=3.14.0; selenium>=4.2.0; numba>=0.55.1; beautifulsoup4>=4.11.1; tqdm>=4.64.0; 
- The package management system *pip* can be used to install EzGM (python >3.7).
   ```
   pip install EzGM
   ```
- Nonetheless, in order to avoid any potential issues related to the conflicts in external dependencies, the following steps **tested in Windows 64-bit** are **recommended** for installation:
   1. Install [Python 3.8.10](https://www.python.org/downloads/release/python-3810/). During the installation, check *Add Python 3.8 to PATH* option.
   2. Clone the EzGM project, and launch the command prompt or terminal within the clone directory.
   3. Create a new virtual envrionment (venv) with Python 3.8 by entering:
   ```
   py -3.8 python -m venv venv
   ```
   4. Activate venv by entering the following. If it is active (venv) should appear in the terminal.
   ```
   venv\Scripts\activate
   ```
   5. Upgrade pip:
   ```
   python -m pip install --upgrade pip
   ```
   6. Install the package requirements based on your system by entering:
   ```
   pip install -r requirements-py38-win64.txt
   ```
   7. Install EzGM by entering:
   ```
   pip install -e .
   ```
- Once the Python is executed in venv where the package is installed, it can be imported as:
```
import EzGM
```
***
## Acknowledgements
Special thanks to Besim Yukselen for his help in the development of ngaw2_download method, and Gerard J. O'Reilly for sharing his knowledge in the field with me. The EzGM.selection.conditional_spectrum method is greatly inspired by Prof. Jack W. Baker whom I thank for sharing his work with the research community.
***
## References
The references associated with each method are provided as the docstring.
If you are going to use the code presented herein for any official study, please refer to: <br /> 
Volkan Ozsarac, Ricardo Monteiro & Gian Michele Calvi (2023) Probabilistic seismic assessment of reinforced concrete bridges using simulated records, Structure and Infrastructure Engineering, 19:4, 554-574, DOI: [10.1080/15732479.2021.1956551](https://doi.org/10.1080/15732479.2021.1956551)
***
## Potential Improvements
- Re-formatting the source code for easier development
- Computation of the exact CS
- Addition of 3 component selection
- Addition of spectral matching methods
- Addition of generalized conditional intensity measure approach (GCIM)
- Addition of alternative code-based ground motion selection procedures