# EzGM
Toolbox for ground motion record selection and processing. 

[![DOI](https://zenodo.org/badge/291944652.svg)](https://zenodo.org/badge/latestdoi/291944652) 

***
## Getting Started
```
import EzGM
```
The package has three different modules: 
1. **EzGM.selection** deals with the record selection. 
   It can be used to perform unconditional or conditional spectrum based selection in which intensity measure can be chosen as Sa(T*) or AvgSa(T*). The tool makes use of 
   [OpenQuake hazard library](https://docs.openquake.org/oq-engine/3.11/openquake.hazardlib.gsim.html#ground-shaking-intensity-models) and 
   thus any available ground motion prediction equation available can directly be used (see Examples 1 and 2). <br />
   It can also be used to perform the selection based on Turkish Building Earthquake Code, TBEC-2018 (see Example 3), and Eurocode 8 Part 1 (see Example 4). <br />
   Currently, the records can be selected from the two publicly available databases: *NGA_W2* and *ESM_2018*. 
   The original flat-files for these databases were modified by discarding the records which are not possible to download. <br />
   The database files which include features to perform record selection are stored as .mat files in path/to/EzGM/Meta_Data.
   If the user desires to use/add another database such as EXSIM_Duzce, s/he must stick to the same format in publicly available databases. <br />
   Upon performing ground motion record selection/scaling if user desires to get formatted records, for the given metadata, s/he should place the available records from metadata file into the Records.zip with the name of database, 
   e.g. *EXSIM_Duzce.zip* for database *EXSIM_Duzce*. 
   <br /> In case of publicly available databases, the user can also download the records directly by using the associated methods since the records are not generally available beforehand.
   To use *ESM_2018* database, the user must have access token (path/to/current/directory/token.txt) from https://esm-db.eu. The token
   can be retrieved using EzGM as well (see Example 1). In order to use *NGA_W2* database, user must have account obtained from https://ngawest2.berkeley.edu.
2. **EzGM.post_oq** can be used along to post-process results of probabilistic seismic hazard analysis (PSHA) from OpenQuake.Engine. Its methods can be used to read and visualize seismic hazard curves and seismic disaggregation results. The module can be particularly useful
while performing conditional spectrum (CS) based record selection for multiple-stripe analysis (MSA) (See Example 5).
3. **EzGM.processing** can be used to process ground motion records. It contains methods for filtering,  baseline, and intensity measure calculations. (see Example 6).

At the moment, no documentation is available for EzGM; hence, users are recommended to see the jupyter notebook examples to get familiar with EzGM.
These can be accessed and run through *binder* which is an online service to deploy interactive computational environments for online repositories.
For EzGM examples, see:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/HEAD?filepath=Examples%2Fbinder)
***
## Installation
- EzGM requires several other packages: <br /> openquake.engine==3.11.4; selenium==3.141.0; webdriverdownloader==1.1.0.3; numba==0.54.0
- The package management system *pip* can be used to install EzGM. Yet, before the installation is recommended in a clean python 3.8 environment to avoid compatibility issues.
```
pip install EzGM
```
- EzGM downloads google-chrome or firefox webdriver while executing ngaw2_download method. Therefore, user-specified browser ('chrome' or 'firefox') must be readily available.
- Installation of Openquake package in Linux and MACOS is straightforward. In case of windows the package may not be installed correctly if anaconda is used, in other words, geos_c.dll or similar .dll files could be mislocated). To fix this simply, on conda prompt window write:
```
conda install shapely
```
***
## Acknowledgements
Special thanks to Besim Yukselen for his help in the development of ngaw2_download method, and Gerard J. O'Reilly for sharing his knowledge in the field with me. The EzGM.selection.conditional_spectrum method is greatly inspired by the CS_Selection code of Prof. Jack W. Baker whom I thank for sharing his work with the research community.
***
## References
The references associated with each method are provided as the docstring.
If you are going to use the code presented herein for any official study, please refer to: <br /> 
Volkan Ozsarac, Ricardo Monteiro & Gian Michele Calvi (2021). Probabilistic seismic assessment of reinforced concrete bridges using simulated records, Structure and Infrastructure Engineering, DOI: [10.1080/15732479.2021.1956551](https://doi.org/10.1080/15732479.2021.1956551)
***
## Potential Improvements
- Computation of the exact CS
- Addition of spectral matching methods
- Addition of generalized conditional intensity measure approach (GCIM)
- Addition of other code-based scaling methods (e.g. ASCE 7-16)
