# EzGM
Toolbox for ground motion record selection and processing. 

[![DOI](https://zenodo.org/badge/291944652.svg)](https://zenodo.org/badge/latestdoi/291944652) 
***

- Upon performing ground motion record selection/scaling if user desires to get formatted records, for the given metadata, s/he should place the available records from metadata file into the Records.zip with the name of database, 
e.g. *EXSIM_Duzce.zip* for database *EXSIM_Duzce*. In case of publicly available databases, the user can also download the records directly by using the associated methods since the records are not generally available beforehand.
- All the flat-files are stored as .mat files in path/to/EzGM/Meta_Data. Currently, there are two publicly available databases: *NGA_W2* and *ESM_2018*. 
The original flat-files are modified by discarding the records which are not possible to download.
- To use *ESM_2018* database, the user must have access token (path/to/current/directory/token.txt) from https://esm-db.eu. The token
can be retrieved using built-in methods (see Example 1). In order to use *NGA_W2* database, user must have account obtained from https://ngawest2.berkeley.edu.

Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/HEAD?filepath=Examples%2Fbinder%2FTutorial.ipynb)

```
pip install EzGM
import EzGM
```
***

- **EzGM.Selection.conditional_spectrum** is used to perform record selection based on *CS(AvgSa)* and *CS(Sa)* for the given metadata. The tool makes use of openquake.hazardlib, thus any available gmpe available can directly be used.
For available ground motion prediction equations, see https://docs.openquake.org/oq-engine/3.11/openquake.hazardlib.gsim.html#ground-shaking-intensity-models.

Example 1

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample1.ipynb)

Example 2

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample2.ipynb)
***

- **EzGM.Selection.tbdy_2018** is used to perform TBDY 2018 (Turkish Building Code) based record selection

Example 3

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample3.ipynb)
***

- **EzGM.Selection.ec8_part1** is used to perform Eurocode 8 part 1 based record selection

Example 4

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample4.ipynb)
***

- **EzGM.OQProc** can be used along with EzGM.Selection.conditional_spectrum to perform conditional spectrum (CS) Based Record Selection for multiple-stripe analysis (MSA)
upon carrying out probabilistic seismic hazard analysis (PSHA) via OpenQuake.Engine.

Example 5

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample5.ipynb)
***

- **EzGM.GMProc** can be used to process ground motion records (filtering, baseline corrections, IM calculations).

Example 6

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample6.ipynb)
***

## Note
- EzGM downloads google-chrome or firefox webdriver while executing ngaw2_download method. Therefore, user-specified browser ('chrome' or 'firefox') must be readily available.
- Installation of Openquake package in Linux and MACOS is straightforward. In case of windows the package may not be installed correctly if anaconda is used, in other words, geos_c.dll or similar .dll files could be mislocated). To fix this simply, on conda prompt window write:
```
conda install shapely
```

## Acknowledgements
- Special thanks to Besim Yukselen for his help in the development of ngaw2_download method, and Gerard J. O'Reilly for sharing his knowledge in the field with me. The EzGM.conditional_spectrum method is greatly inspired by the CS_Selection code of Prof. Jack W. Baker whom I thank for sharing his work with the research community.
***

## Reference
- If you are going to use the code presented herein for any academic study, please refer to:
Volkan Ozsarac, Ricardo Monteiro & Gian Michele Calvi (2021). Probabilistic seismic assessment of reinforced concrete bridges using simulated records, Structure and Infrastructure Engineering, DOI: [10.1080/15732479.2021.1956551](https://doi.org/10.1080/15732479.2021.1956551)
- Other references associated with each method are provided as the docstring.
***

## Potential Improvements
- Computation of the exact CS
- Addition of spectral matching methods
- Addition of generalized conditional intensity measure approach (GCIM)
- Addition of other code-based scaling methods (e.g. ASCE 7-16)
