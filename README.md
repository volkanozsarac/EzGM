# EzGM
Toolbox for ground motion record selection and processing.

Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/HEAD?filepath=Examples%2Fbinder%2FTutorial.ipynb)

```
pip install EzGM
import EzGM
```
***

- EzGM.Selection.conditional_spectrum is used to perform record selection based on CS(AvgSa) and CS(Sa) for the given metadata. The tool makes use of Openquake hazardlib, thus any available gmpe available can directly be used.
- If user desires to get formatted records, for the given metadata, s/he should place the available records from metadata file into the Records.zip with the name of database.
e.g. EXSIM for metadata EXSIM.mat. In case of NGA_W2, user can also download the records directly by inserting account username and password into the associated method. 

- See https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html for available ground motion prediction equations.

Example 1

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample1.ipynb)

Example 2

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample2.ipynb)
***

- EzGM.Selection.tbdy_2018 is used to perform TBDY 2018 (Turkish Building Code) based record selection

Example 3

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample3.ipynb)
***

- EzGM.Selection.ec8_part1 is used to perform Eurocode 8 part 1 based record selection

Example 4

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample4.ipynb)
***

- EzGM.OQProc can be used along with EzGM.Selection.conditional_spectrum to perform conditional spectrum (CS) Based Record Selection for multiple-stripe analysis (MSA)
upon carrying out probabilistic seismic hazard analysis (PSHA) via OpenQuake.Engine.

Example 5

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample5.ipynb)
***

- EzGM.GMProc can be used to process ground motion records (filtering, baseline corrections, IM calculations).

Example 6

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/volkanozsarac/EzGM/master?filepath=Examples%2Fbinder%2FExample6.ipynb)
***

## Note
- On PC, ngaw2_download method can be used only if google-chrome is readily available. EzGM is set to download chromedriver automatically into site-packages if it is not available.
- Installation of Openquake package in Linux and MACOS is straightforward. In case of windows the package may not be installed correctly if anaconda is used, in other words, geos_c.dll or similar .dll files could be mislocated). To fix this simply, on conda prompt window write:
```
conda install shapely
```

## Acknowledgements
- Special thanks to Besim Yukselen for his help in the development of ngaw2_download method, and Gerard J. O'Reilly for sharing his knowledge in the field with me. The EzGM.conditional_spectrum method is greatly inspired by the CS_Selection code of Prof. Jack W. Baker whom I thank for sharing his work with the research community.
***

## Reference
- If you are going to use the code presented herein for any official study, please refer to 
Ozsarac V, Monteiro R.C., Calvi, G.M. (2021). Probabilistic seismic assessment of RC bridges using simulated records. Structure and Infrastructure Engineering.
***

## Potential Improvements
- Computation of exact CS
- Selection from ESM database
- Addition of other scaling or matching methods
- Use of SQL database files instead of .csv or .mat (reduces file size)