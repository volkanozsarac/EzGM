# EzGM
Toolbox for ground motion record selection and processing.

EzGM.cs is used to perform record selection based on CS(AvgSa) and CS(Sa) for the given metadata. The tool makes use of Openquake hazardlib, thus any available gmpe available can directly be used.
If user desires to get formatted records, for the given metadata.
S/he should place the available records from metadata file into the Records.zip with the name of database.
e.g. EXSIM for metadata EXSIM.mat. In case of NGA_W2, user can also download the records directly by inserting account username and password into the associated method. 

EzGM.tbdy_2018 is used to perform record selection in accordance with TBDY2018 (Turkish Building Code). 

EzGM.gm_processor is used to process a ground motion records.
***

# OQproc
Toolbox to visualize of outputs of analysis in OpenQuake.

The script can be used to prepare input required for the CS-based record selection
***

## Required Packages
sys,
os,
shutil,
errno,
stat,
zipfile,
time,
pickle,
copy,
numba,
numpy,
scipy,
matplotlib,
openquake,
selenium,
requests
***

### Note
Installation of Openquake package in Linux and MACOS is straightforward. In case of windows package may not be installed correctly, in other words, geos_c.dll or similar .dll files could be missing). To fix this simply write, pip install shapely or conda install shapely (in case of anaconda). ngaw2_download method can be used if google-chrome is readily available.
***

### Acknowledgements
Special thanks to Besim Yukselen for his help in the development of ngaw2_download method, and Gerard J. O'Reilly for sharing his knowledge in the field with me. The EzGM.conditional_spectrum method is greatly inspired by the CS_Selection code of Prof. Jack W. Baker whom I thank for sharing his work with the research community.

### Reference
If you are going to use the code presented herein any official study, please refer to 
Ozsarac V, Monteiro R.C., Calvi, G.M. (2021). Probabilistic seismic assessment of RC bridges using simulated records (Under Review).
