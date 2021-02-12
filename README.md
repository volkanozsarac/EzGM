# EzGM
Toolbox for ground motion record selection and processing

This software performs record selection based on TBDY2018 (Turkish Building Code), CS(AvgSa) and CS(Sa)
for the given metadata. If user desires to get formatted records,
s/he should place the available records from metadata file into the Records.zip with the name of database.
e.g. EXSIM for metadata EXSIM.mat. In case of NGA_W2, user can also download the records directly by inserting account username and password into the associated method.
The main advantage over existing MATLAB codes is that it makes use of Openquake hazardlib, thus any available gmpe available can directly be used.
Moreover, other methods inside EzGM can be used to process a ground motion record (filter and read NGA records e.g.) and to obtain different intensity measures (see EzGM.gm_parameters). 

# OQproc
Toolbox to visualize of outputs of analysis in OpenQuake.

This script can be used to prepare input required for the CS-based record selection

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

### Note
Installation of Openquake package in Linux and MACOS is straightforward. In case of windows package may not be installed correctly, in other words, geos_c.dll or similar .dll files could be missing). To fix this simply write, pip install shapely or conda install shapely (in case of anaconda). EzGM.cs_master.nga_download method can be used if google-chrome is readily available.

### Acknowledgements
Special thanks to Besim Yukselen for his help in the development of nga_download method, and Gerard J. O'Reilly for his guidance. I thank Prof. Jack Baker whose work inspired the development of EzGM.
