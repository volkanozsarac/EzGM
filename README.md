# EzGM
Toolbox for ground motion record selection and processing

This software performs CS(AvgSa) or CS(Sa) based record selection
for the given metadata. If user desires to get formatted records,
s/he should place the available records from metadata file into the Records.zip with the name of database.
e.g. EXSIM for metadata EXSIM.mat. In case of NGA_W2, user can also download the records directly by inserting account username and password into the associated method.
The main advantage over existing MATLAB codes is that it makes use of Openquake hazardlib, thus any gmpe available there could directly be used.
The example is provided for CS(AvgSa) based ground motion record selection with two hazard scenarios.
Moreover, other methods inside EzGM can be used to process a ground motion record (filter and read NGA records e.g.) and to obtain different intensity measures (see EzGM.gm_parameters). 

# Required Packages
sys,
os,
shutil,
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

# Note:
Installation of Openquake package in Linux and MACOS is straightforward. In case of windows package may not be installed correctly, in other words, geos_c.dll or similar .dll files could be missing). To fix this simply write, pip install shapely or conda install shapely (in case of anaconda)
