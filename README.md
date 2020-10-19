CS-Master-Python
--
Conditional spectrum based record selection and scaling

The example is provided for CS(AvgSa) selection.
This software performs CS(AvgSa) or CS(Sa) based record selection
for the given metadata. If user desires to get formatted records,
s/he should place the available records from metadata file into the Records.zip with the name of database.
e.g. NGA_W1 for metadata NGA_W1.mat. The main advantage over existing MATLAB codes is that it makes use of Openquake hazardlib, thus any gmpe available there could directly be used.

# Note:
Installation of Openquake package in Linux and MACOS is straightforward. In case of windows package may not be installed correctly, in other words, geos_c.dll or similar .dll files could be missing). To fix this: pip install shapely or conda install shapely
