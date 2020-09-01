Details
-------
Conditional spectrum based record selection and scaling

Version: 0.1

Created on 26/08/2020

Author: Volkan Ozsarac

Affiliation: University School for Advanced Studies IUSS Pavia

Earthquake Engineering PhD Candidate

Notes
-----
The example is provided for CS(AvgSa) selection.
This software performs CS(AvgSa) or CS(Sa) based record selection
for the given metadata. If user desires to get formatted records,
s/he should place the available records from metadata file into the Records.zip with the name of database.
e.g. NGA_W1 for metadata NGA_W1.mat. The main advantage over existing MATLAB codes is that it makes use of Openquake hazardlib, thus any gmpe available there could directly be used.
