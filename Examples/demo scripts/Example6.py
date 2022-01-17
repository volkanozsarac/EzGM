#####################################
## Showcase for Processing Toolbox ##
#####################################

from EzGM import processing
from EzGM.selection import utility
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Acquire the run start time
start_time = time()

# Read records
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
gm_path1 = os.path.join(parent_path, 'input files', 'RSN1158_KOCAELI_DZC180.AT2')
gm_path2 = os.path.join(parent_path, 'input files', 'RSN1158_KOCAELI_DZC270.AT2')
dt, npts, desc, t, Ag1 = utility.ReadNGA(inFilename= gm_path1, content=None, outFilename=None)
dt, npts, desc, t, Ag2 = utility.ReadNGA(inFilename= gm_path2, content=None, outFilename=None)

# Apply baseline correction
Ag_corrected = processing.baseline_correction(Ag1, dt, polynomial_type='Linear')

# Apply band-pass filtering
Ag_filtered = processing.butterworth_filter(Ag1, dt, cut_off=(0.1, 25))

# Linear elastic analysis of a single degree of freedom system
u, v, ac, ac_tot = processing.sdof_ltha(Ag1, dt, T = 1.0, xi = 0.05, m = 1)

# Calculate ground motion parameters
param1 = processing.get_parameters(Ag1, dt, T = np.arange(0,4.05,0.05), xi = 0.05)
param2 = processing.get_parameters(Ag2, dt, T = np.arange(0,4.05,0.05), xi = 0.05)

# Obtain RotDxx Spectrum
Periods, Sa_RotDxx = processing.RotDxx_spectrum(Ag1, Ag2, dt, T = np.arange(0,4.05,0.05), xi = 0.05, xx = [0, 50, 100])

# Since the NGAW2 records are already processed we will not see any difference.
plt.figure()
plt.plot(t, Ag1, label='Component 1 - raw')
plt.plot(t, Ag_corrected, label='Component 1 - corrected')
plt.plot(t, Ag_filtered, label='Component 1 - filtered')
plt.legend()
plt.grid(True)
plt.xlabel('Time [sec]')
plt.ylabel('Acceleration [g]')
plt.show()

plt.figure()
plt.plot(param1['Periods'], param1['PSa'], label='Sa1')
plt.plot(param2['Periods'], param2['PSa'], label='Sa2')
plt.plot(Periods, Sa_RotDxx[0], label='RotD00')
plt.plot(Periods, Sa_RotDxx[1], label='RotD50')
plt.plot(Periods, Sa_RotDxx[2], label='RotD100')
plt.legend()
plt.grid(True)
plt.xlabel('Period [sec]')
plt.ylabel('Sa [g]')
plt.show()

# Calculate the total time passed
utility.run_time(start_time)