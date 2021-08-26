from EzGM import GMProc
from EzGM.Utility import file_manager, RunTime
from time import time
import numpy as np
import matplotlib.pyplot as plt

# Acquire the run start time
startTime = time()

# Read records
dt, npts, desc, t, Ag1 = file_manager.ReadNGA(inFilename='RSN1158_KOCAELI_DZC180.AT2', content=None, outFilename=None)
dt, npts, desc, t, Ag2 = file_manager.ReadNGA(inFilename='RSN1158_KOCAELI_DZC270.AT2', content=None, outFilename=None)

# Apply baseline correction
Ag_corrected = GMProc.baseline_correction(Ag1, dt, polynomial_type='Linear')

# Apply band-pass filtering
Ag_filtered = GMProc.butterworth_filter(Ag1, dt, cut_off=(0.1, 25))

# Linear elastic analysis of a single degree of freedom system
u, v, ac, ac_tot = GMProc.sdof_ltha(Ag1, dt, T = 1.0, xi = 0.05, m = 1)

# Calculate ground motion parameters
param1 = GMProc.get_parameters(Ag1, dt, T = np.arange(0,4.05,0.05), xi = 0.05)
param2 = GMProc.get_parameters(Ag2, dt, T = np.arange(0,4.05,0.05), xi = 0.05)

# Obtain RotDxx Spectrum
Periods, Sa_RotD50 = GMProc.RotDxx_spectrum(Ag1, Ag2, dt, T = np.arange(0,4.05,0.05), xi = 0.05, xx = 50)

# Since the NGAW2 records are already processed we will not see any difference.
plt.figure()
plt.plot(t, Ag1, label='Component 1 - raw')
plt.plot(t, Ag_corrected, label='Component 1 - corrected')
plt.plot(t, Ag_filtered, label='Component 1 - filtered')
plt.legend()
plt.grid(True)
plt.xlabel('Time [sec]')
plt.ylabel('Acceleration [g]')

plt.figure()
plt.plot(param1['Periods'], param1['PSa'], label='Sa1')
plt.plot(param2['Periods'], param2['PSa'], label='Sa2')
plt.plot(Periods, Sa_RotD50, label='RotD50')
plt.legend()
plt.grid(True)
plt.xlabel('Period [sec]')
plt.ylabel('Sa [g]')

# Calculate the total time passed
RunTime(startTime)