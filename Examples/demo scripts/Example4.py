####################################################
## Showcase for seismic signal processing toolbox ##
####################################################

from EzGM import signal
from EzGM.utility import read_esm, run_time
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Acquire the run start time
start_time = time()

# Read records
parent_path = os.path.dirname(os.path.realpath(""))
gm_path1 = os.path.join(parent_path, 'input files', '19991112165722_1401_mp_RawAcc_E.asc')
gm_path2 = os.path.join(parent_path, 'input files', '19991112165722_1401_mp_RawAcc_N.asc')
gm_path3 = os.path.join(parent_path, 'input files', '19991112165722_1401_mp_RawAcc_U.asc')
dt, npts, desc, t, ag1 = read_esm(in_filename= gm_path1)
dt, npts, desc, t, ag2 = read_esm(in_filename= gm_path2)
dt, npts, desc, t, ag3 = read_esm(in_filename= gm_path3)

# Apply baseline correction
ag1_corrected = signal.baseline_correction(ag1, dt, polynomial_type='Linear')
ag2_corrected = signal.baseline_correction(ag2, dt, polynomial_type='Linear')
ag3_corrected = signal.baseline_correction(ag3, dt, polynomial_type='Linear')

# Apply band-pass filtering
ag1_filtered = signal.butterworth_filter(ag1_corrected, dt, cut_off=(0.1, 25), filter_order=4, filter_type='bandpass')
ag2_filtered = signal.butterworth_filter(ag2_corrected, dt, cut_off=(0.1, 25), filter_order=4, filter_type='bandpass')
ag3_filtered = signal.butterworth_filter(ag3_corrected, dt, cut_off=(0.1, 25), filter_order=4, filter_type='bandpass')

# Linear elastic analysis of a single degree of freedom system
u1, v1, ac1, ac_tot1 = signal.sdof_ltha(ag1_filtered, dt, periods = 1.0, xi = 0.05)
u2, v2, ac2, ac_tot2 = signal.sdof_ltha(ag2_filtered, dt, periods = 1.0, xi = 0.05)
u3, v3, ac3, ac_tot3 = signal.sdof_ltha(ag3_filtered, dt, periods = 1.0, xi = 0.05)

# Calculate ground motion parameters
param1 = signal.get_parameters(ag1_filtered, dt, periods = np.arange(0,4.05,0.05), xi = 0.05)
param2 = signal.get_parameters(ag2_filtered, dt, periods = np.arange(0,4.05,0.05), xi = 0.05)
param3 = signal.get_parameters(ag3_filtered, dt, periods = np.arange(0,4.05,0.05), xi = 0.05)

# Obtain RotDxx Spectrum
periods, sa_rotdxx = signal.get_sa_rotdxx(ag1_filtered, ag2_filtered, dt, periods = np.arange(0,4.05,0.05), xi = 0.05, xx = [0, 50, 100])

fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12,4))
axs[0].plot(t, ag1, label='Raw', color='red')
axs[0].plot(t, ag1_filtered, label='Filtered', color='blue')
axs[0].grid(True)
axs[0].set_title('H1 Component')
axs[1].plot(t, ag2, color='red')
axs[1].plot(t, ag2_filtered, color='blue')
axs[1].grid(True)
axs[1].set_title('H2 Component')
axs[2].plot(t, ag3, color='red')
axs[2].plot(t, ag3_filtered, color='blue')
axs[2].grid(True)
axs[2].set_title('V Component')

fig.text(0.5, 0.0, 'Time [sec]', ha='center')
fig.text(0.0, 0.5, 'Acceleration [g]', va='center', rotation='vertical')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.66, 0.05, 0.5, 0.5), frameon=False)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(param1['Periods'], param1['PSa'], label='Sa_H1', color = 'blue')
plt.plot(param2['Periods'], param2['PSa'], label='Sa_H2', color = 'magenta')
plt.plot(param3['Periods'], param3['PSa'], label='Sa_V', color = 'black')
plt.plot(periods, sa_rotdxx[0], label='RotD00', ls='--', color = 'green')
plt.plot(periods, sa_rotdxx[1], label='RotD50', ls='--', color = 'orange')
plt.plot(periods, sa_rotdxx[2], label='RotD100', ls='--', color = 'red')
plt.legend(frameon=False)
plt.grid(True)
plt.xlabel('Period [sec]')
plt.ylabel('Sa [g]')
plt.xlim([periods[0],periods[-1]])
plt.show()

# Calculate the total time passed
run_time(start_time)