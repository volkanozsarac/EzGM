# EzGM
Toolbox for ground motion record selection and processing.

```
pip install EzGM
import EzGM
```
***

## Note
ngaw2_download method can be used only if google-chrome is readily available.
Installation of Openquake package in Linux and MACOS is straightforward. In case of windows the package may not be installed correctly, in other words, geos_c.dll or similar .dll files could be mislocated). To fix this simply, write:
```
conda install shapely
or
pip install shapely
```
***

## Acknowledgements
Special thanks to Besim Yukselen for his help in the development of ngaw2_download method, and Gerard J. O'Reilly for sharing his knowledge in the field with me. The EzGM.conditional_spectrum method is greatly inspired by the CS_Selection code of Prof. Jack W. Baker whom I thank for sharing his work with the research community.
***

## Reference
If you are going to use the code presented herein for any official study, please refer to 
Ozsarac V, Monteiro R.C., Calvi, G.M. (2021). Probabilistic seismic assessment of RC bridges using simulated records. Structure and Infrastructure Engineering.
***

### A) Conditional Spectrum Based Record Selection

EzGM.Selection.conditional_spectrum is used to perform record selection based on CS(AvgSa) and CS(Sa) for the given metadata. The tool makes use of Openquake hazardlib, thus any available gmpe available can directly be used.
If user desires to get formatted records, for the given metadata.
S/he should place the available records from metadata file into the Records.zip with the name of database.
e.g. EXSIM for metadata EXSIM.mat. In case of NGA_W2, user can also download the records directly by inserting account username and password into the associated method. 

User may check https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html for available ground motion prediction equations.

```
from EzGM.Selection import conditonal_spectrum
from time import time
from EzGM.Utility import RunTime

startTime = time()
# 1.) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = conditional_spectrum(Tstar=1.0, gmpe='AkkarEtAlRjb2014', database='NGA_W2', pInfo=1)

# 2.) Create target spectrum
cs.create(site_param={'vs30': 500}, rup_param={'rake': 0.0, 'mag': [7.5]},
          dist_param={'rjb': [10]}, Hcont=None, T_Tgt_range=[0.05, 2.5],
          im_Tstar=1.5, epsilon=None, cond=1, useVar=1, corr_func='akkar',
          outdir='Outputs')

# Target spectrum can be plotted at this stage
cs.plot(tgt=1, sim=0, rec=0, save=1, show=1)

# 3.) Select the ground motions
cs.select(nGM=10, selection=1, Sa_def='RotD50', isScaled=1, maxScale=4,
          Mw_lim=[5.5,8], Vs30_lim=[360,760], Rjb_lim=[0,50], fault_lim=None, nTrials=20,
          weights=[1, 2, 0.3], seedValue=0, nLoop=2, penalty=1, tol=10)

# The simulated spectra and spectra of selected records can be plotted at this stage
cs.plot(tgt=0, sim=1, rec=1, save=1, show=1)

# 4.) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
# cs.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456')

# 5.) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
cs.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
RunTime(startTime)
```
***

### B) TBDY 2018 (Turkish Building Code) Based Record Selection

```
from EzGM.Selection import tbdy_2018
from time import time
from EzGM.Utility import RunTime

startTime = time()
# 1.) Initialize the tbdy_2018 object for record selection
spec = tbdy_2018(database='NGA_W2', outdir='Outputs')

# 2.) Select the ground motions
spec.select(SD1=1.073, SDS=2.333, PGA=0.913, nGM=11, selection=1, Tp=1,
            Mw_lim=[6.5, 8], Vs30_lim=[200, 700], Rjb_lim=[0, 20], fault_lim=None, opt=0, 
            maxScale=2)

# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 3.) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
spec.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456')

# 4.) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
spec.write(obj=1, recs=1, recs_f='')
```
***

### C) Eurocode 8 part 1 Based Record Selection

```
from EzGM.Selection import ec8_part1
from time import time
from EzGM.Utility import RunTime

startTime = time()
# 1.) Initialize the tbdy_2018 object for record selection
spec = ec8_part1(database='NGA_W2', outdir='Outputs')

# 2.) Select the ground motions
spec.select(ag=0.2,xi=0.05, I=1.0, Type='Type1',Soil='A', nGM=11, selection=1, Tp=1,
           Mw_lim=[6.5, 8], Vs30_lim=[200, 700], Rjb_lim=[0, 20], fault_lim=None, opt=0, 
           maxScale=2)

# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 3.) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
spec.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456')

# 4.) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
spec.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
RunTime(startTime)
```
***

### D) EzGM can be used to process ground motion records (Filtering, Correction, IML calculation etc.)
```
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
```
***
### E) EzGM can be used to post-process outputs of PSHA in OpenQuake.
Moreover, the class can be used to prepare input required for the CS-based record selection.

```
# Conditional Spectrum (CS) Based Record Selection for Multiple Stripes Analysis
# Upon Carrying out Probabilistic Seismic Hazard Analyss (PSHA) via OpenQuake

from EzGM.Selection import conditional_spectrum
from EzGM import OQProc
from EzGM.Utility import file_manager, RunTime
from time import time
import os
import numpy as np
import matplotlib.pyplot as plt

startTime = time()

# %% Hazard Analysis via OpenQuake
# Set path to OpenQuake model .ini file path
oq_model = 'OQ_Model' # this is the folder where oq model is located
oq_ini = 'job.ini' # this is .ini file used to run hazard model via openquake

# Set command to call OpenQuake
oq = 'oq'

# Directory to place post-processing results
post_dir = 'OQproc_Outputs'

# Read .ini file for post-processing purposes
with open(os.path.join(oq_model,oq_ini)) as f:
    info = f.readlines()
    for line in info:
        if line.startswith('poes'):
            poes = [float(poe) for poe in
                    line.split('\n')[0].split('=')[1].split(',')]
        if line.startswith('export_dir'):
            results_dir = os.path.join(oq_model, line.split('\n')[0].split('=')[1].strip())
        if line.startswith('mag_bin_width'):
            exec(line.strip())
        if line.startswith('distance_bin_width'):
            exec(line.strip())
        if line.startswith('reference_vs30_value'):
            exec(line.strip())


# Create the export directory for analysis results
file_manager.create_dir(results_dir)

# Create the directory for processed results
file_manager.create_dir(post_dir)

# Run the analysis via system command
cwd = os.getcwd() # Current working directory
os.chdir(oq_model) # Change directory, head to OQ_model folder
os.system(oq + ' engine --run ' + oq_ini + ' --exports csv')
os.chdir(cwd) # go back to the previous working directory

# Extract and plot hazard curves in a reasonable format
OQProc.hazard(poes, results_dir, post_dir)

# Extract and plot disaggregation results by M and R
OQProc.disagg_MR(mag_bin_width, distance_bin_width, poes, results_dir, post_dir, n_rows=3)

# Extract and plot disaggregation results by M, R and epsilon
OQProc.disagg_MReps(mag_bin_width, distance_bin_width, poes, results_dir, post_dir, n_rows=3)

# %% Record Selection
ims = []
for file in os.listdir(post_dir):
    if file.startswith('imls'):
        ims.append(file.split('_')[1].split('.')[0])

for im in ims:  # for each im in the im list
    # read hazard and disaggregation info
    imls = np.loadtxt(os.path.join(post_dir, 'imls_' + im + '.out'))
    mean_mags = np.loadtxt(os.path.join(post_dir, 'mean_mags_' + im + '.out'))
    mean_dists = np.loadtxt(os.path.join(post_dir, 'mean_dists_' + im + '.out'))

    # 1.) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
    cs = conditional_spectrum(Tstar=np.arange(0.1, 1.1, 0.1), gmpe='BooreEtAl2014', database='NGA_W2', pInfo=1)

    for i in range(len(poes)):
        # 2.) Create target spectrum
        cs.create(site_param={'vs30': reference_vs30_value}, rup_param={'rake': 0.0, 'mag': [mean_mags[i]]},
                  dist_param={'rjb': [mean_dists[i]]}, Hcont=None, T_Tgt_range=[0.05, 2.5],
                  im_Tstar=imls[i], epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram',
                  outdir=os.path.join('EzGM_Outputs_' + im, 'POE-' + str(poes[i]) + '-in-50-years'))

        # 3.) Select the ground motions
        cs.select(nGM=25, selection=1, Sa_def='RotD50', isScaled=1, maxScale=2.5,
                  Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, nTrials=20,
                  weights=[1, 2, 0.3], seedValue=0, nLoop=2, penalty=1, tol=10)

        # Plot the target spectrum, simulated spectra and spectra of selected records
        cs.plot(tgt=0, sim=0, rec=1, save=1, show=0)
        plt.close('all')

        # 4.) If database == 'NGA_W2' you can first download the records via nga_download method
        # from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
        # cs.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456')

        # 5.) If you have records already inside recs_f\database.zip\database or
        # downloaded records for database = NGA_W2 case, write whatever you want,
        # the object itself, selected and scaled time histories
        cs.write(obj=1, recs=0, recs_f='')

# Calculate the total time passed
RunTime(startTime)

```
