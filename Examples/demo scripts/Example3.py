##################################################################################
# Conditional Spectrum (CS) Based Record Selection for Multiple Stripes Analysis #
# Upon Carrying out Probabilistic Seismic Hazard Analyss (PSHA) via OpenQuake    #
##################################################################################

from EzGM.selection import conditional_spectrum
from EzGM.utility import run_time, create_dir, hazard_curve, disagg_MR, disagg_MReps, check_gmpe_attributes, get_esm_token
from time import time
import os
import numpy as np
import matplotlib.pyplot as plt

start_time = time()

# %% Hazard Analysis via OpenQuake
# Set path to OpenQuake model .ini file path
parent_path = os.path.dirname(os.path.realpath(""))
oq_model = os.path.join(parent_path,'input files','OQ_Model') # this is the folder where oq model is located
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
create_dir(results_dir)

# Create the directory for processed results
create_dir(post_dir)

# Run the analysis via system command
cwd = os.getcwd() # Current working directory
os.chdir(oq_model) # Change directory, head to OQ_model folder
os.system(oq + ' engine --run ' + oq_ini + ' --exports csv')
os.chdir(cwd) # go back to the previous working directory

# Extract and plot hazard curves in a reasonable format
hazard_curve(poes, results_dir, post_dir)

# Extract and plot disaggregation results by M and R
disagg_MR(mag_bin_width, distance_bin_width, results_dir, post_dir, n_rows=3)

# Extract and plot disaggregation results by M, R and epsilon
disagg_MReps(mag_bin_width, distance_bin_width, results_dir, post_dir, n_rows=3)

# Get ESM token for ESM database.
get_esm_token('example_username@email.com', pwd = 'example_password123456')

# Check attributes of ground motion prediction equation BooreEtAl2014
check_gmpe_attributes(gmpe='BooreEtAl2014')

# %% Record Selection
ims = []
for file in os.listdir(post_dir):
    if file.startswith('imls'):
        ims.append(file.split('_')[1].split('.out')[0])

for im in ims:  # for each im in the im list
    # read hazard and disaggregation info
    imls = np.loadtxt(os.path.join(post_dir, 'imls_' + im + '.out'))
    mean_mags = np.loadtxt(os.path.join(post_dir, 'mean_mags_' + im + '.out'))
    mean_dists = np.loadtxt(os.path.join(post_dir, 'mean_dists_' + im + '.out'))

    for i in range(len(poes)):
        # 1.) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
        cs = conditional_spectrum(database='NGA_W2', outdir=os.path.join('EzGM_Outputs_' + im, 'POE-' + str(poes[i]) + '-in-50-years'))

        # 2.) Create target spectrum
        cs.create(Tstar=np.arange(0.1, 1.1, 0.1), gmpe='BooreEtAl2014', selection=1, Sa_def='RotD50', 
                  site_param={'vs30': reference_vs30_value}, rup_param={'rake': [0.0], 'mag': [mean_mags[i]]},
                  dist_param={'rjb': [mean_dists[i]]}, Hcont=None, T_Tgt_range=[0.05, 2.5],
                  im_Tstar=imls[i], epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram')

        # 3.) Select the ground motions
        cs.select(nGM=25, isScaled=1, maxScale=2.5,
                  Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, nTrials=20,
                  weights=[1, 2, 0.3], seedValue=0, nLoop=2, penalty=3, tol=10)

        # Plot the target spectrum, simulated spectra and spectra of selected records
        cs.plot(tgt=0, sim=0, rec=1, save=1, show=0)
        plt.close('all')

        # 4.) If database == 'NGA_W2' you can first download the records via nga_download method
        # from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
        # cs.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456', sleeptime = 2, browser = 'chrome')

        # 5.) If you have records already inside recs_f\database.zip\database or
        # downloaded records for database = NGA_W2 case, write whatever you want,
        # the object itself, selected and scaled time histories
        cs.write(obj=1, recs=0, recs_f='')

# Calculate the total time passed
run_time(start_time)
