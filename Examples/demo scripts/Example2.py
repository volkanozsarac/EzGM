##################################################################################
# Conditional Spectrum (CS) Based Record Selection for Multiple Stripes Analysis #
# Upon Carrying out Probabilistic Seismic Hazard Analyss (PSHA) via OpenQuake    #
##################################################################################

from EzGM.selection import ConditionalSpectrum
from EzGM.utility import run_time, make_dir, hazard_curve, disaggregation_mag_dist, disaggregation_mag_dist_eps, check_gmpe_attributes, get_esm_token
from time import time
import os
import numpy as np

start_time = time()

# Hazard Analysis via OpenQuake
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
make_dir(results_dir)

# Create the directory for processed results
make_dir(post_dir)

# Run the analysis via system command
cwd = os.getcwd() # Current working directory
os.chdir(oq_model) # Change directory, head to OQ_model folder
os.system(oq + ' engine --run ' + oq_ini + ' --exports csv')
os.chdir(cwd) # go back to the previous working directory

# Extract and plot hazard curves in a reasonable format
hazard_curve(poes, results_dir, post_dir, show=0)

# Extract and plot disaggregation results by M and R
disaggregation_mag_dist(mag_bin_width, distance_bin_width, results_dir, post_dir, num_rows=3, show=0)

# Extract and plot disaggregation results by M, R and epsilon
disaggregation_mag_dist_eps(mag_bin_width, distance_bin_width, results_dir, post_dir, num_rows=3, show=0)

# Get token once to avoid repetitive download of it
token = get_esm_token(username='example_username@email.com', password='example_password123456')

# Check attributes of ground motion prediction equation BooreEtAl2014
check_gmpe_attributes(gmpe='BooreEtAl2014')

# Get list of IMs
ims = []
for file in os.listdir(post_dir):
    if file.startswith('imls'):
        ims.append(file.split('_')[1].split('.out')[0])

# Start selecting records for each IM in the IM list
for im in ims: 
    # Read hazard and disaggregation info
    imls = np.loadtxt(os.path.join(post_dir, 'imls_' + im + '.out'))
    mean_mags = np.loadtxt(os.path.join(post_dir, 'mean_mags_' + im + '.out'))
    mean_dists = np.loadtxt(os.path.join(post_dir, 'mean_dists_' + im + '.out'))

    for i in range(len(poes)):
        # a) We can use mean magnitude and mean distance values to compute approximate CS.
        mags = [mean_mags[i]]
        dists = [mean_dists[i]]
        hconts = [1.0]
        rakes = [0.0]
        
        # b) We can also consider all contributing scenarios to compute exact CS.
        disagg = np.loadtxt(os.path.join(post_dir,'MagDist_poe_' + str(poes[i]) + '_' + im + '.out'))
        mags = disagg[:, 0].tolist()
        dists = disagg[:, 1].tolist()
        hconts = disagg[:, 2].tolist()
        rakes = [0.0]*len(mags)
        
        # 1) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
        cs = ConditionalSpectrum(database='ESM_2018', output_directory=os.path.join('EzGM_Outputs_' + im, 'POE-' + str(poes[i]) + '-in-50-years'))

        # 2) Create target spectrum
        cs.create(Tstar=np.arange(0.1, 1.1, 0.1), gmpe='BooreEtAl2014', num_components=1, spectrum_definition='RotD50', 
                  site_param={'vs30': reference_vs30_value}, rup_param={'rake': rakes, 'mag': mags},
                  dist_param={'rjb': dists}, hz_cont=hconts, period_range=[0.05, 2.5],
                  im_Tstar=imls[i], epsilon=None, use_variance=1, correlation_model='baker_jayaram')

        # 3) Select the ground motions
        cs.select(num_records=25, is_scaled=1, max_scale_factor=2.5,
                  mag_limits=None, vs30_limits=None, rjb_limits=None, mech_limits=None, num_simulations=20,
                  error_weights=[1, 2, 0.3], seed_value=0, num_greedy_loops=2, penalty=3, tolerance=10)

        # Plot the target spectrum, simulated spectra and spectra of selected records
        cs.plot(target=0, simulations=0, records=1, save=1, show=0)

        # 4) First the records can be downloaded via download method. 
        # If database='ESM_2018' either available token path or valid credentials must be provided. 
        # Token can be retrieved manually by the user or externally using utility.get_esm_token method. 
        # In this case, the token will be retrieved for previously externally downloaded token, and 
        # the records will be retrieved from [https://esm-db.eu]. 
        # If you already have record database elsewhere you can ignore and comment this part
        cs.download(token_path=token)

        # 5) If you have records already inside zip_parent_path\database.zip\database or downloaded records,
        # write whatever you want, the object itself, selected and scaled time histories
        cs.write(object=1, records=0, zip_parent_path='')

# Calculate the total time passed
run_time(start_time)
