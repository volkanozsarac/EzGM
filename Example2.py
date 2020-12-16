from OQproc import *
from EzGM import *
startTime = time()

# %% Hazard Analysis
# For installation of OpenQuake see
# https://github.com/gem/oq-engine/blob/master/doc/installing/development.md

# Set path to OpenQuake model .ini file path
oq_ini = os.path.join('OQ_Model','job.ini')

# Set command to call OpenQuake
oq = 'oq'

# Directory to place post-processing results
post_dir = 'OQproc_Outputs'

# Read .ini file for post-processing purposes
with open(os.path.join(oq_ini)) as f:
    info = f.readlines()
    for line in info:
        if line.startswith('poes'):
            poes = [float(poe) for poe in 
            line.split('\n')[0].split('=')[1].split(',')]
        if line.startswith('export_dir'):
            if sys.platform.startswith('win'):
                results_dir = os.path.join('OQ_Model',line.split('\n')[0].split('=')[1].strip())
            elif sys.platform.startswith('linux'):
                results_dir = line.split('\n')[0].split('=')[1].strip()
            elif sys.platform.startswith('darwin'):       
                results_dir = line.split('\n')[0].split('=')[1].strip()
        if line.startswith('mag_bin_width'):
            exec(line.strip())               
        if line.startswith('distance_bin_width'):
            exec(line.strip())
        if line.startswith('num_epsilon_bins'):
            exec(line.strip())

# Create the export directory for analysis results
ensure_dir(results_dir)

# Create the directory for processed results
ensure_dir(post_dir)

# Run the analysis via system command
os.system(oq + ' engine --run ' + oq_ini + ' --exports csv')

# Extract and plot hazard curves in a reasonable format
proc_hazard(poes, results_dir, post_dir)

# Extract and plot disaggregation results by M and R
proc_disagg_MR(mag_bin_width,distance_bin_width, poes, results_dir, post_dir, n_rows = 3)

# Extract and plot disaggregation results by M, R and epsilon
proc_disagg_MReps(mag_bin_width,distance_bin_width,num_epsilon_bins, poes, results_dir, post_dir, n_rows = 3)

# %% Record Selection
ims = []
for file in os.listdir(post_dir):
    if file.startswith('imls'):
        ims.append(file.split('_')[1].split('.')[0])

for im in ims: # for each im in the im list
    # read hazard and disagg info
    imls=np.loadtxt(os.path.join(post_dir,'imls_'+im+'.out'))
    mean_mags = np.loadtxt(os.path.join(post_dir,'mean_mags_'+im+'.out'))
    mean_dists = np.loadtxt(os.path.join(post_dir,'mean_dists_'+im+'.out'))
    
    # 1.) Initialize the cs_master object for record selection, check which parameters are required for the gmpe you are using.
    cs = cs_master(Tstar = np.arange(0.1,1.1,0.1), gmpe = 'Boore_EtAl_2014', database = 'NGA_W2', pInfo = 1)
    
    for i in range(len(poes)):
    # 2.) Create target spectrum
     	cs.create(site_param = {'vs30': 400}, rup_param = {'rake': 0.0, 'mag': [mean_mags[i]]},
     	          dist_param = {'rjb': [mean_dists[i]]}, Hcont=None, T_Tgt_range  = [0.05,2.5], 
     	          im_Tstar = imls[i], epsilon = None, cond = 1, useVar = 1, corr_func = 'baker_jayaram', 
     	          outdir = os.path.join('EzGM_Outputs_'+im,'POE-'+str(poes[i])+'-in-50-years'))
    
     	# 3.) Select the ground motions
     	cs.select(nGM = 25, selection = 1, Sa_def = 'RotD50', isScaled = 1, maxScale = 2.5,
     	            Mw_lim = None, Vs30_lim = None, Rjb_lim = None, fault_lim = None, nTrials = 20,
     	            weights = [1,2,0.3], seedValue  = 0, nLoop = 2, penalty = 1, tol = 10)
    
     	# Plot the target spectrum, simulated spectra and spectra of selected records
     	cs.plot(tgt = 0, sim = 0, rec = 1, save = 1, show = 0); plt.close('all')
    
     	# 4.) If database == 'NGA_W2' you can first download the records via nga_download method
     	# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
     	cs.nga_download(username = 'example_username', pwd = 'example_password123456')
    
     	# 5.) If you have records already inside recs_f\database.zip\database or
     	# downloaded records for database = NGA_W2 case, write whatever you want,
     	# the object itself, selected and scaled time histories
     	cs.write(obj = 1, recs = 1, recs_f = '')

# Calculate the total time passed
RunTime(startTime)