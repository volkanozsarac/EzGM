# get the hazard curve
from Utility.process_OQ_hazard import *
rlz = 'hazard_curve-mean'
poe_disagg=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001]
plot_hazard(poe_disagg, rlz)

# Get the disaggregation results
Mbin = 0.25 # magnitude bin used in disaggregation
dbin = 2  # distance bin used in disaggregation
n_rows = 3 # total number of rows for subplots

# for M and R only
from Utility.process_OQ_disagg_v1 import *
plot_disagg(Mbin,dbin,n_rows)

# for M, R and epsilon
# from process_OQ_disagg_v2 import *
# plot_disagg(Mbin,dbin,n_rows)

from Utility.CS_master import *
plt.close('all')
# %% Define the period range of interest
# if IM = SA this is a float, if IM = AvgSa this is a list 
# which contains upper and lower bounds, e.g. [0.6, 2.0]
# Tstar = 0.4
Tstar = [0.1, 1.0]

# The database being used to select ground motions
database = 'EXSIM_Duzce' # 'NGA_W1', 'NGA_W2', 'EXSIM_Duzce'

# Define the gmpe to use
gmpe = 'Akkar_EtAlRjb_2014'

Vs30 = 520

T_CS = [0.05,2.5]

# Create the cs object, check which parameters are required for the gmpe you are using.
# You should create site, rupture and distance dictionaries
cs = cs_master(Tstar, gmpe = gmpe, database = database, T_resample = [1,0.05], pInfo = 1)

poes=np.loadtxt(os.path.join('Hazard_Info','poes.out'))
imls=np.loadtxt(os.path.join('Hazard_Info','imls_AvgSA.out'))
mean_mags = np.loadtxt(os.path.join('Hazard_Info','mean_mags.out'))
mean_dists = np.loadtxt(os.path.join('Hazard_Info','mean_dists.out'))

for i in range(len(poes)):

    # %% Define the rupture parameters to use
    rup_param = {'rake': 0, 'mag': [mean_mags[i]]} # mag is a list contains disagg. results for each scenario
    
    # Define the site parameters to use
    site_param = {'vs30': Vs30}
    
    # Define the distance parameters to use
    dist_param = {'rjb': [mean_dists[i]]} # rjb is a list contains disagg. results for each scenario
    
    # Define the conditioning intensity measure level, Sa or AvgSa [g]
    im_Tstar = imls[i]
    
    # Output directory name
    outdir = os.path.join(database,'POE-'+str(poes[i])+'-in-50-years')
    
    # Create conditional spectrum
    cs.create(im_Tstar, site_param, rup_param, dist_param, Hcont=None, T_CS_range = T_CS, outdir = outdir)
    
    # Plot conditional spectrum
    cs.plot(cs = 1, sim = 0, rec = 0, save = 1)
    
    # %% Select the ground motions
    # define number of ground motion records to select
    nGM = 25
    # define gm selection option, average 2 components or arbitrary 1 component
    selection = 1 
    
    # perform the selection
    cs.select(nGM = 25, selection = 1, Sa_def='RotD50', 
                    Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None)   
 
    # Plot the CS with simulated spectra and spectra of selected records
    cs.plot(cs = 0, sim = 1, rec = 1, save = 1)
    
    # %% Write the selected ground motions to the text files and locate in output directory
    # note you have to have record files in Records.zip for this options
    cs.write(cs = 1, recs = 1)
    plt.close('all')

# %% Calculate the total RunTime between module import time 
# and the execution time of following func.
RunTime()