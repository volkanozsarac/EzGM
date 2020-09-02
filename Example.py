from CS_master import *

# %% Define the period range of interest
# if IM = SA this is a float, if IM = AvgSa this is a list 
# which contains upper and lower bounds, e.g. [0.6, 2.0]
# Tstar = 0.4
Tstar = [0.6, 2.0]

# The database being used to select ground motions
database = 'EXSIM_Duzce' # 'NGA_W1', 'NGA_W2', 'EXSIM_Duzce'

# Define the gmpe to use
gmpe = 'Akkar_EtAlRjb_2014'

# Create the cs object, check which parameters are required for the gmpe you are using.
# You should create site, rupture and distance dictionaries
cs = cs_master(Tstar, gmpe = gmpe, database = database, Tnew = 1, pInfo = 1)

# %% Define the rupture parameters to use
rup_param = {'rake': 0, 'mag': [7, 6.5]} # mag is a list contains disagg. results for each scenario

# Define the site parameters to use
site_param = {'vs30': 620}

# Define the distance parameters to use
dist_param = {'rjb': [10, 12]} # rjb is a list contains disagg. results for each scenario

# Define the conditioning intensity measure level, Sa or AvgSa [g]
im_Tstar = 0.5

# Define the hazard contribution of each scenario if desired 
# if None, they are going to be equal
Hcont = [0.65, 0.35]

# Output directory name
outdir = 'Outputs'

# Create conditional spectrum
cs.create(im_Tstar, site_param, rup_param, dist_param, Hcont=None, T_CS_range = [0.01,4], outdir = outdir)

# Plot conditional spectrum
cs.plot(cs = 1, sim = 0, rec = 0, save = 1)

# %% Select the ground motions
# define number of ground motion records to select
nGM = 30
# define gm selection option, average 2 components or arbitrary 1 component
selection = 2 

# perform the selection
cs.select(nGM = 30, selection = 2, Sa_def='RotD50', 
               Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None)

# Plot the CS with simulated spectra and spectra of selected records
cs.plot(cs = 0, sim = 1, rec = 1, save = 1)

# %% Write the selected ground motions to the text files and locate in output directory
# note you have to have record files in Records.zip for this options
cs.write(cs = 1, recs = 1)

# %% Calculate the total RunTime between module import time 
# and the execution time of following func.
RunTime()