from CS_master import *
from time import time

def RunTime(startTime, finishTime):
    """

    Details
    -------
    Prints the time passed between startTime and Finishtime
    in hours, minutes, seconds

    Parameters
    ----------
    startTime : float
        Start time, obtained via time.time().
    finishTime : float
        Finish time, obtained via time.time().

    Returns
    -------
    None.

    """
    # Procedure to obtained elapsed time in Hr, Min, and Sec
    timeSeconds = finishTime-startTime
    timeMinutes = int(timeSeconds/60);
    timeHours = int(timeSeconds/3600);
    timeMinutes = int(timeMinutes - timeHours*60)
    timeSeconds = timeSeconds - timeMinutes*60 - timeHours*3600
    print("Run time: %d hours: %d minutes: %.2f seconds"  % (timeHours, timeMinutes, timeSeconds))

startTime = time()

# %% Define the period range of interest
# if IM = SA this is a float, if IM = AvgSa this is a list 
# which contains upper and lower bounds, e.g. [0.6, 2.0]
# Tstar = 0.4 
Tstar = [0.6, 2.0]

# The database being used to select ground motions
database = 'NGA_W1'

# Define the gmpe to use
gmpe = 'Akkar_EtAlRjb_2014'

# Create the cs object, check which parameters are required for the gmpe you are using.
# You should create site, rupture and distance dictionaries
cs = cs_master(Tstar, gmpe = gmpe, database = database)

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

# Create conditional spectrum
cs.create(im_Tstar, site_param, rup_param, dist_param, Hcont=None, T_CS_range = [0.01,4])

# Plot conditional spectrum
cs.plot(cs = 1, sim = 0, rec = 0)

# %% Select the ground motions
# define number of ground motion records to select
nGM = 30
# define gm selection option, average 2 components or arbitrary 1 component
selection = 2 

# perform the selection
cs.select(nGM = 30, selection = 2, Sa_def='RotD50', 
               Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None)

# Plot the CS with simulated spectra and spectra of selected records
cs.plot(cs = 0, sim = 1, rec = 1)

# %% Write the selected ground motions to the text files and locate in outdir
# note you have to have record files in Records.zip for this options
# output folder name
outdir = 'Outputs'
# perform the writing process
cs.write(outdir)

# %%
finishTime = time()
RunTime(startTime, finishTime)