from CS_master import *
# 1.) Create the cs object for record selection, check which parameters are required for the gmpe you are using.
cs = cs_master(Tstar = 0.5, gmpe = 'Boore_Atkinson_2008', database = 'NGA_W2', T_resample = [0,0.05], pInfo = 1)

# 2.) Create target spectrum
cs.create(site_param = {'vs30': 520}, rup_param = {'rake': 0.0, 'mag': [7.2, 6.5]}, 
          dist_param = {'rjb': [20, 5]}, Hcont=[0.6,0.4], T_Tgt_range  = [0.05,4], 
          im_Tstar = 2.0, epsilon = None, cond = 1, useVar = 1, outdir = 'Outputs')   

# Plot target spectrum
cs.plot(cs = 1, sim = 0, rec = 0, save = 1)

# 3.) Select the ground motions
cs.select(nGM=10, selection=1, Sa_def='RotD50', isScaled = 1, maxScale = 4,
           Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, nTrials = 20,  
           weights = [1,2,0.3], seedValue  = 0, nLoop = 2, penalty = 0, tol = 10)   

# Plot the CS with simulated spectra and spectra of selected records
cs.plot(cs = 0, sim = 1, rec = 1, save = 1)

# Download selected ground motions from NGA-West2 Database [http://ngawest2.berkeley.edu/]
cs.nga_download(username = 'example_username', pwd = 'example_password123456')

# 4.) Write the selected ground motions to the text files and locate in output directory
# !!!Autodownload feature after performing record selection 
# using NGA_W2 database is going to be added soon.
# !!! note that you have to have record files in Records.zip for this option
# cs.write(cs = 1, recs = 1)

# Calculate the total time passed
RunTime()