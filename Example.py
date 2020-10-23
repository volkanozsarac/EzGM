from EzGM import *
startTime = time()
# 1.) Create the cs object for record selection, check which parameters are required for the gmpe you are using.
cs = cs_master(Tstar = 0.5, gmpe = 'Boore_EtAl_2014', database = 'NGA_W2', T_resample = [0,0.05], pInfo = 1)

# 2.) Create target spectrum
cs.create(site_param = {'vs30': 520}, rup_param = {'rake': 0.0, 'mag': [7.2, 6.5]}, 
          dist_param = {'rjb': [20, 5]}, Hcont=[0.6,0.4], T_Tgt_range  = [0.05,4], 
          im_Tstar = 2.0, epsilon = None, cond = 1, useVar = 1, outdir = 'Outputs')   

# Plot target spectrum
cs.plot(tgt = 1, sim = 0, rec = 0, save = 1, show = 0)

# 3.) Select the ground motions
cs.select(nGM=10, selection=1, Sa_def='RotD50', isScaled = 1, maxScale = 4,
           Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, nTrials = 20,  
           weights = [1,2,0.3], seedValue  = 0, nLoop = 2, penalty = 0, tol = 10)   

# Plot the CS with simulated spectra and spectra of selected records
cs.plot(tgt = 0, sim = 1, rec = 1, save = 1, show = 0)

# 4.) Download selected ground motions from NGA-West2 Database [http://ngawest2.berkeley.edu/]
cs.nga_download(username = 'example_username', pwd = 'example_password123456')

# 5.) !!! Write the selected ground motions to the text files and locate in output directory
# You have to have records already inside Records.zip file for this option
# Or you have to download the records via nga_download routine (database = 'NGA_W2')
cs.write(cs = 1, recs = 1)

# Calculate the total time passed
RunTime(startTime)