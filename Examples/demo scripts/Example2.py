########################################################################
# Conditional Spectrum (CS) Based Record Selection - IM = AvgSa(Tstar) #
########################################################################

from EzGM.Selection import conditional_spectrum
from time import time
from EzGM.Utility import RunTime
import numpy as np

startTime = time()
# 1.) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = conditional_spectrum(Tstar=np.arange(0.1, 1.1, 0.1), gmpe='AkkarEtAlRjb2014', database='NGA_W2', pInfo=1)

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
cs.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456', sleeptime = 3, browser = 'chrome')

# 5.) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
cs.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
RunTime(startTime)
