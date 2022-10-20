####################################################
# Conditional Spectrum (CS) Based Record Selection #
####################################################

from EzGM.selection import conditional_spectrum
from EzGM.utility import check_gmpe_attributes, get_esm_token, run_time
from time import time
import numpy as np

# A) IM = Sa(Tstar) Database = NGA_W2
# -----------------------------------
start_time = time()

# A.1) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = conditional_spectrum(database='NGA_W2', outdir='Outputs_A')

# A.2) Create target spectrum
# check the attributes of gmpe to use 'AkkarEtAlRjb2014'
check_gmpe_attributes(gmpe='AkkarEtAlRjb2014')

# Note that intensity measure component is GEOMETRIC_MEAN:Geometric mean of two horizontal components
cs.create(Tstar=0.5, gmpe='AkkarEtAlRjb2014', selection=1, Sa_def='GeoMean',
          site_param={'vs30': 500}, rup_param={'rake': [0.0], 'mag': [7.5]},
          dist_param={'rjb': [10]}, Hcont=None, T_Tgt_range=[0.1, 4.0],
          im_Tstar=1.0, epsilon=None, cond=1, useVar=1, corr_func='akkar')

# Target spectrum can be plotted at this stage
cs.plot(tgt=1, sim=0, rec=0, save=1, show=1)

# A.3) Select the ground motions
cs.select(nGM=25, isScaled=1, maxScale=4, nTrials=20,
          weights=[1, 2, 0.3], seedValue=0, nLoop=2, penalty=1, tol=10,
          Mw_lim=[5.5, 8], Vs30_lim=[360, 760], Rjb_lim=[0, 50], fault_lim=None)

# The simulated spectra and spectra of selected records can be plotted at this stage
cs.plot(tgt=0, sim=1, rec=1, save=1, show=1)

# A.4) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
cs.ngaw2_download('example_username@email.com', pwd='example_password123456', sleeptime=2, browser='firefox')

# A.5) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
cs.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
run_time(start_time)

# B) IM = Sa(Tstar) Database = ESM_2018
# -------------------------------------
start_time = time()

# B.1) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = conditional_spectrum(database='ESM_2018', outdir='Outputs_B')

# B.2) Create target spectrum
# check the attributes of gmpe to use 'AkkarEtAlRjb2014'
check_gmpe_attributes(gmpe='AkkarEtAlRjb2014')

# Note that intensity measure component is GEOMETRIC_MEAN:Geometric mean of two horizontal components
cs.create(Tstar=0.5, gmpe='AkkarEtAlRjb2014', selection=1, Sa_def='GeoMean',
          site_param={'vs30': 500}, rup_param={'rake': [0.0], 'mag': [7.5]},
          dist_param={'rjb': [10]}, Hcont=None, T_Tgt_range=[0.1, 4.0],
          im_Tstar=1.0, epsilon=None, cond=1, useVar=1, corr_func='akkar')

# Target spectrum can be plotted at this stage
cs.plot(tgt=1, sim=0, rec=0, save=1, show=1)

# B.3) Select the ground motions
cs.select(nGM=25, isScaled=1, maxScale=4, nTrials=20,
          weights=[1, 2, 0.3], seedValue=0, nLoop=2, penalty=1, tol=10,
          Mw_lim=[5.5, 8], Vs30_lim=[360, 760], Rjb_lim=[0, 50], fault_lim=None)

# The simulated spectra and spectra of selected records can be plotted at this stage
cs.plot(tgt=0, sim=1, rec=1, save=1, show=1)

# B.4) If database == 'ESM_2018' you can first download the records via esm2018_download method
# from ESM_2018 database [https://esm-db.eu].

# In order to access token file must be retrieved initially.
# copy paste the readily available token.txt into EzGM or generate new one using get_esm_token method.
get_esm_token('example_username@email.com', pwd='example_password123456')

# If token is ready esm2018_download method can be used
cs.esm2018_download()

# B.5) If you have records already inside recs_f\database.zip\database or
# downloaded records for the database , write whatever you want,
# the object itself, selected and scaled time histories
cs.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
run_time(start_time)

# C) IM = AvgSa(Tstar) Database = NGA_W2
# --------------------------------------
start_time = time()

# C.1) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = conditional_spectrum(database='NGA_W2', outdir='Outputs_C')

# C.2) Create target spectrum
# check the attributes of gmpe to use 'BooreEtAl2014'
check_gmpe_attributes(gmpe='BooreEtAl2014')

# create the target spectrum by inserting necessary atribute info
# Note that intensity measure component is RotD50
cs.create(Tstar=np.arange(0.2, 2.4, 0.2), gmpe='BooreEtAl2014', selection=1, Sa_def='RotD50',
          site_param={'vs30': 620}, rup_param={'rake': [0.0, 0.0], 'mag': [6.5, 6.0]},
          dist_param={'rjb': [20, 30]}, Hcont=None, T_Tgt_range=[0.1, 4.5],
          im_Tstar=0.25, epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram')

# Target spectrum can be plotted at this stage
cs.plot(tgt=1, sim=0, rec=0, save=1, show=1)

# C.3) Select the ground motions
cs.select(nGM=25, isScaled=1, maxScale=4,
          Mw_lim=[5.5, 8], Vs30_lim=[360, 760], Rjb_lim=[0, 50], fault_lim=None, nTrials=20,
          weights=[1, 2, 0.3], seedValue=0, nLoop=2, penalty=1, tol=10)

# The simulated spectra and spectra of selected records can be plotted at this stage
cs.plot(tgt=0, sim=1, rec=1, save=1, show=1)

# C.4) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
cs.ngaw2_download(username='example_username@email.com', pwd='example_password123456', sleeptime=2, browser='chrome')

# C.5) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
cs.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
run_time(start_time)
