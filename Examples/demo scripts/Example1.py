####################################################
# Conditional Spectrum (CS) Based Record Selection #
####################################################

from EzGM.selection import ConditionalSpectrum
from EzGM.utility import check_gmpe_attributes, run_time
from time import time
import numpy as np

# A) IM = Sa(Tstar) Database = NGA_W2
# -----------------------------------
start_time = time()

# A.1) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = ConditionalSpectrum(database='NGA_W2', output_directory='Outputs_A')

# A.2) Create target spectrum
# check the attributes of gmpe to use 'AkkarEtAlRjb2014'
check_gmpe_attributes(gmpe='AkkarEtAlRjb2014')

# Note that intensity measure component is GEOMETRIC_MEAN:Geometric mean of two horizontal components
cs.create(Tstar=0.5, gmpe='AkkarEtAlRjb2014', num_components=2, spectrum_definition='GeoMean',
          site_param={'vs30': 500}, rup_param={'rake': [0.0], 'mag': [7.5]},
          dist_param={'rjb': [10]}, hz_cont=None, period_range=[0.1, 4.0],
          im_Tstar=1.0, epsilon=None, use_variance=1, correlation_model='akkar')

# Target spectrum can be plotted at this stage
cs.plot(target=1, simulations=0, records=0, save=1, show=1)

# A.3) Select the ground motions
cs.select(num_records=25, is_scaled=1, max_scale_factor=4, num_simulations=20,
          error_weights=[1, 2, 0.3], seed_value=0, num_greedy_loops=2, penalty=1, tolerance=10,
          mag_limits=[5.5, 8], vs30_limits=[360, 760], rjb_limits=[0, 50], mech_limits=None)

# The simulated spectra and spectra of selected records can be plotted at this stage
cs.plot(target=0, simulations=1, records=1, save=1, show=1)

# A.4) First the records can be downloaded via download method. 
# If database='NGA_W2' the preferred browser to execute the method should be provided.
# In this case records will be retrieved from [http://ngawest2.berkeley.edu/]. 
# If you already have record database elsewhere you can ignore and comment this part
cs.download(username='example_username@email.com', password='example_password123456', sleeptime=2, browser='firefox')

# A.5) If you have records already inside zip_parent_path\database.zip\database or downloaded records,
# write whatever you want, the object itself, selected and scaled time histories
cs.write(object=1, records=1, zip_parent_path='')

# Calculate the total time passed
run_time(start_time)

# B) IM = Sa(Tstar) Database = ESM_2018
# -------------------------------------
start_time = time()

# B.1) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = ConditionalSpectrum(database='ESM_2018', output_directory='Outputs_B')

# B.2) Create target spectrum
# check the attributes of gmpe to use 'AkkarEtAlRjb2014'
check_gmpe_attributes(gmpe='AkkarEtAlRjb2014')

# Note that intensity measure component is GEOMETRIC_MEAN:Geometric mean of two horizontal components
cs.create(Tstar=0.5, gmpe='AkkarEtAlRjb2014', num_components=2, spectrum_definition='GeoMean',
          site_param={'vs30': 500}, rup_param={'rake': [0.0], 'mag': [7.5]},
          dist_param={'rjb': [10]}, hz_cont=None, period_range=[0.1, 4.0],
          im_Tstar=1.0, epsilon=None, use_variance=1, correlation_model='akkar')

# Target spectrum can be plotted at this stage
cs.plot(target=1, simulations=0, records=0, save=1, show=1)

# B.3) Select the ground motions
cs.select(num_records=25, is_scaled=1, max_scale_factor=4, num_simulations=20,
          error_weights=[1, 2, 0.3], seed_value=0, num_greedy_loops=2, penalty=1, tolerance=10,
          mag_limits=None, vs30_limits=None, rjb_limits=None, mech_limits=None)

# The simulated spectra and spectra of selected records can be plotted at this stage
cs.plot(target=0, simulations=1, records=1, save=1, show=1)

# B.4) First the records can be downloaded via download method. 
# If database='ESM_2018' either available token path or valid credentials must be provided. 
# Token can be retrieved manually by the user or externally using utility.get_esm_token method. 
# In this case, the token will be retrieved internally for the provided credentials, and 
# the records will be retrieved from [https://esm-db.eu]. 
# If you already have record database elsewhere you can ignore and comment this part
cs.download(username='example_username@email.com', password='example_password123456', token_path=None)

# B.5) If you have records already inside zip_parent_path\database.zip\database or downloaded records,
# write whatever you want, the object itself, selected and scaled time histories
cs.write(object=1, records=1, zip_parent_path='')

# Calculate the total time passed
run_time(start_time)

# C) IM = AvgSa(Tstar) Database = NGA_W2
# --------------------------------------
start_time = time()

# C.1) Initialize the conditional_spectrum object for record selection, check which parameters are required for the gmpe you are using.
cs = ConditionalSpectrum(database='NGA_W2', output_directory='Outputs_C')

# C.2) Create target spectrum
# check the attributes of gmpe to use 'BooreEtAl2014'
check_gmpe_attributes(gmpe='BooreEtAl2014')

# create the target spectrum by inserting necessary atribute info
# Note that intensity measure component is RotD50
cs.create(Tstar=np.arange(0.2, 2.4, 0.2), gmpe='BooreEtAl2014', num_components=2, spectrum_definition='RotD50',
          site_param={'vs30': 620}, rup_param={'rake': [0.0, 0.0], 'mag': [6.5, 6.0]},
          dist_param={'rjb': [20, 30]}, hz_cont=[0.4, 0.6], period_range=[0.1, 4.5],
          im_Tstar=0.25, epsilon=None, use_variance=1, correlation_model='baker_jayaram')

# Target spectrum can be plotted at this stage
cs.plot(target=1, simulations=0, records=0, save=1, show=1)

# C.3) Select the ground motions
cs.select(num_records=25, is_scaled=1, max_scale_factor=4,
          mag_limits=[5.5, 8], vs30_limits=[360, 760], rjb_limits=[0, 50], mech_limits=None, num_simulations=20,
          error_weights=[1, 2, 0.3], seed_value=0, num_greedy_loops=2, penalty=1, tolerance=10)

# The simulated spectra and spectra of selected records can be plotted at this stage
cs.plot(target=0, simulations=1, records=1, save=1, show=1)

# C.4) First the records can be downloaded via download method. 
# If database='NGA_W2' the preferred browser to execute the method should be provided.
# In this case records will be retrieved from [http://ngawest2.berkeley.edu/]. 
# If you already have record database elsewhere you can ignore and comment this part
cs.download(username='example_username@email.com', password='example_password123456', sleeptime=2, browser='chrome')

# C.5) If you have records already inside zip_parent_path\database.zip\database or downloaded records,
# write whatever you want, the object itself, selected and scaled time histories
cs.write(object=1, records=1, zip_parent_path='')

# Calculate the total time passed
run_time(start_time)
