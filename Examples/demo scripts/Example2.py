######################################################
# Record Selection in Accordance with Building Codes #
######################################################

from EzGM.selection import CodeSpectrum
from EzGM.utility import run_time
from time import time
import os

# Path to user-defined target spectrum
parent_path = os.path.dirname(os.path.realpath(""))
target_path = os.path.join(parent_path,'input files','Target_Spectrum.txt')

# if target_path=none, target spectrum is generated based on site parameters, and specified code
# Comment the next line if you want to use the user-defined spectrum.
target_path=None 

start_time = time()
# 1) Initialize the code_spectrum object for record selection
# Set the record selection settings at this stage
spec = CodeSpectrum(database='NGA_W2', output_directory='Outputs', target_path=target_path, num_records=11, num_components=2,
            mag_limits=[6.5, 8], vs30_limits=[200, 700], rjb_limits=[0, 20], mech_limits=None, selection_algorithm=1, 
            max_scale_factor=2.5, max_rec_per_event=3)

# 2) Select the ground motions
# If no user-defined spectrum is targeted, site parameters must be inserted by user to construct target spectrum
# Comment on lines associated with building code that you are not interested.

# 2.a) According to TBEC 2018 (Turkish Building Earthquake Code)
spec.select_tbec2018(lat=41.0582, long=29.00951, dd_level=2, site_class='ZC', predominant_period=1)
# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 2.b) According to ASCE 7-16 (Minimum Design Loads and Associated Criteria for Buildings and Other Structures, 2016)
spec.select_asce7_16(lat=34, long=-118, risk_cat='II', site_class='C', fundamental_periods = [1, 1], lower_bound_period = None, upper_bound_period = None)
# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 2.c) According to EC8-1 (Eurocode 8 Part 1)
spec.select_ec8_part1(ag=0.2, xi=0.05, importance_class='II', target_type='Type1', site_class='C', predominant_period=1)
# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 3) First the records can be downloaded via download method. 
# If database='NGA_W2' the preferred browser to execute the method should be provided.
# In this case records will be retrieved from [http://ngawest2.berkeley.edu/]. 
# If you already have record database elsewhere you can ignore and comment this part
spec.download(username='example_username@email.com', password='example_password123456', sleeptime=2, browser='chrome')

# 4) If you have records already inside zip_parent_path\database.zip\database or downloaded records,
# write whatever you want, the object itself, selected and scaled time histories
spec.write(object=1, records=1, zip_parent_path='')

# Calculate the total time passed
run_time(start_time)
