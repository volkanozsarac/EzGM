######################################################
# Record Selection in Accordance with Building Codes #
######################################################

from EzGM.selection import code_spectrum
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
# 1.) Initialize the code_spectrum object for record selection
# Set the record selection settings at this stage
spec = code_spectrum(database='NGA_W2', outdir='Outputs', target_path=target_path, nGM=11, selection=2,
            Mw_lim=[6.5, 8], Vs30_lim=[200, 700], Rjb_lim=[0, 20], fault_lim=None, opt=1, 
            maxScale=2.5, RecPerEvent=3)

# 2.) Select the ground motions
# If no user-defined spectrum is targeted, site parameters must be inserted by user to construct target spectrum
# Comment on lines associated with building code that you are not interested.

# 2.a) According to TBEC 2018 (Turkish Building Earthquake Code)
spec.tbec2018(Lat=41.0582, Long=29.00951, DD=2, SiteClass='ZC', Tp=1)
# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 2.b) According to ASCE 7-16 (Minimum Design Loads and Associated Criteria for Buildings and Other Structures, 2016)
spec.asce7_16(Lat=34, Long=-118, RiskCat='II', SiteClass='C', T1_small=1, T1_big=1, Tlower = None, Tupper = None)
# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 2.c) According to EC8-1 (Eurocode 8 Part 1)
spec.ec8_part1(ag=0.2, xi=0.05, ImpClass='II', Type='Type1', SiteClass='C', Tp=1)
# selected records can be plotted at this stage
spec.plot(save=1, show=1)

# 3.) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
spec.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456', sleeptime = 2, browser = 'chrome')

# 4.) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
spec.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
run_time(start_time)
