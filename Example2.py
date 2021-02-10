############################################################
# TBDY 2018 (Turkish Building Code) Based Record Selection #
############################################################

from EzGM import *
startTime = time()
# 1.) Initialize the tbdy_2018 object for record selection
spec = tbdy_2018(database='NGA_W2', outdir='Outputs')

# 2.) Select the ground motions
spec.select(SD1=1.073, SDS=2.333, PGA=0.913, nGM=11, selection=1, Tp=1, 
           Mw_lim=[6.5,8], Vs30_lim=[200,700], Rjb_lim=[0,20], fault_lim=None, opt=1)

# selected records can be plotted at this stage
spec.plot(save = 0, show = 1)

# 3.) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
cs.nga_download(username = 'example_username', pwd = 'example_password123456')

# 4.) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
spec.write(obj = 1, recs = 1, recs_f = '')

# Calculate the total time passed
RunTime(startTime)
