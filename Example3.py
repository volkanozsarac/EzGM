############################################################
# TBDY 2018 (Turkish Building Code) Based Record Selection #
############################################################

import EzGM

startTime = EzGM.time()
# 1.) Initialize the tbdy_2018 object for record selection
spec = EzGM.ec8_part1(database='NGA_W2', outdir='Outputs')

# 2.) Select the ground motions
spec.select(ag=0.2,xi=0.05, I=1.0, Type='Type1',Soil='A', nGM=3, selection=1, Tp=1,
           Mw_lim=[6.5, 8], Vs30_lim=[200, 700], Rjb_lim=[0, 20], fault_lim=None, opt=2, 
           maxScale=2, weights = [1,1])

# selected records can be plotted at this stage
spec.plot(save=0, show=1)

# 3.) If database == 'NGA_W2' you can first download the records via nga_download method
# from NGA-West2 Database [http://ngawest2.berkeley.edu/] and then use write method
# spec.ngaw2_download(username = 'example_username@email.com', pwd = 'example_password123456')

# 4.) If you have records already inside recs_f\database.zip\database or
# downloaded records for database = NGA_W2 case, write whatever you want,
# the object itself, selected and scaled time histories
# spec.write(obj=1, recs=1, recs_f='')

# Calculate the total time passed
EzGM.RunTime(startTime)