def process_OQ_hazard(path_hazard_results,rlz,output_filename='hazard_results.pkl'):
    """
    Details:
    This script will take the results of a hazard analysis performed for
    multiple sites and multiples intensity measures.
    It returns the hazard curves defined in terms of mean annual probability of
    exceedance, intensity measure, and the site coordinates in arrays

    Information:
    Author: Gerard J. O'Reilly
    First Version: April 2020

    Notes:

    References:

    Inputs:
    path_hazard_results: Path to the hazard results
    rlz: OpenQuake realisation (rlz) number
    output_filename: Save outputs to a pickle file (optional)

    Returns:
    lon: Longitude
    lat: Latitude
    id_no: OpenQuake ID number
    im: Intensity measure
    s: intensity
    apoe: Annual probability of exceedance, H
    inv_t: investigation time, e.g. 50 years

    """


    # Initialise the arrays that are to be used
    lat = []
    lon = []
    im = []
    s =[]
    poe = []
    apoe = []
    id_no = []

    import os
    import pandas as pd
    import numpy as np
    import pickle

    # Read through each file in the outputs folder
    for file in os.listdir(path_hazard_results):
        if file.startswith(rlz):
            
            # print(file)
            # Strip the IM out of the file name
            im_type = (file.rsplit('-')[2]).rsplit('_')[0]

            # Get the id number of the file
            idn = (file.rsplit('_')[2]).rsplit('.')[0]

            # Load the results in as a dataframe
            df = pd.read_csv(''.join([path_hazard_results,'/',file]),skiprows=1)

            # Get the column headers (but they have a 'poe-' string in them to strip out)
            iml = list(df.columns.values)[3:] # List of headers
            iml = [float(i[4:]) for i in iml] # Strip out the actual IM values
            f = open(''.join([path_hazard_results,'/',file]), "r")
            temp1 = f.readline().split(',')
            temp2 = list(filter(None, temp1))
            inv_t = float(temp2[5].replace(" investigation_time=",""))
            f.close()

            # For each of the sites investigated
            for site in np.arange(len(df)):

                # Append each site's info to the output array
                lat.append([df.lat[site]][0])
                lon.append([df.lon[site]][0])
                im.append(im_type)
                s.append(iml)
                id_no.append(idn)

                # Get the array of poe in inv_t
                poe.append(df.iloc[site,3:].values)

                # For each array of poe, convert it to annual poe
                temp =[]
                for i in np.arange(len(poe[-1])):
                    temp.append(-np.log(1-poe[-1][i])/inv_t)
                apoe.append(temp)

    # If requested, create a results file also
    with open(output_filename, 'wb') as file:
        pickle.dump([lon, lat, id_no, im, s, apoe], file)

    # Create a set of outputs
    return lon, lat, id_no, im, s, apoe, inv_t

def get_iml(poes,apoe_data,iml_data,inv_t):
    
    """
    Details:
    This script will take results of PSHA analysis, and return
    the intensity measure levels for desired probability of exceedance values
    
    Information:
    Author: Volkan Ozsarac
    First Version: May 2020

    Notes:

    References:

    Inputs:
    poes: desired probability of exceedance values to calculate their
        corresponding intensity measure levels
    apoe_data: annual probability of exceedance values (obtained from process_OQ_hazard)
    iml_data: intensity measure levels (obtained from process_OQ_hazard)
    inv_t: investigation time (obtained from process_OQ_hazard)
    
    Returns:
    iml: intensity measure levels corresponding to poes
    """    
    
    import numpy as np
    from scipy import interpolate
    
    infs=np.isinf(apoe_data)
    apoe_data=apoe_data[~infs]
    iml_data=iml_data[~infs]
    nans = np.isnan(apoe_data)
    apoe_data=apoe_data[~nans]
    iml_data=iml_data[~nans]
    
    Ninterp = 1e5
    iml_range=np.arange(min(iml_data),max(iml_data),(max(iml_data)-min(iml_data))/Ninterp)
    apoe_fit = interpolate.interp1d(iml_data,apoe_data,kind='quadratic')(iml_range)
    poe = 1-(1-apoe_fit)**inv_t
    
    idxs=[]
    for i in range(len(poes)):
        temp = abs(poe-poes[i]).tolist()
        idxs.append(temp.index(min(temp)))    # These are actual points where the analysis are carried out and losses are calculated for
    iml=iml_range[idxs]
    
    return iml

def ensure_dir(file_path):
    # procedure to control if the folder path exist. 
    # If it does not the path will be created
    import os
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def plot_hazard(poes, rlz):
    """
    Details
    -------
    This script will save hazard curve plots
    and write iml's corresponding to the given poes
    
    Information
    -----------
    Author: Volkan Ozsarac
    First Version: August 2020

    Parameters
    ----------
    poes : list
        Probabilities of exceedance in tw years for which im levels will be obtained.
    rlz : str
        realization name to plot.

    Returns
    -------
    None.

    """

    import matplotlib.pyplot as plt
    from matplotlib import style # import syle
    import numpy as np
    import os
    cwd = os. getcwd()
    style.use('ggplot')
    output_dir = os.path.join(cwd,'Hazard_Info')
    ensure_dir(output_dir)
    
    fname = os.path.join(output_dir,'hazard_results.pkl')
    path_hazard_results = os.path.join(cwd,'OpenQuake Model','Outputs')
    lon, lat, id_no, im, s, apoe, inv_t = process_OQ_hazard(path_hazard_results,rlz,fname)
    
    imls=[]
    for i in range(len(s)):
        plt.loglog(s[i],apoe[i],label=im[i])
        iml=get_iml(np.asarray(poes),np.asarray(apoe[i]),np.asarray(s[i]),inv_t)
        imls.append(iml)
        fname=os.path.join(output_dir,'imls_'+im[i]+'.out')
        f=open(fname,'w+')
        for j in iml:
            f.write("%.3f\n" % j)
        f.close()

    fname=os.path.join(output_dir,'poes.out')
    f=open(fname,'w+')
    for j in poes:
        f.write("%.4f\n" % j)
    f.close()    
    
    plt.xlabel('IM [g]')
    plt.ylabel('Annual Probability of Exceedance')
    plt.legend()
    plt.grid(True)
    plt.title('Mean Hazard Curves for Lat:%s Lon:%s' % (str(lat[0]),str(lon[0])))
    fname=os.path.join(output_dir,'Hazard_Curves.png')
    plt.savefig(fname)
    
    for i in range(len(apoe)):
        poe = 1-(1-np.asarray(apoe[i]))**inv_t; poe.shape = (len(poe),1)
        imls = np.asarray(s[i]); imls.shape = (len(imls),1)
        haz_cur = np.concatenate([imls,poe],axis=1)
        fname=os.path.join(output_dir,'HazardCurve_'+im[i]+'.out')
        np.savetxt(fname,haz_cur)