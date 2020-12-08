"""
|-----------------------------------------------------------------------|
|                                                                       |
|    EzGM                                                               |
|    Toolbox for ground motion record                                   |
|    selection and processing                                           |
|    Version: 0.5                                                       |
|                                                                       |
|    Created on 06/11/2020                                              |
|    Update on 08/12/2020                                               |
|    Author: Volkan Ozsarac                                             |
|    Affiliation: University School for Advanced Studies IUSS Pavia     |
|    Earthquake Engineering PhD Candidate                               |
|                                                                       |
|-----------------------------------------------------------------------|
"""

# Import python libraries
import sys
import os
import shutil
import zipfile
from time import gmtime, time, sleep
import pickle
import copy
from numba import njit
import numpy as np
import numpy.matlib
from scipy.stats import skew
from scipy.signal import butter, filtfilt
from scipy.integrate import cumtrapz
from scipy.fft import fft, fftfreq, fftshift
from scipy.io import loadmat
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
from openquake.hazardlib import gsim, imt, const
from selenium import webdriver
import requests

class cs_master:
    """
    This class is used to
        1) Create target spectrum
            Uncoditional spectrum using specified gmpe
            Conditional spectrum using average spectral acceleration
            Conditional spectrum using spectral acceleration
            with and without considering variance
        2) Selecting suitable ground motion sets for target spectrum
        3) Scaling and processing of selected ground motion records
    """
    
    def __init__(self, Tstar = 0.5, gmpe = 'Boore_EtAl_2014', database = 'NGA_W2', pInfo = 1):
        """
        Details
        -------
        Loads the database and add spectral values for Tstar 
        if they are not present via interpolation.
        
        Parameters
        ----------
        Tstar    : int, float, numpy.ndarray, the default is None.
            Conditioning period or periods in case of AvgSa [sec].
        gmpe     : str, optional
            GMPE model (see OpenQuake library). 
            The default is 'Boore_EtAl_2014'.
        database : str, optional
            database to use: NGA_W2, EXSIM_Duzce, etc.
            The default is NGA_W1.        
        pInfo    : int, optional
            flag to print required input for the gmpe which is going to be used. 
            (0: no, 1:yes)
            The default is 1.
            
        Returns
        -------
        None.
        """
        
        # add Tstar to self
        if isinstance(Tstar,int) or isinstance(Tstar,float):
            self.Tstar = np.array([Tstar])
        elif isinstance(Tstar,numpy.ndarray):
            self.Tstar = Tstar    
        
        # Add the input the ground motion database to use
        matfile = os.path.join('Meta_Data',database)
        self.database = loadmat(matfile, squeeze_me=True)
        self.database['Name'] = database
        
        # check if AvgSa or Sa is used as IM, 
        # then in case of Sa(T*) add T* and Sa(T*) if not present
        if not self.Tstar[0] in self.database['Periods'] and len(self.Tstar) == 1: 
            f = interpolate.interp1d(self.database['Periods'], self.database['Sa_1'],axis=1)            
            Sa_int = f(self.Tstar[0]); Sa_int.shape = (len(Sa_int),1)           
            Sa = np.append(self.database['Sa_1'], Sa_int, axis=1)
            Periods = np.append(self.database['Periods'],self.Tstar[0])            
            self.database['Sa_1'] = Sa[:,np.argsort(Periods)]

            if database.startswith("NGA"):
                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_2'],axis=1)
                Sa_int = f(self.Tstar[0]); Sa_int.shape = (len(Sa_int),1)
                Sa = np.append(self.database['Sa_2'], Sa_int, axis=1)
                self.database['Sa_2'] = Sa[:,np.argsort(Periods)]  
                
                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_RotD50'],axis=1)
                Sa_int = f(self.Tstar[0]); Sa_int.shape = (len(Sa_int),1)
                Sa = np.append(self.database['Sa_RotD50'], Sa_int, axis=1)
                self.database['Sa_RotD50'] = Sa[:,np.argsort(Periods)]  

            self.database['Periods'] = Periods[np.argsort(Periods)]
            
        # Define the GMPE object and print the required input information
        # More can be added here
        if gmpe == 'Boore_Atkinson_2008':
            self.bgmpe = gsim.boore_atkinson_2008.BooreAtkinson2008()
        if gmpe == 'Boore_EtAl_2014':
            self.bgmpe = gsim.boore_2014.BooreEtAl2014()
        if gmpe == 'Akkar_EtAlRjb_2014':
            self.bgmpe = gsim.akkar_2014.AkkarEtAlRjb2014()
        if gmpe == 'AkkarCagnan2010':            
            self.bgmpe = gsim.akkar_cagnan_2010.AkkarCagnan2010()
        if gmpe == 'ChiouYoungs2008':
            self.bgmpe = gsim.chiou_youngs_2008.ChiouYoungs2008()
        if gmpe == 'CampbellBozorgnia2008':            
            self.bgmpe = gsim.campbell_bozorgnia_2008.CampbellBozorgnia2008()
        if gmpe == 'ChiouYoungs2014':
            self.bgmpe = gsim.chiou_youngs_2014.ChiouYoungs2014()
        if gmpe == 'CampbellBozorgnia2014':            
            self.bgmpe = gsim.campbell_bozorgnia_2014.CampbellBozorgnia2014()
        if gmpe == 'Idriss2014':
            self.bgmpe = gsim.idriss_2014.Idriss2014()
        if gmpe == 'AbrahamsonEtAl2014':
            self.bgmpe = gsim.abrahamson_2014.AbrahamsonEtAl2014()
    
        if pInfo == 1:  # print the selected gmpe info
            print('For the selected gmpe;')
            print('The mandatory input distance parameters are %s' % list(self.bgmpe.REQUIRES_DISTANCES))
            print('The mandatory input rupture parameters are %s' % list(self.bgmpe.REQUIRES_RUPTURE_PARAMETERS))
            print('The mandatory input site parameters are %s' % list(self.bgmpe.REQUIRES_SITES_PARAMETERS))
            print('The defined intensity measure component is %s' % self.bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT)
            print('The defined tectonic region type is %s' % self.bgmpe.DEFINED_FOR_TECTONIC_REGION_TYPE)

    def get_correlation(self, T1,T2):
        """
        Details
        -------
        Compute the inter-period correlation for any two Sa(T) values.
        
        Parameters
        ----------
            T1: int
                First period
            T2: int
                Second period
                
        Returns
        -------
        rho: int
             Predicted correlation coefficient
    
        """
    
        def BakerJayaramCorrelationModel(T1, T2, orth = 0):
            """
            Details
            -------
            Valid for T = 0.01-10sec
        
            References
            ----------
            Baker JW, Jayaram N. Correlation of Spectral Acceleration Values from NGA Ground Motion Models.
            Earthquake Spectra 2008; 24(1): 299–317. DOI: 10.1193/1.2857544.
        
            Parameters
            ----------
                T1: int
                    First period
                T2: int
                    Second period
                orth: int, default is 0
                    1 if the correlation coefficient is computed for the two
                       orthogonal components
        
            Returns
            -------
            rho: int
                 Predicted correlation coefficient
            """
        
            t_min = min(T1, T2)
            t_max = max(T1, T2)
        
            c1 = 1.0 - np.cos(np.pi / 2.0 - np.log(t_max / max(t_min, 0.109)) * 0.366)
        
            if t_max < 0.2:
                c2 = 1.0 - 0.105 * (1.0 - 1.0 / (1.0 + np.exp(100.0 * t_max - 5.0))) * (t_max - t_min) / (t_max - 0.0099)
            else:
                c2 = 0
        
            if t_max < 0.109:
                c3 = c2
            else:
                c3 = c1
        
            c4 = c1 + 0.5 * (np.sqrt(c3) - c3) * (1.0 + np.cos(np.pi * t_min / 0.109))
        
            if t_max <= 0.109:
                rho = c2
            elif t_min > 0.109:
                rho = c1
            elif t_max < 0.2:
                rho = min(c2, c4)
            else:
                rho = c4
            
            if orth:
                rho = rho * (0.79 - 0.023 * np.log(np.sqrt(t_min * t_max)))
        
            return rho
        
        def AkkarCorrelationModel(T1, T2):
            """
            Details
            -------
            Valid for T = 0.01-4sec
            
            References
            ----------
            Akkar S., Sandikkaya MA., Ay BO., 2014, Compatible ground-motion
            prediction equations for damping scaling factors and vertical to
            horizontal spectral amplitude ratios for the broader Europe region,
            Bull Earthquake Eng, 12, pp. 517-547.
        
            Parameters
            ----------
                T1: int
                    First period
                T2: int
                    Second period
        
            :return float:
                The predicted correlation coefficient.
            """
            periods = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14,
                               0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3,
                               0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6,
                               0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.3, 1.4, 1.5,
                               1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4])
        
            if np.any([T1,T2] < periods[0]) or\
                    np.any([T1,T2] > periods[-1]):
                raise ValueError("contains values outside of the "
                                 "range supported by the Akkar et al. (2014) "
                                 "correlation model")
            
            if T1 == T2:
                rho = 1.0
            else:
                with open(os.path.join('Meta_Data','akkar_coeff_table.npy'), 'rb') as f:
                    coeff_table = np.load(f)
                rho = interpolate.interp2d(periods, periods, coeff_table, kind='linear')(T1, T2)[0]
        
            return rho
    
        correlation_function_handles = {
            'baker_jayaram': BakerJayaramCorrelationModel,
            'akkar': AkkarCorrelationModel,
        }
    
        # Check for existing correlation function
        if self.corr_func not in correlation_function_handles:
            raise ValueError('Not a valid correlation function')
        else:
            rho = \
                correlation_function_handles[self.corr_func](T1,T2)
                
        return rho

    def create(self, site_param = {'vs30': 520}, rup_param = {'rake': 0.0, 'mag': [7.2, 6.5]}, 
               dist_param = {'rjb': [20, 5]}, Hcont=[0.6,0.4], T_Tgt_range  = [0.01,4], 
               im_Tstar = 1.0, epsilon = None, cond = 1, useVar = 1, corr_func= 'baker_jayaram',
               outdir = 'Outputs'):
        """
        Details
        -------
        Creates the target spectrum (conditional or unconditional).
    
        Notes
        -----
        Requires libraries from openquake.
        
        References
        ----------
        Baker JW. Conditional Mean Spectrum: Tool for Ground-Motion Selection.
        Journal of Structural Engineering 2011; 137(3): 322–331.
        DOI: 10.1061/(ASCE)ST.1943-541X.0000215.
        Kohrangi, M., Bazzurro, P., Vamvatsikos, D., and Spillatura, A.
        Conditional spectrum-based ground motion record selection using average 
        spectral acceleration. Earthquake Engineering & Structural Dynamics, 
        2017, 46(10): 1667–1685.

        Parameters
        ----------
        site_param : dictionary
            Contains required site parameters to define target spectrum.
        rup_param  : dictionary
            Contains required rupture parameters to define target spectrum.
        dist_param : dictionary
            Contains required distance parameters to define target spectrum.
        Hcont      : list, optional, the default is None.
            Hazard contribution for considered scenarios. 
            If None hazard contribution is the same for all scenarios.
        im_Tstar   : int, float, optional, the default is 1.
            Conditioning intensity measure level [g] (conditional selection)
        epsilon    : list, optional, the default is None.
            Epsilon values for considered scenarios (conditional selection)
        T_Tgt_range: list, optional, the default is [0.01,4].
            Lower and upper bound values for the period range of target spectrum.
        cond       : int, optional
            0 to run unconditional selection
            1 to run conditional selection
        useVar     : int, optional, the default is 1.
            0 not to use variance in target spectrum
            1 to use variance in target spectrum
        corr_func: str, optional, the default is baker_jayaram
            correlation model to use "baker_jayaram","akkar"
        outdir     : str, optional, the default is 'Outputs'.
            output directory to create.

        Returns
        -------
        None.                    
        """
        
        def Sa_avg(bgmpe,scenario,T):
            """
            Details
            -------
            GMPM of average spectral acceleration
            The code will get as input the selected periods, 
            Magnitude, distance and all other parameters of the 
            selected GMPM and 
            will return the median and logarithmic Spectral 
            acceleration of the product of the Spectral
            accelerations at selected periods; 
        
            Parameters
            ----------
            bgmpe : openquake object
                the openquake gmpe object.
            scenario : list
                [sites, rup, dists] source, distance and site context 
                of openquake gmpe object for the specified scenario.
            T : numpy.ndarray
                Array of interested Periods (sec).
        
            Returns
            -------
            Sa : numpy.ndarray
                Mean of logarithmic average spectral acceleration prediction.
            sigma : numpy.ndarray
               logarithmic standard deviation of average spectral acceleration prediction.
            """
            
            n = len(T);
            mu_lnSaTstar = np.zeros(n)
            sigma_lnSaTstar = np.zeros(n)
            MoC = np.zeros((n,n))
            # Get the GMPE output
            for i in range(n):
                mu_lnSaTstar[i], stddvs_lnSaTstar = bgmpe.get_mean_and_stddevs(scenario[0], scenario[1], scenario[2], imt.SA(period=T[i]),[const.StdDev.TOTAL])
                # convert to sigma_arb
                # One should uncomment this line if the arbitary component is used for
                # record selection.
                # ro_xy = 0.79-0.23*np.log(T[k])
                ro_xy = 1
                sigma_lnSaTstar[i] = np.log(((np.exp(stddvs_lnSaTstar[0][0])**2)*(2/(1+ro_xy)))**0.5)
                
                for j in range(n):
                    rho = self.get_correlation(T[i], T[j])
                    MoC [i,j] = rho
        
            SPa_avg_meanLn = (1/n) *sum(mu_lnSaTstar) # logarithmic mean of Sa,avg
        
            SPa_avg_std = 0
            for i in range(n):
                for j in range(n):
                    SPa_avg_std = SPa_avg_std  + (MoC[i,j] *sigma_lnSaTstar[i]*sigma_lnSaTstar[j]) # logarithmic Var of the Sa,avg
        
            SPa_avg_std = SPa_avg_std*(1/n)**2
            # compute mean of logarithmic average spectral acceleration
            # and logarithmic standard deviation of 
            # spectral acceleration prediction
            Sa    = SPa_avg_meanLn
            sigma = np.sqrt(SPa_avg_std)
            return Sa, sigma
        
        def rho_AvgSA_SA(bgmpe,scenario,T,Tstar):
            """
            Details
            -------  
            Function to compute the correlation between Spectra acceleration and AvgSA.
            
            Parameters
            ----------
            bgmpe : openquake object
                the openquake gmpe object.
            scenario : list
                [sites, rup, dists] source, distance and site context 
                of openquake gmpe object for the specified scenario.
            T     : numpy.ndarray
                Array of interested period to calculate correlation coefficient.
        
            Tstar : numpy.ndarray
                Period range where AvgSa is calculated.
        
            Returns
            -------
            rho : int
                Predicted correlation coefficient.
            """
            
            rho=0
            for j in range(len(Tstar)):
                rho_bj = self.get_correlation(T, Tstar[j])
                _, sig1 = bgmpe.get_mean_and_stddevs(scenario[0], scenario[1], scenario[2], imt.SA(period=Tstar[j]),[const.StdDev.TOTAL])
                rho = rho_bj*sig1[0][0] + rho
        
            _, Avg_sig = Sa_avg(bgmpe,scenario,Tstar)
            rho = rho/(len(Tstar)*Avg_sig)
            return rho

        # create the output directory and add the path to self
        cwd = os. getcwd()
        outdir_path = os.path.join(cwd,outdir)
        self.outdir = outdir_path
        create_outdir(self.outdir)
        
        # add target spectrum settings to self
        self.cond = cond
        self.useVar = useVar
        self.corr_func = corr_func
        
        if cond == 0: # there is no conditioning period
            del self.Tstar
            
        # Get number of scenarios, and their contribution
        nScenarios = len(rup_param['mag'])
        if Hcont is None:
            self.Hcont = [1/nScenarios for _ in range(nScenarios)]
        else:
            self.Hcont = Hcont
        
        # Period range of the target spectrum
        temp = np.abs(self.database['Periods'] - np.min(T_Tgt_range))
        idx1 = np.where(temp==np.min(temp))[0][0]
        temp = np.abs(self.database['Periods'] - np.max(T_Tgt_range))
        idx2 = np.where(temp==np.min(temp))[0][0]
        T_Tgt = self.database['Periods'][idx1:idx2+1]

        # Get number of scenarios, and their contribution
        Hcont_mat = np.matlib.repmat(np.asarray(self.Hcont),len(T_Tgt),1)
        
        # Conditional spectrum, log parameters
        TgtMean = np.zeros((len(T_Tgt),nScenarios))

        # Covariance
        TgtCov = np.zeros((nScenarios,len(T_Tgt),len(T_Tgt)))
        
        for n in range(nScenarios):

            # gmpe spectral values
            mu_lnSaT = np.zeros(len(T_Tgt))
            sigma_lnSaT = np.zeros(len(T_Tgt))
            
            # correlation coefficients
            rho_T_Tstar = np.zeros(len(T_Tgt))

            # Covariance
            Cov = np.zeros((len(T_Tgt),len(T_Tgt)))
            
            # Set the contexts for the scenario
            sites = gsim.base.SitesContext()
            for key in site_param.keys():
                temp = np.array([site_param[key]])
                setattr(sites, key, temp)
            
            rup = gsim.base.RuptureContext()
            for key in rup_param.keys():
                if key == 'mag':
                    temp = np.array([rup_param[key][n]])
                else:                     
                    # temp = np.array([rup_param[key]])
                    temp = rup_param[key]
                setattr(rup, key, temp)
                    
            dists = gsim.base.DistancesContext()
            for key in dist_param.keys():
                if key == 'rjb':
                    temp = np.array([dist_param[key][n]])
                else:                     
                    temp = np.array([dist_param[key]]) 
                setattr(dists, key, temp)
                
            scenario = [sites,rup,dists]
        
            for i in range(len(T_Tgt)):
                # Get the GMPE ouput for a rupture scenario
                mu0, sigma0 = self.bgmpe.get_mean_and_stddevs(sites, rup, dists, imt.SA(period=T_Tgt[i]), [const.StdDev.TOTAL])
                mu_lnSaT[i] = mu0[0]
                sigma_lnSaT[i] = sigma0[0][0]
                
                if self.cond == 1:
                    # Compute the correlations between each T and Tstar
                    rho_T_Tstar[i] = rho_AvgSA_SA(self.bgmpe,scenario,T_Tgt[i],self.Tstar)
            
            if self.cond == 1:
                # Get the GMPE output and calculate Avg_Sa_Tstar
                mu_lnSaTstar,sigma_lnSaTstar = Sa_avg(self.bgmpe,scenario,self.Tstar)
                
                if epsilon is None:
                    # Back calculate epsilon
                    rup_eps = (np.log(im_Tstar) - mu_lnSaTstar) / sigma_lnSaTstar
                else:
                    rup_eps = epsilon[n]
                    
                # Get the value of the ln(CMS), conditioned on T_star
                TgtMean[:,n] = mu_lnSaT + rho_T_Tstar * rup_eps * sigma_lnSaT
                
            elif self.cond == 0:
                TgtMean[:,n] = mu_lnSaT 
        
            for i in range(len(T_Tgt)):
                for j in range(len(T_Tgt)):
                    
                    var1 = sigma_lnSaT[i] ** 2
                    var2 = sigma_lnSaT[j] ** 2
                    # using Baker & Jayaram 2008 as correlation model
                    sigma_Corr = self.get_correlation(T_Tgt[i], T_Tgt[j]) * np.sqrt(var1 * var2)
                    
                    if self.cond == 1:
                        varTstar = sigma_lnSaTstar ** 2
                        sigma11 = np.matrix([[var1, sigma_Corr], [sigma_Corr, var2]])
                        sigma22 = np.array([varTstar])
                        sigma12 = np.array([rho_T_Tstar[i] * np.sqrt(var1 * varTstar), rho_T_Tstar[j] * np.sqrt(varTstar * var2)])
                        sigma12.shape = (2,1); sigma22.shape = (1,1)
                        sigma_cond = sigma11 - sigma12 * 1. / (sigma22) * sigma12.T
                        Cov[i, j] = sigma_cond[0, 1]
                        
                    elif self.cond == 0:
                        Cov[i, j] = sigma_Corr

            # Get the value of standard deviation of target spectrum
            TgtCov[n,:,:] = Cov

        # over-write coveriance matrix with zeros if no variance is desired in the ground motion selection
        if self.useVar == 0:
            TgtCov = np.zeros(TgtCov.shape)
            
        TgtMean_fin = np.sum(TgtMean*Hcont_mat,1)
        # all 2D matrices are the same for each kk scenario, since sigma is only T dependent
        TgtCov_fin = TgtCov[0,:,:]
        Cov_elms = np.zeros((len(T_Tgt),nScenarios))
        for ii in range(len(T_Tgt)):
            for kk in range(nScenarios):
                # Hcont[kk] = contribution of the k-th scenario
                Cov_elms[ii,kk] = (TgtCov[kk,ii,ii]+(TgtMean[ii,kk]-TgtMean_fin[ii])**2) * self.Hcont[kk] 

        cov_diag=np.sum(Cov_elms,1)
        TgtCov_fin[np.eye(len(T_Tgt))==1] = cov_diag

        # Find covariance values of zero and set them to a small number so that
        # random number generation can be performed
        TgtCov_fin[np.abs(TgtCov_fin)<1e-10] = 1e-10
        
        TgtSigma_fin = np.sqrt(np.diagonal(TgtCov_fin))
        TgtSigma_fin[np.isnan(TgtSigma_fin)] = 0

        # Add target spectrum to self
        self.mu_ln = TgtMean_fin
        self.sigma_ln = TgtSigma_fin
        self.T = T_Tgt
        self.cov = TgtCov_fin

        if cond == 1:
            # add intensity measure level to self
            if epsilon is None:
                self.im_Tstar = im_Tstar
            else:
                f = interpolate.interp1d(self.T,np.exp(self.mu_ln)) 
                Sa_int = f(self.Tstar)
                self.im_Tstar = np.exp(np.sum(np.log(Sa_int))/len(self.Tstar))
                self.epsilon = epsilon
        
        print('Target spectrum is created.')

    def simulate_spectra(self):
        """
        Details
        -------
        Generates simulated response spectra with best matches to the target values.

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        
        # Set initial seed for simulation
        if self.seedValue != 0:
            np.random.seed(0)
        else:
            np.random.seed(sum(gmtime()[:6]))

        devTotalSim = np.zeros((self.nTrials,1))
        specDict = {}
        nT = len(self.T)
        # Generate simulated response spectra with best matches to the target values
        for j in range(self.nTrials):
            specDict[j] = np.zeros((self.nGM,nT))
            for i in range(self.nGM):
                # Note: we may use latin hypercube sampling here instead. I leave it as Monte Carlo for now
                specDict[j][i,:] = np.exp(np.random.multivariate_normal(self.mu_ln,self.cov))
                
            devMeanSim = np.mean(np.log(specDict[j]), axis = 0) - self.mu_ln # how close is the mean of the spectra to the target
            devSigSim = np.std(np.log(specDict[j]), axis=0) -  self.sigma_ln # how close is the mean of the spectra to the target
            devSkewSim = skew(np.log(specDict[j]), axis=0)                   # how close is the skewness of the spectra to zero (i.e., the target)
            
            devTotalSim[j] = self.weights[0] * np.sum(devMeanSim**2)  + \
                             self.weights[1] * np.sum(devSigSim**2)   + \
                             0.1 * (self.weights[2]) * np.sum(devSkewSim**2) # combine the three error metrics to compute a total error

        recUse = np.argmin(np.abs(devTotalSim))   # find the simulated spectra that best match the targets 
        self.sim_spec = np.log(specDict[recUse])  # return the best set of simulations

    def search_database(self):
        """
        Details
        -------
        Search the database and does the filtering.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        sampleBig : numpy.ndarray
            An array which contains the IMLs from filtered database.
        soil_Vs30 : numpy.ndarray
            An array which contains the Vs30s from filtered database.
        magnitude : numpy.ndarray
            An array which contains the magnitudes from filtered database.
        Rjb : numpy.ndarray
            An array which contains the Rjbs from filtered database.
        mechanism : numpy.ndarray
            An array which contains the fault type info from filtered database.
        Filename_1 : numpy.ndarray
            An array which contains the filename of 1st gm component from filtered database.
            If selection is set to 1, it will include filenames of both components.
        Filename_2 : numpy.ndarray
            An array which contains the filenameof 2nd gm component filtered database.
            If selection is set to 1, it will be None value.
        NGA_num : numpy.ndarray
            If NGA_W2 is used as record database, record sequence numbers from filtered
            database will be saved, for other databases this variable is None.
        """
        
        if self.selection == 1: # SaKnown = Sa_arb

            if self.database['Name'] == "NGA_W2":
            
                SaKnown    = np.append(self.database['Sa_1'],self.database['Sa_2'], axis=0)
                soil_Vs30  = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
                Mw         = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)   
                Rjb        = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)  
                fault      = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
                Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
                NGA_num = np.append(self.database['NGA_num'],self.database['NGA_num'], axis=0)
            
            elif self.database['Name'].startswith("EXSIM"):
                SaKnown    = self.database['Sa_1']
                soil_Vs30  = self.database['soil_Vs30']
                Mw         = self.database['magnitude']
                Rjb        = self.database['Rjb']
                fault      = self.database['mechanism']
                Filename_1 = self.database['Filename_1']
    
        if self.selection == 2: # SaKnown = Sa_g.m. or RotD50
            if self.Sa_def == 'GeoMean':
                SaKnown = np.sqrt(self.database['Sa_1']*self.database['Sa_2'])
            elif 'RotD50': # SaKnown = Sa_RotD50.
                SaKnown = self.database['Sa_RotD50']
            else:
                print('Unexpected Sa definition, exiting...')
                sys.exit()

            soil_Vs30  = self.database['soil_Vs30']
            Mw         = self.database['magnitude']
            Rjb        = self.database['Rjb']
            fault      = self.database['mechanism']
            Filename_1 = self.database['Filename_1']
            Filename_2 = self.database['Filename_2']
            NGA_num = self.database['NGA_num']
                
        perKnown = self.database['Periods']        
        
        # Limiting the records to be considered using the `notAllowed' variable
        # Sa cannot be negative or zero, remove these.
        notAllowed = np.unique(np.where(SaKnown <= 0)[0]).tolist()        
            
        if not self.Vs30_lim is None: # limiting values on soil exist
            mask = (soil_Vs30 > min(self.Vs30_lim)) * (soil_Vs30 < max(self.Vs30_lim) * np.invert(np.isnan(soil_Vs30)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if not self.Mw_lim is None: # limiting values on magnitude exist
            mask = (Mw > min(self.Mw_lim)) * (Mw < max(self.Mw_lim) * np.invert(np.isnan(Mw)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if not self.Rjb_lim is None: # limiting values on Rjb exist
            mask = (Rjb > min(self.Rjb_lim)) * (Rjb < max(self.Rjb_lim) * np.invert(np.isnan(Rjb)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)
            
        if not self.fault_lim is None: # limiting values on mechanism exist
            mask = (fault == self.fault_lim * np.invert(np.isnan(fault)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)
        
        # get the unique values
        notAllowed = (list(set(notAllowed)))
        Allowed = [i for i in range(SaKnown.shape[0])]
        for i in notAllowed:
            Allowed.remove(i)
            
        # Use only allowed records
        SaKnown    = SaKnown[Allowed,:]
        soil_Vs30  = soil_Vs30[Allowed]
        Mw         = Mw[Allowed]
        Rjb        = Rjb[Allowed]
        fault      = fault[Allowed]
        Filename_1 = Filename_1[Allowed]
        
        if self.selection == 1:
            Filename_2 = None
        else:
            Filename_2 = Filename_2[Allowed]

        if self.database['Name'] == "NGA_W2":
            NGA_num    = NGA_num[Allowed]
        else:
            NGA_num = None
        
        # Arrange the available spectra in a usable format and check for invalid input
        # Match periods (known periods and periods for error computations)
        recPer = []
        for i in range(len(self.T)):
            recPer.append(np.where(perKnown == self.T[i])[0][0])
        
        # Check for invalid input
        sampleBig = SaKnown[:,recPer]
        if np.any(np.isnan(sampleBig)):
            print('NaNs found in input response spectra')
            sys.exit()
            
        return sampleBig, soil_Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num
        
    def select(self, nGM=30, selection=1, Sa_def='RotD50', isScaled = 1, maxScale = 4,
               Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None,
               nTrials = 20,  weights = [1,2,0.3], seedValue  = 0, 
               nLoop = 2, penalty = 0, tol = 10):
        """
        Details
        -------
        Perform the ground motion selection.
        
        References
        ----------
        Jayaram, N., Lin, T., and Baker, J. W. (2011). 
        A computationally efficient ground-motion selection algorithm for 
        matching a target response spectrum mean and variance.
        Earthquake Spectra, 27(3), 797-815.
        
        
        Parameters
        ----------
        nGM : int, optional, the default is 30.
            Number of ground motions to be selected.
        selection : int, optional, The default is 1.
            1 for single-component selection and arbitrary component sigma.
            2 for two-component selection and average component sigma. 
        Sa_def : str, optional, the default is 'RotD50'.
            The spectra definition. Necessary if selection = 2.
            'GeoMean' or 'RotD50'. 
        isScaled : int, optional, the default is 1.
            0 not to allow use of amplitude scaling for spectral matching.
            1 to allow use of amplitude scaling for spectral matching.
        maxScale : float, optional, the default is 4.
            The maximum allowable scale factor
        Mw_lim : list, optional, the default is None.
            The limiting values on magnitude. 
        Vs30_lim : list, optional, the default is None.
            The limiting values on Vs30. 
        Rjb_lim : list, optional, the default is None.
            The limiting values on Rjb. 
        mechanism_lim : int, optional, the default is None.
            The limiting fault mechanism. 
            0 for unspecified fault 
            1 for strike-slip fault
            2 for normal fault
            3 for reverse fault
        seedValue  : int, optional, the default is 0.
            For repeatability. For a particular seedValue not equal to
            zero, the code will output the same set of ground motions.
            The set will change when the seedValue changes. If set to
            zero, the code randomizes the algorithm and different sets of
            ground motions (satisfying the target mean and variance) are
            generated each time.
        weights : numpy.ndarray or list, optional, the default is [1,2,0.3].
            Weights for error in mean, standard deviation and skewness
        nTrials : int, optional, the default is 20.
            nTrials sets of response spectra are simulated and the best set (in terms of
            matching means, variances and skewness is chosen as the seed). The user
            can also optionally rerun this segment multiple times before deciding to
            proceed with the rest of the algorithm. It is to be noted, however, that
            the greedy improvement technique significantly improves the match between
            the means and the variances subsequently.
        nLoop   : int, optional, the default is 2.
            Number of loops of optimization to perform.
        penalty : int, optional, the default is 0.
            > 0 to penalize selected spectra more than 
            3 sigma from the target at any period, = 0 otherwise.
        tol     : int, optional, the default is 10.
            Tolerable percent error to skip optimization 

        Returns
        -------
        None.
        """
        
        # Add selection settings to self
        self.nGM = nGM
        self.selection = selection
        self.Sa_def = Sa_def
        self.isScaled = isScaled
        self.Mw_lim = Mw_lim
        self.Vs30_lim = Vs30_lim
        self.Rjb_lim = Rjb_lim
        self.fault_lim = fault_lim
        self.seedValue = seedValue
        self.weights = weights
        self.nTrials = nTrials
        self.maxScale = maxScale
        self.nLoop = nLoop
        self.tol = tol
        self.penalty = penalty
        
        # Exsim provides a single gm component
        if self.database['Name'].startswith("EXSIM"):
            print('Warning! Selection = 1 for this database')
            self.selection = 1
        
        # Simulate response spectra
        self.simulate_spectra() 
        
        # Search the database and filter
        sampleBig, Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num = self.search_database()
        
        # Processing available spectra
        sampleBig = np.log(sampleBig)
        nBig = sampleBig.shape[0]
        
        # Find best matches to the simulated spectra from ground-motion database
        recID = np.ones((self.nGM), dtype = int)*(-1)
        finalScaleFac = np.ones((self.nGM))
        sampleSmall = np.ones((self.nGM,sampleBig.shape[1]))
        
        if self.cond == 1 and self.isScaled == 1:
            # Calculate IMLs for the sample
            f = interpolate.interp1d(self.T,np.exp(sampleBig),axis=1)
            sampleBig_imls = np.exp(np.sum(np.log(f(self.Tstar)),axis=1)/len(self.Tstar))
                
        if self.cond == 1 and len(self.Tstar) == 1:
            # These indices are required in case IM = Sa(T) to break the loop
            ind2 = (np.where(self.T != self.Tstar[0])[0][0]).tolist()
            
        # Find nGM ground motions, inital subset
        for i in range(self.nGM):
            err = np.zeros((nBig))
            scaleFac = np.ones((nBig))
            
            # From the sample of ground motions
            for j in range(nBig):
                
                if self.isScaled == 1: # Calculate scaling facator
                
                    if self.cond == 1: # Calculate using conditioning IML
                        scaleFac[j] = self.im_Tstar/sampleBig_imls[j]
                        
                    elif self.cond == 0: # Calculate using minimization of mean squared root error
                        scaleFac[j] = np.sum(np.exp(sampleBig[j,:])*np.exp(self.sim_spec[i,:]))/np.sum(np.exp(sampleBig[j,:])**2)

                # check if scaling factor is greater than the limit                
                # check if this record have already been selected
                if np.any(recID == j) or scaleFac[j] > self.maxScale:
                    err[j] = 1000000
                else: # calculate the error
                    err[j] = np.sum((np.log(np.exp(sampleBig[j,:])*scaleFac[j]) - self.sim_spec[i,:])**2)
                    
            recID[i] = int(np.argsort(err)[0])    
            if err.min() >= 1000000:
                print('Warning: Possible problem with simulated spectrum. No good matches found')
                print(recID[i])
                sys.exit()

            if self.isScaled == 1:
                finalScaleFac[i] = scaleFac[recID[i]]
            
            # Save the selected spectra
            sampleSmall[i,:] = np.log(np.exp(sampleBig[recID[i],:])*finalScaleFac[i])
            
        # Optimizing the selected subset of ground motions
        # Currently only option is to use Greedy subset modification procedure
        # This seems to be sufficient enough
        @njit
        def penalize(devTotal,sampleSmall,mu_ln,sigma_ln,nGM,penalty):
            """
            This function is used to penalize the bad spectra
            njit speeds up the computation
            """
            for m in range(nGM):
                devTotal += np.sum(np.abs(np.exp(sampleSmall[m,:]) > np.exp(mu_ln + 3*sigma_ln))) * penalty
            return devTotal

        for k in range(self.nLoop): # Number of passes
            
            for i in range(self.nGM): # Loop for nGM
                
                minDev = 100000
                scaleFac = np.ones((nBig))
                sampleSmall = np.delete(sampleSmall, i, 0)
                recID = np.delete(recID, i)
                
                # Try to add a new spectra to the subset list
                for j in range(nBig):  
                    
                    # Calculate the scaling factor
                    if self.isScaled == 1: 
                        
                        # using conditioning IML
                        if self.cond == 1: 
                            scaleFac[j] = self.im_Tstar/sampleBig_imls[j]
                            
                        # using minimization of mean squared root error
                        elif self.cond == 0: 
                            scaleFac[j] = np.sum(np.exp(sampleBig[j,:])*np.exp(self.sim_spec[i,:]))/np.sum(np.exp(sampleBig[j,:])**2)
                    
                    # Add to the sample the scaled spectra
                    sampleSmall = np.concatenate((sampleSmall,sampleBig[j,:].reshape(1,sampleBig.shape[1]) + np.log(scaleFac[j])),axis=0)

                    # Greedy subset modification procedure
                    devMean = np.mean(sampleSmall,axis=0) - self.mu_ln # Compute deviations from target
                    devSig = np.std(sampleSmall, axis=0) - self.sigma_ln
                    devTotal = weights[0] * np.sum(devMean**2) + weights[1] * np.sum(devSig**2)
                    
                    # Penalize bad spectra (set penalty to zero if this is not required)
                    if self.penalty > 0:
                        # for m in range(sampleSmall.shape[0]):
                        #     devTotal += np.sum(np.abs(np.exp(sampleSmall[m,:]) > np.exp(self.mu_ln + 3*self.sigma_ln))) * penalty
                        devTotal = penalize(devTotal,sampleSmall,self.mu_ln,self.sigma_ln,nGM,penalty)
                    
                    # Check if we exceed the scaling limit
                    if scaleFac[j] > self.maxScale or np.any(np.array(recID) == j):
                        devTotal += 1000000
                    
                    # Should cause improvement and record should not be repeated
                    if devTotal < minDev:
                        minID = j
                        minDev = devTotal
                    
                    # Empty the slot to try a new candidate
                    sampleSmall = np.delete(sampleSmall, -1, 0)
                
                # Add new element in the right slot
                if self.isScaled == 1:
                    finalScaleFac[i] = scaleFac[minID]
                else:
                    finalScaleFac[i] = 1
                sampleSmall = np.concatenate((sampleSmall[:i,:], sampleBig[minID,:].reshape(1,sampleBig.shape[1]) + np.log(scaleFac[minID]), 
                                sampleSmall[i:,:]),axis=0)
                recID = np.concatenate((recID[:i],np.array([minID]),recID[i:]))
            
            # Lets check if the selected ground motions are good enough, if the errors are sufficiently small stop!
            if self.cond == 1 and len(self.Tstar) == 1: # if conditioned on SaT, ignore error at T*
                medianErr = np.max(np.abs(np.exp(np.mean(sampleSmall[:,ind2],axis=0)) - np.exp(self.mu_ln[ind2]))/np.exp(self.mu_ln[ind2]))*100
                stdErr = np.max(np.abs(np.std(sampleSmall[:,ind2], axis=0) - self.sigma_ln[ind2])/self.sigma_ln[ind2])*100  
            else:
                medianErr = np.max(np.abs(np.exp(np.mean(sampleSmall,axis=0)) - np.exp(self.mu_ln))/np.exp(self.mu_ln))*100
                stdErr = np.max(np.abs(np.std(sampleSmall, axis=0) - self.sigma_ln)/self.sigma_ln)*100

            if medianErr < self.tol and stdErr < self.tol:
                break
        print('Ground motion selection is finished.')
        print('For T ∈ [%.2f - %2.f]'% (self.T[0],self.T[-1]))
        print('Max error in median = %.2f %%' % medianErr)
        print('Max error in standard deviation = %.2f %%' % stdErr)
        if medianErr < self.tol and stdErr < self.tol:
            print('The errors are within the target %d percent %%' % self.tol)
            
        recID = recID.tolist()
        # Add selected record information to self
        self.rec_scale = finalScaleFac
        self.rec_spec = sampleSmall
        self.rec_Vs30 = Vs30[recID]
        self.rec_Rjb = Rjb[recID]
        self.rec_Mw = Mw[recID]
        self.rec_fault = fault[recID]
        self.rec_h1 = Filename_1[recID]
        
        if self.selection == 1:
            self.rec_h2 = None
        elif self.selection == 2:
            self.rec_h2 = Filename_2[recID]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recID]
        else:
            self.rec_rsn = None
            
    def write(self, obj = 0, recs = 1, recs_f = ''):
        """
        
        Details
        -------
        Writes the cs_master object, selected and scaled records
        
        Parameters
        ----------
        obj : int, optional
            flag to write the object into the pickle file. The default is 0.
        recs : int, optional
            flag to write the selected and scaled time histories. 
            The default is 1.
        recs_f : str, optional
            This is option could be used if the user already has all the 
            records in database. This is the folder path which contains 
            "database.zip" file. The records must be placed inside
            recs_f/database.zip/database/ 
            The default is ''.

        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.

        """
        
        if recs == 1:
            # set the directories and file names
            zipName = os.path.join(recs_f,self.database['Name'] + '.zip')
            if self.database['Name'] == 'NGA_W2':
                try:
                    zipName = self.Unscaled_rec_file
                except:
                    pass
                
            n = len(self.rec_h1)
            path_dts = os.path.join(self.outdir,'GMR_dts.txt')
            path_durs = os.path.join(self.outdir,'GMR_durs.txt')
            dts = np.zeros((n))
            durs = np.zeros((n))

            if not self.rec_h2 is None:
                path_H1 = os.path.join(self.outdir,'GMR_H1_names.txt')
                path_H2 = os.path.join(self.outdir,'GMR_H2_names.txt')
                h2s = open(path_H2, 'w')
                
            else:
                path_H1 = os.path.join(self.outdir,'GMR_names.txt')
                
            h1s = open(path_H1, 'w')      
            
            if self.database['Name'] == 'NGA_W2':
                
                if zipName != os.path.join(recs_f,self.database['Name'] + '.zip'):
                    rec_paths = self.rec_h1
                else:
                    rec_paths = [self.database['Name']+'/'+self.rec_h1[i] for i in range(n)]
                contents = ContentFromZip(rec_paths,zipName)
                
                # Save the H1 gm components
                for i in range(n):
                    dts[i], _, _, t, inp_acc = ReadNGA(inFilename = self.rec_h1[i],content = contents[i])
                    durs[i] = t[-1]
                    gmr_file = self.rec_h1[i].replace('/','_')[:-4]+'_SF_'+"{:.3f}".format(self.rec_scale[i])+'.txt'
                    path = os.path.join(self.outdir,gmr_file)
                    acc_Sc = self.rec_scale[i] * inp_acc
                    np.savetxt(path, acc_Sc, fmt='%1.4e')
                    h1s.write(gmr_file+'\n')
                
                # Save the H2 gm components
                if not self.rec_h2 is None:
                    
                    if zipName != os.path.join(recs_f,self.database['Name'] + '.zip'):
                        rec_paths = self.rec_h2
                    else:
                        rec_paths = [self.database['Name']+'/'+self.rec_h2[i] for i in range(n)]
                        
                    contents = ContentFromZip(rec_paths,zipName)
                    for i in range(n):
                        _, _, _, _, inp_acc = ReadNGA(inFilename = self.rec_h2[i],content = contents[i])
                        gmr_file = self.rec_h2[i].replace('/','_')[:-4]+'_SF_'+"{:.3f}".format(self.rec_scale[i])+'.txt'
                        path = os.path.join(self.outdir,gmr_file)
                        acc_Sc = self.rec_scale[i] * inp_acc
                        np.savetxt(path, acc_Sc, fmt='%1.4e')
                        h2s.write(gmr_file+'\n')                
                    
                    h2s.close()
    
            if self.database['Name'].startswith('EXSIM'):
                sf = 1/981 # cm/s**2 to g
                rec_paths = [self.database['Name']+'/'+self.rec_h1[i].split('_acc')[0]+'/' 
                             + self.rec_h1[i] for i in range(n)]               
                contents = ContentFromZip(rec_paths,zipName)
                
                for i in range(n):
                    dts[i], _, _, t, inp_acc = ReadEXSIM(inFilename = self.rec_h1[i],content = contents[i])
                    durs[i] = t[-1]
                    gmr_file = self.rec_h1[i][:-4]+'_SF_'+"{:.3f}".format(self.rec_scale[i])+'.txt'
                    path = os.path.join(self.outdir,gmr_file)
                    acc_Sc = self.rec_scale[i] * inp_acc * sf
                    np.savetxt(path, acc_Sc, fmt='%1.4e')
                    h1s.write(gmr_file+'\n')
                    
            h1s.close()
            np.savetxt(path_dts,dts, fmt='%.5f')
            np.savetxt(path_durs,durs, fmt='%.5f')
            
        if obj == 1:
            # save some info as pickle obj
            path_cs = os.path.join(self.outdir,'CS.pkl')  
            cs_obj = vars(copy.deepcopy(self)) # use copy.deepcopy to create independent obj
            cs_obj['database'] = self.database['Name']
            cs_obj['gmpe'] = str(cs_obj['bgmpe']).replace('[','',).replace(']','')
            del cs_obj['bgmpe'] 
            del cs_obj['outdir']
            with open(path_cs, 'wb') as file:
                pickle.dump(cs_obj, file)
        
        print('Finished writing process, the files are located in\n%s' % self.outdir)
        
    def plot(self, tgt = 0, sim = 0, rec = 1, save = 0, show = 1):
        """
        Details
        -------
        Plots the target spectrum and spectra 
        of selected simulations and records.

        Parameters
        ----------
        tgt    : int, optional
            Flag to plot target spectrum.
            The default is 1.
        sim    : int, optional
            Flag to plot simulated response spectra vs. target spectrum.
            The default is 0.
        rec    : int, optional
            Flag to plot Selected response spectra of selected records
            vs. target spectrum. 
            The default is 1.
        save   : int, optional
            Flag to save plotted figures in pdf format.
            The default is 0.
        show  : int, optional
            Flag to show figures
            The default is 0.
            
        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """

        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIG_SIZE = 14
        BIGGER_SIZE = 18
        
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIG_SIZE)       # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.ioff()

        if self.cond == 1:
            if len(self.Tstar) == 1:
                hatch = [self.Tstar*0.98, self.Tstar*1.02]
            else:
                hatch = [self.Tstar.min(), self.Tstar.max()]

        if tgt == 1:
            # Plot Target spectrum vs. Simulated response spectra
            fig,ax = plt.subplots(1,2, figsize = (16,8))
            plt.suptitle('Target Spectrum', y = 0.95)
            ax[0].loglog(self.T,np.exp(self.mu_ln),color = 'red', lw=2, label='Target - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T,np.exp(self.mu_ln+2*self.sigma_ln),color = 'red', linestyle='--', lw=2, label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T,np.exp(self.mu_ln-2*self.sigma_ln),color = 'red', linestyle='--', lw=2, label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
            
            ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
            ax[0].set_xlabel('Period [sec]')
            ax[0].set_ylabel('Spectral Acceleration [g]')
            ax[0].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(), frameon = False)
            ax[0].set_xlim([self.T[0],self.T[-1]])
            if self.cond == 1:
                ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
                   
            # Sample and target standard deviations
            if self.useVar == 1:
                ax[1].semilogx(self.T,self.sigma_ln,color = 'red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
                ax[1].set_xlabel('Period [sec]')
                ax[1].set_ylabel('Dispersion')
                ax[1].grid(True)
                ax[1].legend(frameon = False)
                ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_xlim([self.T[0],self.T[-1]])
                ax[1].set_ylim(bottom=0)
                if self.cond == 1:
                    ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
            
            if save == 1:
                plt.savefig(os.path.join(self.outdir,'Targeted.pdf'))

        if sim == 1:
            # Plot Target spectrum vs. Simulated response spectra
            fig,ax = plt.subplots(1,2, figsize = (16,8))
            plt.suptitle('Target Spectrum vs. Simulated Spectra', y = 0.95)
            
            for i in range(self.nGM):
                ax[0].loglog(self.T,np.exp(self.sim_spec[i,:]),color = 'gray', lw=1,label='Selected');

            ax[0].loglog(self.T,np.exp(self.mu_ln),color = 'red', lw=2, label='Target - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T,np.exp(self.mu_ln+2*self.sigma_ln),color = 'red', linestyle='--', lw=2, label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T,np.exp(self.mu_ln-2*self.sigma_ln),color = 'red', linestyle='--', lw=2, label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

            ax[0].loglog(self.T,np.exp(np.mean(self.sim_spec,axis=0)),color = 'blue', lw=2, label='Selected - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T,np.exp(np.mean(self.sim_spec,axis=0)+2*np.std(self.sim_spec,axis=0)),color = 'blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T,np.exp(np.mean(self.sim_spec,axis=0)-2*np.std(self.sim_spec,axis=0)),color = 'blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
            
            ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
            ax[0].set_xlabel('Period [sec]')
            ax[0].set_ylabel('Spectral Acceleration [g]')
            ax[0].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(), frameon = False)
            ax[0].set_xlim([self.T[0],self.T[-1]])
            if self.cond == 1:
                ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

            if self.useVar == 1:
                # Sample and target standard deviations
                ax[1].semilogx(self.T,self.sigma_ln,color = 'red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
                ax[1].semilogx(self.T,np.std(self.sim_spec,axis=0),color = 'black', linestyle='--', lw=2, label='Selected - $\sigma_{ln}$')
                ax[1].set_xlabel('Period [sec]')
                ax[1].set_ylabel('Dispersion')
                ax[1].grid(True)
                ax[1].legend(frameon = False)
                ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_xlim([self.T[0],self.T[-1]])
                ax[1].set_ylim(bottom=0)
                if self.cond == 1:
                    ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
            
            if save == 1:
                plt.savefig(os.path.join(self.outdir,'Simulated.pdf'))
            
        if rec == 1:
            # Plot Target spectrum vs. Selected response spectra
            fig,ax = plt.subplots(1,2, figsize = (16,8))
            plt.suptitle('Target Spectrum vs. Spectra of Selected Records', y = 0.95)

            for i in range(self.nGM):
                ax[0].loglog(self.T,np.exp(self.rec_spec[i,:]),color = 'gray', lw=1,label='Selected');

            ax[0].loglog(self.T,np.exp(self.mu_ln),color = 'red', lw=2, label='Target - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T,np.exp(self.mu_ln+2*self.sigma_ln),color = 'red', linestyle='--', lw=2, label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T,np.exp(self.mu_ln-2*self.sigma_ln),color = 'red', linestyle='--', lw=2, label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

            ax[0].loglog(self.T,np.exp(np.mean(self.rec_spec,axis=0)),color = 'blue', lw=2, label='Selected - $e^{\mu_{ln}}$')
            ax[0].loglog(self.T,np.exp(np.mean(self.rec_spec,axis=0)+2*np.std(self.rec_spec,axis=0)),color = 'blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
            ax[0].loglog(self.T,np.exp(np.mean(self.rec_spec,axis=0)-2*np.std(self.rec_spec,axis=0)),color = 'blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
            
            ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
            ax[0].set_xlabel('Period [sec]')
            ax[0].set_ylabel('Spectral Acceleration [g]')
            ax[0].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(), frameon = False)
            ax[0].set_xlim([self.T[0],self.T[-1]])
            if self.cond == 1:
                ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
                   
            # Sample and target standard deviations
            ax[1].semilogx(self.T,self.sigma_ln,color = 'red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
            ax[1].semilogx(self.T,np.std(self.rec_spec,axis=0),color = 'black', linestyle='--', lw=2, label='Selected - $\sigma_{ln}$')
            ax[1].set_xlabel('Period [sec]')
            ax[1].set_ylabel('Dispersion')
            ax[1].grid(True)
            ax[1].legend(frameon = False)
            ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].set_xlim([self.T[0],self.T[-1]])
            ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[1].set_ylim(bottom=0)
            if self.cond == 1:
                ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
            
            if save == 1:
                plt.savefig(os.path.join(self.outdir,'Selected.pdf'))

        # Show the figure
        if show == 1: 
            plt.show()

    def nga_download(self, username , pwd):
        """
        
        Details
        -------
        
        This function has been created as a web automation tool in order to 
        download unscaled record time histories from NGA-West2 Database 
        (https://ngawest2.berkeley.edu/) by Record Sequence Numbers (RSNs).

        Parameters
        ----------
        username     : str
                Account username (e-mail)
                            e.g.: 'username@mail.com'
        pwd          : str
                Account password
                            e.g.: 'password!12345'

        """

        def download_url(url, save_path, chunk_size=128):
            """
            
            Details
            -------
            
            This function downloads file from given url.Herein, it is being used 

            Parameters
            ----------
            url          : str
                    e.g.: 'www.example.com/example_file.pdf'
            save_path    : str
                    Save directory.

            """
            r = requests.get(url, stream=True)
            with open(save_path, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)     

        def find_latest_ver():
            """
            
            Details
            -------
            
            This function finds the latest version of the chrome driver from  
            'https://chromedriver.chromium.org/'.

            """
            r = requests.get('https://chromedriver.chromium.org/')
            a = r.text
            start = a.find('Latest stable release:')
            text = a.replace(a[0:start],'')
            start = text.find('path=')      

            text = text.replace(text[0:start+5],'')
            end = text.find("/")
            latest_ver = text.replace(text[end::],'')
            return latest_ver       

        def add_driver_to_the_PATH(save_path):
            paths = sys.path
            package = [i for i in paths if 'site-packages' in i][0]
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(package)

        def dir_size(Down_Dir):
            total_size = 0
            for path, dirs, files in os.walk(Down_Dir):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            return total_size       

        def download_wait(Down_Dir):
            delta_size = 100
            flag = 0
            flag_lim = 5
            while delta_size > 0 and flag < flag_lim:
                print
                size_0 = dir_size(Down_Dir)
                sleep(6)
                size_1 = dir_size(Down_Dir)
                if size_1-size_0 > 0:
                    delta_size = size_1-size_0
                else:
                    flag += 1
                    print(flag_lim-flag)
            print(f'Downloaded files are located in\n{Down_Dir}')          

        def seek_and_download():
            """
            
            Details
            -------
            
            This function finds the latest version of the chrome driver from  
            'https://chromedriver.chromium.org/' and downloads the compatible
            version to the OS and extract it to the path.

            """
            paths = sys.path
            package = [i for i in paths if 'site-packages' in i][0]
            if sys.platform.startswith('win'):
                current_platform = 'win32'
                aim_driver = 'chromedriver.exe'
            elif sys.platform.startswith('linux'):
                current_platform = 'linux64'
                aim_driver = 'chromedriver'
            elif sys.platform.startswith('darwin'):       
                current_platform = 'mac64'
                aim_driver = 'chromedriver'
            if aim_driver not in os.listdir(package):
                latest_ver = find_latest_ver()
                save_path = os.path.join(os.getcwd(),'chromedriver.zip')
                url = f"https://chromedriver.storage.googleapis.com/{latest_ver}/chromedriver_{current_platform}.zip"
                download_url(url, save_path, chunk_size=128)
                add_driver_to_the_PATH(save_path)
                print ('chromedriver downloaded successfully!!')
                os.remove(save_path)
            else:
                print("chromedriver allready exists!!")

        def go_to_sign_in_page(Download_Dir):
            """
            
            Details
            -------
            
            This function starts the webdriver in headless mode and 
            opens the sign in page to 'https://ngawest2.berkeley.edu/'

            Parameters
            ----------
            Download_Dir     : str
                    Directory for the output time histories to be downloaded

            """

            ChromeOptions = webdriver.ChromeOptions()
            prefs = {"download.default_directory" : Download_Dir}
            ChromeOptions.add_experimental_option("prefs",prefs)
            ChromeOptions.headless = True
            if sys.platform.startswith('win'):
                aim_driver = 'chromedriver.exe'
            elif sys.platform.startswith('linux'):
                aim_driver = 'chromedriver'
            elif sys.platform.startswith('darwin'):   
                aim_driver = 'chromedriver'
            path_of_driver = os.path.join([i for i in sys.path if 'site-packages' in i][0] , aim_driver)
            if not os.path.exists(path_of_driver):
                print('Downloading the chromedriver!!')
                seek_and_download()
                if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                    os.chmod(path_of_driver, 0o777)
            driver = webdriver.Chrome(executable_path = path_of_driver ,options=ChromeOptions)
            url_sign_in = 'https://ngawest2.berkeley.edu/users/sign_in'
            driver.get(url_sign_in)
            return driver       

        def sign_in_with_given_creds(driver,USERNAME,PASSWORD):
            """
            
            Details
            -------
            
            This function signs in to 'https://ngawest2.berkeley.edu/' with
            given account credentials

            Parameters
            ----------
            driver     : selenium webdriver object
                    Please use the driver have been generated as output of 
                    'go_to_sign_in_page' function
            USERNAME   : str
                    Account username (e-mail)
                                e.g.: 'username@mail.com'
            PASSWORD   : str
                    Account password
                                e.g.: 'password!12345' 
            """
            print("Signing in with given account!...")
            driver.find_element_by_id('user_email').send_keys(USERNAME)
            driver.find_element_by_id('user_password').send_keys(PASSWORD)
            driver.find_element_by_id('user_submit').click()
            try:
                alert = driver.find_element_by_css_selector('p.alert')
                warn = alert.text
                print(warn)
            except:
                warn = ''
                pass
            return driver, warn     

        def Download_Given(RSNs,Download_Dir,driver):
            """
            
            Details
            -------
            
            This function dowloads the timehistories which have been indicated with their RSNs
            from 'https://ngawest2.berkeley.edu/'.

            Parameters
            ----------
            RSNs     : str
                    A string variable contains RSNs to be downloaded which uses ',' as delimeter
                    between RNSs
                                e.g.: '1,5,91,35,468'
            Download_Dir     : str
                    Directory for the output timehistories to be downloaded
            driver     : selenium webdriver object
                    Please use the driver have been generated as output of 
                    sign_in_with_given_creds' function

            """
            url_get_record = 'https://ngawest2.berkeley.edu/spectras/new?sourceDb_flag=1'   
            print("Listing the Records!....")
            driver.get(url_get_record)
            sleep(2)
            driver.find_element_by_xpath("//button[@type='button']").submit()
            sleep(2)
            driver.find_element_by_id('search_search_nga_number').send_keys(RSNs)
            sleep(3)
            driver.find_element_by_xpath("//button[@type='button' and @onclick='uncheck_plot_selected();reset_selectedResult();OnSubmit();']").submit()
            try:
                note = driver.find_element_by_id('notice').text
                print(note)
            except:
                note = 'NO'

            if 'NO' in note:
                print("\033[1;31mCould not be able to download records!")
                driver.quit()
                sys.exit()
                pass
            else:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                sleep(3)
                driver.find_element_by_xpath("//button[@type='button' and @onclick='getSelectedResult(true)']").click()
                print("Downloading the Records!...")
                obj = driver.switch_to.alert
                msg=obj.text
                print ("Alert shows following message: "+ msg )
                sleep(5)
                obj.accept()       
                obj = driver.switch_to.alert
                msg=obj.text
                print ("Alert shows following message: "+ msg )
                sleep(3)
                obj.accept()
                print("Downloading the Records!...")  
                download_wait(Download_Dir)
                driver.quit()
        
        if self.database['Name'] == 'NGA_W2':
            print('\nStarted executing nga_download method...')
            self.username = username
            self.pwd = pwd
            driver = go_to_sign_in_page(self.outdir)
            driver,warn = sign_in_with_given_creds(driver,self.username,self.pwd)
            if str(warn) == 'Invalid email or password.':
                print(warn)
                driver.quit()
                sys.exit()
            else:
                RSNs = ''
                for i in self.rec_rsn:
                    RSNs += str(int(i)) + ','
    
                RSNs = RSNs[:-1:]
                files_before_download = set(os.listdir(self.outdir))
                Download_Given(RSNs,self.outdir,driver)
                files_after_download = set(os.listdir(self.outdir))
                Downloaded_File = str(list(files_after_download.difference(files_before_download))[0])
                file_extension = Downloaded_File[Downloaded_File.find('.')::]
                time_tag = gmtime()
                time_tag_str = f'{time_tag[0]}'
                for i in range(1,len(time_tag)):
                    time_tag_str += f'_{time_tag[i]}'
                new_file_name = f'unscaled_records_{time_tag_str}{file_extension}'
                Downloaded_File = os.path.join(self.outdir,Downloaded_File)
                Downloaded_File_Rename = os.path.join(self.outdir,new_file_name)
                os.rename(Downloaded_File,Downloaded_File_Rename)
                self.Unscaled_rec_file = Downloaded_File_Rename
        else: 
            print('You have to use NGA_W2 database to use nga_download method.')

def ContentFromZip(paths,zipName):
    """
    Details
    -------
    This function reads the contents of all selected records
    from the zipfile in which the records are located
    
    Parameters
    ----------
    paths : list
        Containing file list which are going to be read from the zipfile.
    zipName    : str
        Path to the zip file where file lists defined in "paths" are located.

    Returns
    -------
    contents   : dictonary
        Containing raw contents of the files which are read from the zipfile.

    """
    contents = {}
    with zipfile.ZipFile(zipName, 'r') as myzip:
        for i in range(len(paths)):
            with myzip.open(paths[i]) as myfile:
                contents[i] = [x.decode('utf-8') for x in myfile.readlines()]    
                
    return contents

def ReadNGA(inFilename=None, content=None, outFilename=None):
    """
    Details
    -------
    This function process acceleration history for NGA data file (.AT2 format).
    
    Parameters
    ----------
    inFilename : str, optional
        Location and name of the input file.
        The default is None
    content    : str, optional
        Raw content of the .AT2 file.
        The default is None
    outFilename : str, optional
        location and name of the output file. 
        The default is None.

    Notes
    -----
    At least one of the two variables must be defined: inFilename, content

    Returns
    -------
    dt   : float
        time interval of recorded points.
    npts : int
        number of points in ground motion record file.
    desc : str
        Description of the earthquake (e.g., name, year, etc).
    t    : numpy.array (n x 1)
        time array, same length with npts.
    acc  : numpy.array (n x 1)
        acceleration array, same length with time unit 
        usually in (g) unless stated as other.
    """

    try:
        # Read the file content from inFilename
        if content is None:
            with open(inFilename,'r') as inFileID:
                content = inFileID.readlines()
    
        # check the first line
        temp = str(content[0]).split()
        try:
            # description is in the end
            float(temp[0])
            flag = 1
        except:
            # description is in the begining
            flag = 0
    
        counter = 0
        desc, row4Val, acc_data = "","",[]
    
        if flag == 1:
            for x in content:
                if counter == len(content)-3:
                    desc = x
                elif counter == len(content)-1:
                    row4Val = x
                    if row4Val[0][0] == 'N':
                        val = row4Val.split()
                        npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                        try:
                            dt = float(val[(val.index('DT='))+1])
                        except:
                            dt = float(val[(val.index('DT='))+1].replace('SEC,',''))
                    else:
                        val = row4Val.split()
                        npts = float(val[0])
                        dt = float(val[1])
    
                elif counter < len(content)-4:
                    data = str(x).split()
                    for value in data:
                        a = float(value)
                        acc_data.append(a)
                    acc = np.asarray(acc_data)
                counter = counter + 1
    
        if flag == 0:
            for x in content:
                if counter == 1:
                    desc = x
                elif counter == 3:
                    row4Val = x
                    if row4Val[0][0] == 'N':
                        val = row4Val.split()
                        npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                        try:
                            dt = float(val[(val.index('DT='))+1])
                        except:
                            dt = float(val[(val.index('DT='))+1].replace('SEC,',''))
                    else:
                        val = row4Val.split()
                        npts = float(val[0])
                        dt = float(val[1])
    
                elif counter > 3:
                    data = str(x).split()
                    for value in data:
                        a = float(value)
                        acc_data.append(a)
                    acc = np.asarray(acc_data)
                counter = counter + 1
    
        t = [] # save time history
        for i in range (0,len(acc_data)):
            ti = i * dt
            t.append(ti)
    
        if outFilename is not None:
            np.savetxt(outFilename, acc, fmt='%1.4e')
    
        npts = int(npts)
        return dt, npts, desc, t, acc
    
    except:
        print("processMotion FAILED!: The record file is not in the directory")
        print(inFilename)

def ReadEXSIM(inFilename=None, content=None, outFilename=None):
    """
    Details
    -------
    This function process acceleration history for EXSIM data file.
    
    Parameters
    ----------
    inFilename : str, optional
        Location and name of the input file.
        The default is None
    content    : str, optional
        Raw content of the exsim record file.
        The default is None
    outFilename : str, optional
        location and name of the output file. 
        The default is None.

    Returns
    -------
    dt   : float
        time interval of recorded points.
    npts : int
        number of points in ground motion record file.
    desc : str
        Description of the earthquake (e.g., name, year, etc).
    time : numpy.array (n x 1)
        time array, same length with npts.
    acc  : numpy.array (n x 1)
        acceleration array, same length with time unit 
        usually in (g) unless stated as other.
    """
    
    try:
        # Read the file content from inFilename
        if content is None:
            with open(inFilename,'r') as inFileID:
                content = inFileID.readlines()
    
        desc = content[:12]
        dt = float(content[6].split()[1])
        npts = int(content[5].split()[0])
        acc = []
        t = []
        
        for i in range(12,len(content)):
            temp = content[i].split()
            acc.append(float(temp[1]))
        
        acc = np.asarray(acc)
        if len(acc) < 20000:
            acc = acc[2500:10800] # get rid of zeros
        dur = len(acc)*dt
        t = np.arange(0,dur,dt)

        if outFilename is not None:
            np.savetxt(outFilename, acc, fmt='%1.4e')

        return dt, npts, desc, t, acc
    
    except:
        print("processMotion FAILED!: The record file is not in the directory")
        print(inFilename)
    
def create_outdir(outdir_path):
    """  
    Parameters
    ----------
    outdir_path : str
        output directory to create.

    Returns
    -------
    None.
    """
    shutil.rmtree(outdir_path, ignore_errors=True)
    os.makedirs(outdir_path)
    
def RunTime(startTime):
    """
    Details
    -------
    Prints the time passed between startTime and Finishtime (now)
    in hours, minutes, seconds. startTime is a global variable.

    Parameters
    ----------
    startTime : The initial time obtained via time()

    Returns
    -------
    None.
    """
    finishTime = time()
    # Procedure to obtained elapsed time in Hr, Min, and Sec
    timeSeconds = finishTime-startTime
    timeMinutes = int(timeSeconds/60);
    timeHours = int(timeSeconds/3600);
    timeMinutes = int(timeMinutes - timeHours*60)
    timeSeconds = timeSeconds - timeMinutes*60 - timeHours*3600
    print("Run time: %d hours: %d minutes: %.2f seconds"  % (timeHours, timeMinutes, timeSeconds))
    
def baseline_correction(values,dt,polynomial_type):
    """
    Details
    -------
    This script will return baseline corrected values for the given signal
    
    Notes
    -----
    Applicable for Constant, Linear, Quadratic and Cubic polynomial functions
        
    References
    ----------
    Kramer, Steven L. 1996. Geotechnical Earthquake Engineering. Prentice Hall
        
    Parameters
    ----------
    values: numpy.ndarray
        signal values      
    dt: float          
        sampling interval
    polynomal_type: str
        type of baseline correction 'Constant', 'Linear', 'Quadratic', 'Cubic'    
        
    Returns
    -------
    values_corrected: numpy.ndarray
        corrected values
        
    """   
        
    if polynomial_type == 'Constant':       n = 0
    elif polynomial_type == 'Linear':       n = 1
    elif polynomial_type == 'Quadratic':    n = 2
    elif polynomial_type == 'Cubic':        n = 3
    
    time = np.linspace(0,(len(values)-1)*dt,len(values))    # Time array
    P = np.polyfit(time,values,n);                          # Best fit line of values
    po_va = np.polyval(P,time);                             # Matrix of best fit line
    values_corrected = values - po_va;                      # Baseline corrected values
    
    return values_corrected

def butterworth_filter(values,dt, cut_off=(0.1, 25), **kwargs):
    """
    Details
    -------
    This script will return filtered values for the given signal
    
    References
    ----------
    Kramer, Steven L. 1996. Geotechnical Earthquake Engineering, Prentice Hall
        
    Parameters
    ----------
    values: numpy.ndarray
        Input signal
    cut_off: tuple, list, optional          
        Lower and upper cut off frequencies for the filter, if None then no filter. 
        e.g. (None, 15) applies a lowpass filter at 15Hz, whereas (0.1, 10) applies
        a bandpass filter at 0.1Hz to 10Hz.
    filter_order: int        
        Order of the Butterworth filter (default = 4)
    remove_gibbs: str, the default is None.
        Pads with zeros to remove the Gibbs filter effect
        if = 'start' then pads at start,
        if = 'end' then pads at end,
        if = 'mid' then pads half at start and half at end
        
    kwargs: keyword arguments, optional
        gibbs_extra: int, the default is 1
            Each increment of the value doubles the record length using zero padding.
        gibbs_range: int, the default is 50
            gibbs index range
    Returns
    -------
    values_filtered: numpy.ndarray
        Filtered signal
    """
    
    if isinstance(cut_off, list) or isinstance(cut_off, tuple):
        pass
    else:
        raise ValueError("cut_off must be list or tuple.")
    if len(cut_off) != 2:
        raise ValueError("cut_off must be length 2.")
    if cut_off[0] is not None and cut_off[1] is not None:
        filter_type = "band"
        cut_off = np.array(cut_off)
    elif cut_off[0] is None:
        filter_type = 'low'
        cut_off = cut_off[1]
    else:
        filter_type = 'high'
        cut_off = cut_off[0]

    filter_order = kwargs.get('filter_order', 4)
    remove_gibbs = kwargs.get('remove_gibbs', None)
    gibbs_extra = kwargs.get('gibbs_extra', 1)
    gibbs_range = kwargs.get('gibbs_range', 50)
    sampling_rate = 1.0 / dt
    nyq = sampling_rate * 0.5

    values_filtered = values
    org_len = len(values_filtered)

    if remove_gibbs is not None:
        # Pad end of record with extra zeros then cut it off after filtering
        nindex = int(np.ceil(np.log2(len(values_filtered)))) + gibbs_extra
        new_len = 2 ** nindex
        diff_len = new_len - org_len
        if remove_gibbs == 'start':
            s_len = 0
            f_len = s_len + org_len
        elif remove_gibbs == 'end':
            s_len = diff_len
            f_len = s_len + org_len
        else:
            s_len = int(diff_len / 2)
            f_len = s_len + org_len

        end_value = np.mean(values_filtered[-gibbs_range:])
        start_value = np.mean(values_filtered[:gibbs_range])
        temp = start_value * np.ones(new_len)
        temp[f_len:] = end_value
        temp[s_len:f_len] = values_filtered
        values_filtered = temp
    else:
        s_len = 0
        f_len = org_len

    wp = cut_off / nyq
    b, a = butter(filter_order, wp, btype=filter_type)
    values_filtered = filtfilt(b, a, values_filtered)
    # removing extra zeros from gibbs effect
    values_filtered = values_filtered[s_len:f_len]

    return values_filtered

def sdof_ltha(Ag,dt,T,xi,m):
    """
    Details
    -------
    This script will carry out linear time history analysis for SDOF system
    It currently uses Newmark Beta Method
    
    References
    ---------- 
    Chopra, A.K. 2012. Dynamics of Structures: Theory and 
    Applications to Earthquake Engineering, Prentice Hall.
    N. M. Newmark, “A Method of Computation for Structural Dynamics,”
    ASCE Journal of the Engineering Mechanics Division, Vol. 85, 1959, pp. 67-94.
    
    Notes
    -----
    * Linear Acceleration Method: Gamma = 1/2, Beta = 1/6
    * Average Acceleration Method: Gamma = 1/2, Beta = 1/4
    * Average acceleration method is unconditionally stable,
      whereas linear acceleration method is stable only if dt/Tn <= 0.55
      Linear acceleration method is preferable due to its accuracy.
    
    Parameters
    ----------
    Ag: numpy.ndarray    
        Acceleration values
    dt: float
        Time step [sec]
    T:  float, numpy.ndarray
        Considered period array e.g. 0 sec, 0.1 sec ... 4 sec
    xi: float
        Damping ratio, e.g. 0.05 for 5%
    m:  float
        Mass of SDOF system
        
    Returns
    -------
    u: numpy.ndarray       
        Relative displacement response history
    v: numpy.ndarray   
        Relative velocity response history
    ac: numpy.ndarray 
        Relative acceleration response history
    ac_tot: numpy.ndarray 
        Total acceleration response history
    """
    
    # Get the length of acceleration history array
    n1 = max(Ag.shape)
    # Get the length of period array
    n2 = max(T.shape); T = T.reshape((1,n2))
    
    # Assign the external force
    p = -m*Ag
    
    # Calculate system properties which depend on period
    fn = np.ones(T.shape); fn = 1/T             # frequency
    wn = np.ones(T.shape); wn = 2*np.pi*fn      # circular natural frequency
    k  = np.ones(T.shape); k = m*wn**2          # actual stiffness
    c  = np.ones(T.shape); c = 2*m*wn*xi        # actual damping coefficient
    
    # Newmark Beta Method coefficients
    Gamma = np.ones((1,n2))*(1/2)
    # Use linear acceleration method for dt/T<=0.55
    Beta = np.ones((1,n2))*1/6
    # Use average acceleration method for dt/T>0.55
    Beta[np.where(dt/T > 0.55)] = 1/4
    
    # Compute the constants used in Newmark's integration
    a1 = Gamma/(Beta*dt)
    a2 = 1/(Beta*dt**2)
    a3 = 1/(Beta*dt)
    a4 = Gamma/Beta
    a5 = 1/(2*Beta)
    a6 = (Gamma/(2*Beta)-1)*dt
    kf = k + a1*c + a2*m
    a = a3*m + a4*c
    b = a5*m + a6*c
    
    # Initialize the history arrays
    u = np.zeros((n1,n2))        # relative displacement history
    v = np.zeros((n1,n2))        # relative velocity history
    ac = np.zeros((n1,n2))       # relative acceleration history
    ac_tot = np.zeros((n1,n2)) # total acceleration history

    # Set the Initial Conditions
    u[0] = 0
    v[0] = 0
    ac[0] = (p[0] - c*v[0] - k*u[0])/m
    ac_tot[0] = ac[0] + Ag[0]

    for i in range(n1-1):
        dpf = (p[i+1] - p[i]) + a*v[i] + b*ac[i]
        du = dpf/kf
        dv = a1*du - a4*v[i] - a6*ac[i]
        da = a2*du - a3*v[i] - a5*ac[i]
    
        # Update history variables
        u[i+1] = u[i]+du
        v[i+1] = v[i]+dv
        ac[i+1] = ac[i]+da
        ac_tot[i+1] = ac[i+1] + Ag[i+1]
   
    return u,v,ac,ac_tot

def gm_parameters(Ag,dt,T,xi):
    """
    Details
    -------
    This script will return spectra and ground motion parameters for a given record
        
    References
    ---------- 
    Kramer, Steven L. 1996. Geotechnical Earthquake Engineering, Prentice Hall
    Chopra, A.K. 2012. Dynamics of Structures: Theory and 
    Applications to Earthquake Engineering, Prentice Hall.
        
    Parameters
    ----------
    Ag: numpy.ndarray    
        Acceleration values [m/s2]
    dt: float
        Time step [sec]
    T:  float, numpy.ndarray
        Considered period array e.g. 0 sec, 0.1 sec ... 4 sec
    xi: float
        Damping ratio, e.g. 0.05 for 5%
        
    Returns
    -------
    param: dictionary
        Contains the following intensity measures:
    PSa(T): numpy.ndarray       
        Elastic pseudo-acceleration response spectrum [m/s2]
    PSv(T): numpy.ndarray   
        Elastic pseudo-velocity response spectrum [m/s]
    Sd(T): numpy.ndarray 
        Elastic displacement response spectrum  - relative displacement [m]
    Sv(T): numpy.ndarray 
        Elastic velocity response spectrum - relative velocity at [m/s]
    Sa(T): numpy.ndarray 
        Elastic accleration response spectrum - total accelaration [m/s2]
    Ei_r(T): numpy.ndarray 
        Relative input energy spectrum for elastic system [N.m]
    Ei_a(T): numpy.ndarray 
        Absolute input energy spectrum for elastic system [N.m]
    Periods: numpy.ndarray 
        Periods where spectral values are calculated [sec]
    FAS: numpy.ndarray 
        Fourier amplitude spectra
    PAS: numpy.ndarray 
        Power amplitude spectra
    PGA: float
        Peak ground acceleration [m/s2]
    PGV: float
        Peak ground velocity [m/s]
    PGD: float
        Peak ground displacement [m]
    Aint: numpy.ndarray 
        Arias intensity ratio vector with time [m/s]
    Arias: float 
        Maximum value of arias intensity ratio [m/s]
    HI: float
        Housner intensity ratio [m]
        Requires T to be defined between (0.1-2.5 sec)
        Otherwise not applicable, and equal to 'N.A'
    CAV: float
        Cumulative absolute velocity [m/s]        
    t_5_75: list
        Significant duration time vector between 5% and 75% of energy release (from Aint)
    D_5_75: float
        Significant duration between 5% and 75% of energy release (from Aint)
    t_5_95: list    
        Significant duration time vector between 5% and 95% of energy release (from Aint)
    D_5_95: float
        Significant duration between 5% and 95% of energy release (from Aint)
    t_bracketed: list 
        Bracketed duration time vector (acc>0.05g)
        Not applicable, in case of low intensity records, 
        Thus, equal to 'N.A'
    D_bracketed: float
        Bracketed duration (acc>0.05g)
        Not applicable, in case of low intensity records, 
        Thus, equal to 'N.A'
    t_uniform: list 
        Uniform duration time vector (acc>0.05g)
        Not applicable, in case of low intensity records, 
        Thus, equal to 'N.A'
    D_uniform: float 
        Uniform duration (acc>0.05g)
        Not applicable, in case of low intensity records, 
        Thus, equal to 'N.A'
    Tm: float
        Mean period
    Tp: float             
        Predominant Period
    aRMS: float 
        Root mean square root of acceleration [m/s2]
    vRMS: float
        Root mean square root of velocity [m/s]
    dRMS: float  
        Root mean square root of displacement [m]
    Ic: float     
        End time might which is used herein, is not always a good choice
    ASI: float   
        Acceleration spectrum intensity [m/s]
        Requires T to be defined between (0.1-0.5 sec)
        Otherwise not applicable, and equal to 'N.A'
    MASI: float [m]
        Modified acceleration spectrum intensity
        Requires T to be defined between (0.1-2.5 sec)
        Otherwise not applicable, and equal to 'N.A'
    VSI: float [m]
        Velocity spectrum intensity
        Requires T to be defined between (0.1-2.5 sec)
        Otherwise not applicable, and equal to 'N.A'
    """
        
    # INITIALIZATION
    T = T[T!=0] # do not use T = zero for response spectrum calculations
    param = {'Periods':T}
    
    # GET SPECTRAL VALUES
    # Get the length of acceleration history array
    n1 = max(Ag.shape)
    # Get the length of period array
    n2 = max(T.shape)
    # Create the time array
    t = np.linspace(0,(n1-1)*dt,n1)
    # Get ground velocity and displacement through integration
    Vg = cumtrapz(Ag, t, initial=0)
    Dg = cumtrapz(Vg, t, initial=0)
    # Mass (kg)
    m = 1
    # Carry out linear time history analyses for SDOF system
    u,v,ac,ac_tot = sdof_ltha(Ag, dt, T, xi, m)
    # Calculate the spectral values
    param['Sd'] = np.max(np.abs((u)),axis = 0)
    param['Sv'] = np.max(np.abs((v)),axis = 0)
    param['Sa'] = np.max(np.abs((ac_tot)),axis = 0)
    param['PSv'] = (2*np.pi/T)*param['Sd']
    param['PSa'] = ((2*np.pi/T)**2)*param['Sd']
    ei_r = cumtrapz(-numpy.matlib.repmat(Ag, n2, 1).T, u, axis=0, initial=0)*m
    ei_a = cumtrapz(-numpy.matlib.repmat(Dg, n2, 1).T, ac_tot, axis=0, initial=0)*m
    param['Ei_r'] = ei_r[-1]
    param['Ei_a'] = ei_a[-1]

    # GET PEAK GROUND ACCELERATION, VELOCITY AND DISPLACEMENT
    param['PGA'] = np.max(np.abs(Ag))
    param['PGV'] = np.max(np.abs(Vg))
    param['PGD'] = np.max(np.abs(Dg))
    
    # GET ARIAS INTENSITY
    Aint = np.cumsum(Ag**2)*np.pi*dt/(2*9.81)
    param['Arias'] = Aint[-1]
    temp = np.zeros((len(Aint),2)); temp[:,0] = t; temp[:,1] = Aint;
    param['Aint'] = temp
    
    # GET HOUSNER INTENSITY
    try:
        index1 = np.where(T==0.1)[0][0]
        index2 = np.where(T==2.5)[0][0]
        param['HI'] = np.trapz(param['PSv'][index1:index2],T[index1:index2])
    except:
        param['HI'] = 'N.A.'
    
    # SIGNIFICANT DURATION (5%-75% Ia)
    mask = (Aint>=0.05*Aint[-1])*(Aint<=0.75*Aint[-1])
    timed = t[mask]
    t1 = round(timed[0],3); t2 = round(timed[-1],3)
    param['t_5_75'] = [t1,t2]
    param['D_5_75'] = round(t2-t1,3)
    
    # SIGNIFICANT DURATION (5%-95% Ia)
    mask = (Aint>=0.05*Aint[-1])*(Aint<=0.95*Aint[-1])
    timed = t[mask]
    t1 = round(timed[0],3); t2 = round(timed[-1],3)
    param['t_5_95'] = [t1,t2]
    param['D_5_95'] = round(t2-t1,3)
    
    # BRACKETED DURATION (0.05g)
    try:
        mask = np.abs(Ag)>=0.05*9.81; indices = np.where(mask)[0]
        # mask = np.abs(Ag)>=0.05*np.max(np.abs(Ag)); indices = np.where(mask)[0]
        t1 = round(t[indices[0]],3); t2 = round(t[indices[-1]],3);   
        param['t_bracketed'] = [t1,t2]
        param['D_bracketed'] = round(t2-t1,3)
    except: # in case of ground motions with low intensities
        param['t_bracketed'] = 'N.A.'
        param['D_bracketed'] = 'N.A.'
        
    # UNIFORM DURATION (0.05g)
    try:
        mask = np.abs(Ag)>=0.05*9.81; indices = np.where(mask)[0]
        # mask = np.abs(Ag)>=0.05*np.max(np.abs(Ag)); indices = np.where(mask)[0]
        t_treshold = t[indices]
        param['t_uniform'] = [t_treshold]
        param['D_uniform'] = round((len(t_treshold)-1)*dt)
    except: # in case of ground motions with low intensities
        param['t_uniform'] = 'N.A.'
        param['D_uniform'] = 'N.A.'

    # CUMULATVE ABSOLUTE VELOCITY
    param['CAV'] = np.trapz(np.abs(Ag),t)
    
    # CHARACTERISTIC INTENSITY, ROOT MEAN SQUARE OF ACC, VEL, DISP
    Td = t[-1] # note this might not be the best indicative, different Td might be chosen
    param['aRMS'] = np.sqrt(np.trapz(Ag**2,t)/Td)
    param['vRMS'] = np.sqrt(np.trapz(Vg**2,t)/Td)
    param['dRMS'] = np.sqrt(np.trapz(Dg**2,t)/Td)
    param['Ic'] = param['aRMS']**(1.5)*np.sqrt(Td)
    
    # ACCELERATION AND VELOCITY SPECTRUM INTENSITY
    try:
        index3 = np.where(T==0.5)[0][0]
        param['ASI'] = np.trapz(param['Sa'][index1:index3],T[index1:index3])
    except:
        param['ASI'] = 'N.A.'
    try:
        param['MASI'] = np.trapz(param['Sa'][index1:index2],T[index1:index2])
        param['VSI'] = np.trapz(param['Sv'][index1:index2],T[index1:index2])
    except:
        param['MASI'] = 'N.A.'
        param['VSI'] = 'N.A.'       
    
    # GET FOURIER AMPLITUDE AND POWER AMPLITUDE SPECTRUM
    # Number of sample points, add zeropads
    N = 2 ** int(np.ceil(np.log2(len(Ag))))
    Famp = fft(Ag,N)
    Famp = np.abs(fftshift(Famp))*dt
    freq = fftfreq(N, dt)
    freq = fftshift(freq)
    Famp = Famp[freq>0]
    freq = freq[freq>0]
    Pamp = Famp**2/(np.pi*t[-1]*param['aRMS']**2);
    FAS = np.zeros((len(Famp),2)); FAS[:,0] = freq; FAS[:,1] = Famp;
    PAS = np.zeros((len(Famp),2)); PAS[:,0] = freq; PAS[:,1] = Pamp;
    param['FAS'] = FAS
    param['PAS'] = FAS
    
    # MEAN PERIOD
    mask = (freq>0.25)*(freq<20) ; indices = np.where(mask)[0]
    fi = freq[indices]
    Ci = Famp[indices]
    param['Tm'] = np.sum(Ci**2/fi)/np.sum(Ci**2)
    
    # PREDOMINANT PERIOD
    mask = param['Sa'] == max(param['Sa']); indices = np.where(mask)[0]
    param['Tp'] = T[indices]
    
    return param

def RotDxx_spectrum(Ag1,Ag2,dt,T,xi,xx):
    """
    Details
    -------
    This script will return RotDxx spectrum
    It currently uses Newmark Beta Method
    
    References
    ---------- 
    Boore, D. M. (2006). Orientation-Independent Measures of Ground Motion. 
    Bulletin of the Seismological Society of America, 96(4A), 1502–1511.
    Boore, D. M. (2010). Orientation-Independent, Nongeometric-Mean Measures 
    of Seismic Intensity from Two Horizontal Components of Motion. 
    Bulletin of the Seismological Society of America, 100(4), 1830–1835.
    
    Notes
    -----
    * Linear Acceleration Method: Gamma = 1/2, Beta = 1/6
    * Average Acceleration Method: Gamma = 1/2, Beta = 1/4
    * Average acceleration method is unconditionally stable,
      whereas linear acceleration method is stable only if dt/Tn <= 0.55
      Linear acceleration method is preferable due to its accuracy.
        
    Parameters
    ----------
    Ag1 : numpy.ndarray    
        Acceleration values of 1st horizontal ground motion component
    Ag2 : numpy.ndarray    
        Acceleration values of 2nd horizontal ground motion component
    dt: float
        Time step [sec]
    T:  float, numpy.ndarray
        Considered period array e.g. 0 sec, 0.1 sec ... 4 sec
    xi: float
        Damping ratio, e.g. 0.05 for 5%
    xx: int
        Percentile to calculate, e.g. 50 for RotD50
        
    Returns
    -------
    Sa_RotDxx: numpy.ndarray 
        RotDxx Spectra
    """

    # Verify if the length of arrays are the same
    if len(Ag1) == len(Ag2):
        pass
    elif len(Ag1) > len(Ag2):
        Ag2 = np.append(Ag2,np.zeros(len(Ag1)-len(Ag2)))
    elif len(Ag2) > len(Ag1):
        Ag1 = np.append(Ag1,np.zeros(len(Ag2)-len(Ag1)))
        
    # Get the length of period array 
    n2 = max(T.shape)
    
    # Mass (kg)
    m = 1

    # Carry out linear time history analyses for SDOF system
    u1,_,_,_ = sdof_ltha(Ag1, dt, T, xi, m)
    u2,_,_,_ = sdof_ltha(Ag2, dt, T, xi, m)
    
    # RotD definition is taken from Boore 2010.
    Rot_Disp = np.zeros((180,n2))
    for theta in range (0,180,1):
        Rot_Disp[theta] = np.max(u1*np.cos(np.deg2rad(theta))+u2*np.sin(np.deg2rad(theta)), axis = 0)

    Rot_Acc = Rot_Disp*(2*np.pi/T)**2
    Sa_RotDxx = np.percentile(Rot_Acc, xx, axis = 0)
    
    return Sa_RotDxx
