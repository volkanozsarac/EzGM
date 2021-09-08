"""
Record Selection ToolBox
"""

# Import python libraries
import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from numba import njit
from openquake.hazardlib import gsim, imt, const
from scipy import interpolate
from scipy.io import loadmat
from scipy.stats import skew
from time import gmtime
from .Utility import downloader, file_manager

class conditional_spectrum(downloader, file_manager):
    """
    This class is used to
        1) Create target spectrum
            Unconditional spectrum using specified gmpe
            Conditional spectrum using average spectral acceleration
            Conditional spectrum using spectral acceleration
            with and without considering variance
        2) Selecting suitable ground motion sets for target spectrum
        3) Scaling and processing of selected ground motion records
    """

    def __init__(self, Tstar=0.5, gmpe='BooreEtAl2014', database='NGA_W2', pInfo=1):
        # TODO: Combine all metadata into single sql file.
        """
        Details
        -------
        Loads the database and add spectral values for Tstar 
        if they are not present via interpolation.
        
        Parameters
        ----------
        Tstar    : int, float, numpy.array, the default is None.
            Conditioning period or periods in case of AvgSa [sec].
        gmpe     : str, optional
            GMPE model (see OpenQuake library). 
            The default is 'BooreEtAl2014'.
        database : str, optional
            database to use: NGA_W2 or EXSIM_Duzce
            The default is NGA_W2.        
        pInfo    : int, optional
            flag to print required input for the gmpe which is going to be used. 
            (0: no, 1:yes)
            The default is 1.
            
        Returns
        -------
        None.
        """

        # add Tstar to self
        if isinstance(Tstar, int) or isinstance(Tstar, float):
            self.Tstar = np.array([Tstar])
        elif isinstance(Tstar, numpy.ndarray):
            self.Tstar = Tstar

        # Add the input the ground motion database to use
        matfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Meta_Data', database)
        self.database = loadmat(matfile, squeeze_me=True)
        self.database['Name'] = database

        # initialize the objects being used
        downloader.__init__(self)
        file_manager.__init__(self)

        # check if AvgSa or Sa is used as IM, 
        # then in case of Sa(T*) add T* and Sa(T*) if not present
        if not self.Tstar[0] in self.database['Periods'] and len(self.Tstar) == 1:
            f = interpolate.interp1d(self.database['Periods'], self.database['Sa_1'], axis=1)
            Sa_int = f(self.Tstar[0])
            Sa_int.shape = (len(Sa_int), 1)
            Sa = np.append(self.database['Sa_1'], Sa_int, axis=1)
            Periods = np.append(self.database['Periods'], self.Tstar[0])
            self.database['Sa_1'] = Sa[:, np.argsort(Periods)]

            if database.startswith("NGA"):
                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_2'], axis=1)
                Sa_int = f(self.Tstar[0])
                Sa_int.shape = (len(Sa_int), 1)
                Sa = np.append(self.database['Sa_2'], Sa_int, axis=1)
                self.database['Sa_2'] = Sa[:, np.argsort(Periods)]

                f = interpolate.interp1d(self.database['Periods'], self.database['Sa_RotD50'], axis=1)
                Sa_int = f(self.Tstar[0])
                Sa_int.shape = (len(Sa_int), 1)
                Sa = np.append(self.database['Sa_RotD50'], Sa_int, axis=1)
                self.database['Sa_RotD50'] = Sa[:, np.argsort(Periods)]

            self.database['Periods'] = Periods[np.argsort(Periods)]
            
        try: # this is smth like self.bgmpe = gsim.boore_2014.BooreEtAl2014()
            self.bgmpe = gsim.get_available_gsims()[gmpe]()
            
        except: 
            raise KeyError('Not a valid gmpe')

        if pInfo == 1:  # print the selected gmpe info
            print('For the selected gmpe;')
            print('The mandatory input distance parameters are %s' % list(self.bgmpe.REQUIRES_DISTANCES))
            print('The mandatory input rupture parameters are %s' % list(self.bgmpe.REQUIRES_RUPTURE_PARAMETERS))
            print('The mandatory input site parameters are %s' % list(self.bgmpe.REQUIRES_SITES_PARAMETERS))
            print('The defined intensity measure component is %s' % self.bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT)
            print('The defined tectonic region type is %s\n' % self.bgmpe.DEFINED_FOR_TECTONIC_REGION_TYPE)

    @staticmethod
    def BakerJayaramCorrelationModel(T1, T2, orth=0):
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

    @staticmethod
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

        if np.any([T1, T2] < periods[0]) or \
                np.any([T1, T2] > periods[-1]):
            raise ValueError("Period array contains values outside of the "
                             "range supported by the Akkar et al. (2014) "
                             "correlation model")

        if T1 == T2:
            rho = 1.0
        else:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Meta_Data', 'akkar_coeff_table.npy'), 'rb') as f:
                coeff_table = np.load(f)
            rho = interpolate.interp2d(periods, periods, coeff_table, kind='linear')(T1, T2)[0]

        return rho

    def get_correlation(self, T1, T2):
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

        correlation_function_handles = {
            'baker_jayaram': self.BakerJayaramCorrelationModel,
            'akkar': self.AkkarCorrelationModel,
        }

        # Check for existing correlation function
        if self.corr_func not in correlation_function_handles:
            raise ValueError('Not a valid correlation function')
        else:
            rho = \
                correlation_function_handles[self.corr_func](T1, T2)

        return rho

    def Sa_avg(self, bgmpe, scenario, T):
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
        T : numpy.array
            Array of interested Periods (sec).
    
        Returns
        -------
        Sa : numpy.array
            Mean of logarithmic average spectral acceleration prediction.
        sigma : numpy.array
           logarithmic standard deviation of average spectral acceleration prediction.
        """

        n = len(T)
        mu_lnSaTstar = np.zeros(n)
        sigma_lnSaTstar = np.zeros(n)
        MoC = np.zeros((n, n))
        # Get the GMPE output
        for i in range(n):
            mu_lnSaTstar[i], stddvs_lnSaTstar = bgmpe.get_mean_and_stddevs(scenario[0], scenario[1], scenario[2],
                                                                           imt.SA(period=T[i]), [const.StdDev.TOTAL])
            # convert to sigma_arb
            # One should uncomment this line if the arbitary component is used for
            # record selection.
            # ro_xy = 0.79-0.23*np.log(T[k])
            ro_xy = 1
            sigma_lnSaTstar[i] = np.log(((np.exp(stddvs_lnSaTstar[0][0]) ** 2) * (2 / (1 + ro_xy))) ** 0.5)

            for j in range(n):
                rho = self.get_correlation(T[i], T[j])
                MoC[i, j] = rho

        SPa_avg_meanLn = (1 / n) * sum(mu_lnSaTstar)  # logarithmic mean of Sa,avg

        SPa_avg_std = 0
        for i in range(n):
            for j in range(n):
                SPa_avg_std = SPa_avg_std + (
                        MoC[i, j] * sigma_lnSaTstar[i] * sigma_lnSaTstar[j])  # logarithmic Var of the Sa,avg

        SPa_avg_std = SPa_avg_std * (1 / n) ** 2
        # compute mean of logarithmic average spectral acceleration
        # and logarithmic standard deviation of 
        # spectral acceleration prediction
        Sa = SPa_avg_meanLn
        sigma = np.sqrt(SPa_avg_std)
        return Sa, sigma

    def rho_AvgSA_SA(self, bgmpe, scenario, T, Tstar):
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
        T     : numpy.array
            Array of interested period to calculate correlation coefficient.
    
        Tstar : numpy.array
            Period range where AvgSa is calculated.
    
        Returns
        -------
        rho : int
            Predicted correlation coefficient.
        """

        rho = 0
        for j in range(len(Tstar)):
            rho_bj = self.get_correlation(T, Tstar[j])
            _, sig1 = bgmpe.get_mean_and_stddevs(scenario[0], scenario[1], scenario[2], imt.SA(period=Tstar[j]),
                                                 [const.StdDev.TOTAL])
            rho = rho_bj * sig1[0][0] + rho

        _, Avg_sig = self.Sa_avg(bgmpe, scenario, Tstar)
        rho = rho / (len(Tstar) * Avg_sig)
        return rho

    def create(self, site_param={'vs30': 520}, rup_param={'rake': 0.0, 'mag': [7.2, 6.5]},
               dist_param={'rjb': [20, 5]}, Hcont=[0.6, 0.4], T_Tgt_range=[0.01, 4],
               im_Tstar=1.0, epsilon=None, cond=1, useVar=1, corr_func='baker_jayaram',
               outdir='Outputs'):
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

        # create the output directory and add the path to self
        cwd = os.getcwd()
        outdir_path = os.path.join(cwd, outdir)
        self.outdir = outdir_path
        self.create_dir(self.outdir)

        # add target spectrum settings to self
        self.cond = cond
        self.useVar = useVar
        self.corr_func = corr_func

        if cond == 0:  # there is no conditioning period
            del self.Tstar

        # Get number of scenarios, and their contribution
        nScenarios = len(rup_param['mag'])
        if Hcont is None:
            self.Hcont = [1 / nScenarios for _ in range(nScenarios)]
        else:
            self.Hcont = Hcont

        # Period range of the target spectrum
        temp = np.abs(self.database['Periods'] - np.min(T_Tgt_range))
        idx1 = np.where(temp == np.min(temp))[0][0]
        temp = np.abs(self.database['Periods'] - np.max(T_Tgt_range))
        idx2 = np.where(temp == np.min(temp))[0][0]
        T_Tgt = self.database['Periods'][idx1:idx2 + 1]

        # Get number of scenarios, and their contribution
        Hcont_mat = np.matlib.repmat(np.asarray(self.Hcont), len(T_Tgt), 1)

        # Conditional spectrum, log parameters
        TgtMean = np.zeros((len(T_Tgt), nScenarios))

        # Covariance
        TgtCov = np.zeros((nScenarios, len(T_Tgt), len(T_Tgt)))

        for n in range(nScenarios):

            # gmpe spectral values
            mu_lnSaT = np.zeros(len(T_Tgt))
            sigma_lnSaT = np.zeros(len(T_Tgt))

            # correlation coefficients
            rho_T_Tstar = np.zeros(len(T_Tgt))

            # Covariance
            Cov = np.zeros((len(T_Tgt), len(T_Tgt)))

            # TODO: it could be better to calculate some parameters automatically elsewhere
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

            scenario = [sites, rup, dists]

            for i in range(len(T_Tgt)):
                # Get the GMPE ouput for a rupture scenario
                mu0, sigma0 = self.bgmpe.get_mean_and_stddevs(sites, rup, dists, imt.SA(period=T_Tgt[i]),
                                                              [const.StdDev.TOTAL])
                mu_lnSaT[i] = mu0[0]
                sigma_lnSaT[i] = sigma0[0][0]

                if self.cond == 1:
                    # Compute the correlations between each T and Tstar
                    rho_T_Tstar[i] = self.rho_AvgSA_SA(self.bgmpe, scenario, T_Tgt[i], self.Tstar)

            if self.cond == 1:
                # Get the GMPE output and calculate Avg_Sa_Tstar
                mu_lnSaTstar, sigma_lnSaTstar = self.Sa_avg(self.bgmpe, scenario, self.Tstar)

                if epsilon is None:
                    # Back calculate epsilon
                    rup_eps = (np.log(im_Tstar) - mu_lnSaTstar) / sigma_lnSaTstar
                else:
                    rup_eps = epsilon[n]

                # Get the value of the ln(CMS), conditioned on T_star
                TgtMean[:, n] = mu_lnSaT + rho_T_Tstar * rup_eps * sigma_lnSaT

            elif self.cond == 0:
                TgtMean[:, n] = mu_lnSaT

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
                        sigma12 = np.array([rho_T_Tstar[i] * np.sqrt(var1 * varTstar),
                                            rho_T_Tstar[j] * np.sqrt(varTstar * var2)])
                        sigma12.shape = (2, 1)
                        sigma22.shape = (1, 1)
                        sigma_cond = sigma11 - sigma12 * 1. / (sigma22) * sigma12.T
                        Cov[i, j] = sigma_cond[0, 1]

                    elif self.cond == 0:
                        Cov[i, j] = sigma_Corr

            # Get the value of standard deviation of target spectrum
            TgtCov[n, :, :] = Cov

        # over-write coveriance matrix with zeros if no variance is desired in the ground motion selection
        if self.useVar == 0:
            TgtCov = np.zeros(TgtCov.shape)

        TgtMean_fin = np.sum(TgtMean * Hcont_mat, 1)
        # all 2D matrices are the same for each kk scenario, since sigma is only T dependent
        TgtCov_fin = TgtCov[0, :, :]
        Cov_elms = np.zeros((len(T_Tgt), nScenarios))
        for ii in range(len(T_Tgt)):
            for kk in range(nScenarios):
                # Hcont[kk] = contribution of the k-th scenario
                Cov_elms[ii, kk] = (TgtCov[kk, ii, ii] + (TgtMean[ii, kk] - TgtMean_fin[ii]) ** 2) * self.Hcont[kk]

        cov_diag = np.sum(Cov_elms, 1)
        TgtCov_fin[np.eye(len(T_Tgt)) == 1] = cov_diag

        # Find covariance values of zero and set them to a small number so that
        # random number generation can be performed
        # TgtCov_fin[np.abs(TgtCov_fin) < 1e-10] = 1e-10
        min_eig = np.min(np.real(np.linalg.eigvals(TgtCov_fin)))
        if min_eig < 0:
            TgtCov_fin -= 10 * min_eig * np.eye(*TgtCov_fin.shape)

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
                f = interpolate.interp1d(self.T, np.exp(self.mu_ln))
                Sa_int = f(self.Tstar)
                self.im_Tstar = np.exp(np.sum(np.log(Sa_int)) / len(self.Tstar))
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

        devTotalSim = np.zeros((self.nTrials, 1))
        specDict = {}
        nT = len(self.T)
        # Generate simulated response spectra with best matches to the target values
        for j in range(self.nTrials):
            specDict[j] = np.zeros((self.nGM, nT))
            for i in range(self.nGM):
                # Note: we may use latin hypercube sampling here instead. I leave it as Monte Carlo for now
                specDict[j][i, :] = np.exp(np.random.multivariate_normal(self.mu_ln, self.cov))

            devMeanSim = np.mean(np.log(specDict[j]),
                                 axis=0) - self.mu_ln  # how close is the mean of the spectra to the target
            devSigSim = np.std(np.log(specDict[j]),
                               axis=0) - self.sigma_ln  # how close is the mean of the spectra to the target
            devSkewSim = skew(np.log(specDict[j]),
                              axis=0)  # how close is the skewness of the spectra to zero (i.e., the target)

            devTotalSim[j] = self.weights[0] * np.sum(devMeanSim ** 2) + \
                             self.weights[1] * np.sum(devSigSim ** 2) + \
                             0.1 * (self.weights[2]) * np.sum(
                devSkewSim ** 2)  # combine the three error metrics to compute a total error

        recUse = np.argmin(np.abs(devTotalSim))  # find the simulated spectra that best match the targets
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
        sampleBig : numpy.array
            An array which contains the IMLs from filtered database.
        soil_Vs30 : numpy.array
            An array which contains the Vs30s from filtered database.
        magnitude : numpy.array
            An array which contains the magnitudes from filtered database.
        Rjb : numpy.array
            An array which contains the Rjbs from filtered database.
        mechanism : numpy.array
            An array which contains the fault type info from filtered database.
        Filename_1 : numpy.array
            An array which contains the filename of 1st gm component from filtered database.
            If selection is set to 1, it will include filenames of both components.
        Filename_2 : numpy.array
            An array which contains the filenameof 2nd gm component filtered database.
            If selection is set to 1, it will be None value.
        NGA_num : numpy.array
            If NGA_W2 is used as record database, record sequence numbers from filtered
            database will be saved, for other databases this variable is None.
        """

        if self.selection == 1:  # SaKnown = Sa_arb

            if self.database['Name'] == "NGA_W2":
                SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
                soil_Vs30 = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
                Mw = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)
                Rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
                fault = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
                Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
                NGA_num = np.append(self.database['NGA_num'], self.database['NGA_num'], axis=0)
                eq_ID = np.append(self.database['EQID'], self.database['EQID'], axis=0)

            elif self.database['Name'].startswith("EXSIM"):
                SaKnown = self.database['Sa_1']
                soil_Vs30 = self.database['soil_Vs30']
                Mw = self.database['magnitude']
                Rjb = self.database['Rjb']
                fault = self.database['mechanism']
                Filename_1 = self.database['Filename_1']
                eq_ID = self.database['EQID']

        elif self.selection == 2:  # SaKnown = Sa_g.m. or RotD50
            if self.Sa_def == 'GeoMean':
                SaKnown = np.sqrt(self.database['Sa_1'] * self.database['Sa_2'])
            elif self.Sa_def == 'RotD50':  # SaKnown = Sa_RotD50.
                SaKnown = self.database['Sa_RotD50']
            else:
                raise ValueError('Unexpected Sa definition, exiting...')

            soil_Vs30 = self.database['soil_Vs30']
            Mw = self.database['magnitude']
            Rjb = self.database['Rjb']
            fault = self.database['mechanism']
            Filename_1 = self.database['Filename_1']
            Filename_2 = self.database['Filename_2']
            NGA_num = self.database['NGA_num']
            eq_ID = self.database['EQID']

        else:
            raise ValueError('Selection can only be performed for one or two components at the moment, exiting...')

        perKnown = self.database['Periods']

        # Limiting the records to be considered using the `notAllowed' variable
        # Sa cannot be negative or zero, remove these.
        notAllowed = np.unique(np.where(SaKnown <= 0)[0]).tolist()

        if self.Vs30_lim is not None:  # limiting values on soil exist
            mask = (soil_Vs30 > min(self.Vs30_lim)) * (soil_Vs30 < max(self.Vs30_lim) * np.invert(np.isnan(soil_Vs30)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Mw_lim is not None:  # limiting values on magnitude exist
            mask = (Mw > min(self.Mw_lim)) * (Mw < max(self.Mw_lim) * np.invert(np.isnan(Mw)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Rjb_lim is not None:  # limiting values on Rjb exist
            mask = (Rjb > min(self.Rjb_lim)) * (Rjb < max(self.Rjb_lim) * np.invert(np.isnan(Rjb)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.fault_lim is not None:  # limiting values on mechanism exist
            mask = (fault == self.fault_lim * np.invert(np.isnan(fault)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        # get the unique values
        notAllowed = (list(set(notAllowed)))
        Allowed = [i for i in range(SaKnown.shape[0])]
        for i in notAllowed:
            Allowed.remove(i)

        # Use only allowed records
        SaKnown = SaKnown[Allowed, :]
        soil_Vs30 = soil_Vs30[Allowed]
        Mw = Mw[Allowed]
        Rjb = Rjb[Allowed]
        fault = fault[Allowed]
        eq_ID = eq_ID[Allowed]
        Filename_1 = Filename_1[Allowed]

        if self.selection == 1:
            Filename_2 = None
        else:
            Filename_2 = Filename_2[Allowed]

        if self.database['Name'] == "NGA_W2":
            NGA_num = NGA_num[Allowed]
        else:
            NGA_num = None

        # Arrange the available spectra in a usable format and check for invalid input
        # Match periods (known periods and periods for error computations)
        recPer = []
        for i in range(len(self.T)):
            recPer.append(np.where(perKnown == self.T[i])[0][0])

        # Check for invalid input
        sampleBig = SaKnown[:, recPer]
        if np.any(np.isnan(sampleBig)):
            raise ValueError('NaNs found in input response spectra')

        if self.nGM > len(NGA_num):
            raise ValueError('There are not enough records which satisfy',
                             'the given record selection criteria...',
                             'Please use broaden your selection criteria...')

        return sampleBig, soil_Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num, eq_ID

    def select(self, nGM=30, selection=1, Sa_def='RotD50', isScaled=1, maxScale=4,
               Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None,
               nTrials=20, weights=[1, 2, 0.3], seedValue=0,
               nLoop=2, penalty=0, tol=10):
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
        fault_lim : int, optional, the default is None.
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
        weights : numpy.array or list, optional, the default is [1,2,0.3].
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
        sampleBig, Vs30, Mw, Rjb, fault, Filename_1, Filename_2, NGA_num, eq_ID = self.search_database()

        # Processing available spectra
        sampleBig = np.log(sampleBig)
        nBig = sampleBig.shape[0]

        # Find best matches to the simulated spectra from ground-motion database
        recID = np.ones(self.nGM, dtype=int) * (-1)
        finalScaleFac = np.ones(self.nGM)
        sampleSmall = np.ones((self.nGM, sampleBig.shape[1]))
        weights = np.array(weights)

        if self.cond == 1 and self.isScaled == 1:
            # Calculate IMLs for the sample
            f = interpolate.interp1d(self.T, np.exp(sampleBig), axis=1)
            sampleBig_imls = np.exp(np.sum(np.log(f(self.Tstar)), axis=1) / len(self.Tstar))

        if self.cond == 1 and len(self.Tstar) == 1:
            # These indices are required in case IM = Sa(T) to break the loop
            ind2 = (np.where(self.T != self.Tstar[0])[0][0]).tolist()

        # Find nGM ground motions, inital subset
        for i in range(self.nGM):
            err = np.zeros(nBig)
            scaleFac = np.ones(nBig)

            # Calculate the scaling factor
            if self.isScaled == 1:
                # using conditioning IML
                if self.cond == 1:
                    scaleFac = self.im_Tstar / sampleBig_imls
                # using error minimization
                elif self.cond == 0:
                    scaleFac = np.sum(np.exp(sampleBig) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(
                        np.exp(sampleBig) ** 2, axis=1)
            else:
                scaleFac = np.ones(nBig)

            mask = scaleFac > self.maxScale
            idxs = np.where(~mask)[0]
            err[mask] = 1000000
            err[~mask] = np.sum((np.log(
                np.exp(sampleBig[idxs, :]) * scaleFac[~mask].reshape(len(scaleFac[~mask]), 1)) -
                                 self.sim_spec[i, :]) ** 2, axis=1)

            recID[i] = int(np.argsort(err)[0])
            if err.min() >= 1000000:
                raise Warning('Possible problem with simulated spectrum. No good matches found')

            if self.isScaled == 1:
                finalScaleFac[i] = scaleFac[recID[i]]

            # Save the selected spectra
            sampleSmall[i, :] = np.log(np.exp(sampleBig[recID[i], :]) * finalScaleFac[i])

        # Apply Greedy subset modification procedure
        # Use njit to speed up the optimization algorithm
        @njit
        def find_rec(sampleSmall, scaleFac, mu_ln, sigma_ln, recIDs):

            def mean_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].mean())

                return np.array(res)

            def std_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].std())

                return np.array(res)

            minDev = 100000
            for j in range(nBig):
                # Add to the sample the scaled spectra
                temp = np.zeros((1, len(sampleBig[j, :])))
                temp[:, :] = sampleBig[j, :]
                tempSample = np.concatenate((sampleSmall, temp + np.log(scaleFac[j])), axis=0)
                devMean = mean_numba(tempSample) - mu_ln  # Compute deviations from target
                devSig = std_numba(tempSample) - sigma_ln
                devTotal = weights[0] * np.sum(devMean * devMean) + weights[1] * np.sum(devSig * devSig)

                # Check if we exceed the scaling limit
                if scaleFac[j] > maxScale or np.any(recIDs == j):
                    devTotal = devTotal + 1000000
                # Penalize bad spectra
                elif penalty > 0:
                    for m in range(nGM):
                        devTotal = devTotal + np.sum(
                            np.abs(np.exp(tempSample[m, :]) > np.exp(mu_ln + 3.0 * sigma_ln))) * penalty
                        devTotal = devTotal + np.sum(
                            np.abs(np.exp(tempSample[m, :]) < np.exp(mu_ln - 3.0 * sigma_ln))) * penalty

                # Should cause improvement and record should not be repeated
                if devTotal < minDev:
                    minID = j
                    minDev = devTotal

            return minID

        for k in range(self.nLoop):  # Number of passes

            for i in range(self.nGM):  # Loop for nGM
                sampleSmall = np.delete(sampleSmall, i, 0)
                recID = np.delete(recID, i)

                # Calculate the scaling factor
                if self.isScaled == 1:
                    # using conditioning IML
                    if self.cond == 1:
                        scaleFac = self.im_Tstar / sampleBig_imls
                    # using error minimization
                    elif self.cond == 0:
                        scaleFac = np.sum(np.exp(sampleBig) * np.exp(self.sim_spec[i, :]), axis=1) / np.sum(
                            np.exp(sampleBig) ** 2, axis=1)
                else:
                    scaleFac = np.ones(nBig)

                # Try to add a new spectra to the subset list
                minID = find_rec(sampleSmall, scaleFac, self.mu_ln, self.sigma_ln, recID)

                # Add new element in the right slot
                if self.isScaled == 1:
                    finalScaleFac[i] = scaleFac[minID]
                else:
                    finalScaleFac[i] = 1
                sampleSmall = np.concatenate(
                    (sampleSmall[:i, :], sampleBig[minID, :].reshape(1, sampleBig.shape[1]) + np.log(scaleFac[minID]),
                     sampleSmall[i:, :]), axis=0)
                recID = np.concatenate((recID[:i], np.array([minID]), recID[i:]))

            # Lets check if the selected ground motions are good enough, if the errors are sufficiently small stop!
            if self.cond == 1 and len(self.Tstar) == 1:  # if conditioned on SaT, ignore error at T*
                medianErr = np.max(
                    np.abs(np.exp(np.mean(sampleSmall[:, ind2], axis=0)) - np.exp(self.mu_ln[ind2])) / np.exp(
                        self.mu_ln[ind2])) * 100
                stdErr = np.max(
                    np.abs(np.std(sampleSmall[:, ind2], axis=0) - self.sigma_ln[ind2]) / self.sigma_ln[ind2]) * 100
            else:
                medianErr = np.max(
                    np.abs(np.exp(np.mean(sampleSmall, axis=0)) - np.exp(self.mu_ln)) / np.exp(self.mu_ln)) * 100
                stdErr = np.max(np.abs(np.std(sampleSmall, axis=0) - self.sigma_ln) / self.sigma_ln) * 100

            if medianErr < self.tol and stdErr < self.tol:
                break
        print('Ground motion selection is finished.')
        print('For T ∈ [%.2f - %.2f]' % (self.T[0], self.T[-1]))
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
        self.eq_ID = eq_ID[recID]
        self.rec_h1 = Filename_1[recID]

        if self.selection == 1:
            self.rec_h2 = None
        elif self.selection == 2:
            self.rec_h2 = Filename_2[recID]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recID]
        else:
            self.rec_rsn = None

    def plot(self, tgt=0, sim=0, rec=1, save=0, show=1):
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

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.ioff()

        if self.cond == 1:
            if len(self.Tstar) == 1:
                hatch = [self.Tstar * 0.98, self.Tstar * 1.02]
            else:
                hatch = [self.Tstar.min(), self.Tstar.max()]

        if tgt == 1:
            # Plot Target spectrum vs. Simulated response spectra
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            plt.suptitle('Target Spectrum', y=0.95)
            ax[0].loglog(self.T, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                             label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                             label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

            ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
            ax[0].set_xlabel('Period [sec]')
            ax[0].set_ylabel('Spectral Acceleration [g]')
            ax[0].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
            ax[0].set_xlim([self.T[0], self.T[-1]])
            if self.cond == 1:
                ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

            # Sample and target standard deviations
            if self.useVar == 1:
                ax[1].semilogx(self.T, self.sigma_ln, color='red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
                ax[1].set_xlabel('Period [sec]')
                ax[1].set_ylabel('Dispersion')
                ax[1].grid(True)
                ax[1].legend(frameon=False)
                ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[0].set_xlim([self.T[0], self.T[-1]])
                ax[1].set_ylim(bottom=0)
                if self.cond == 1:
                    ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

            if save == 1:
                plt.savefig(os.path.join(self.outdir, 'Targeted.pdf'))

        if sim == 1:
            # Plot Target spectrum vs. Simulated response spectra
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            plt.suptitle('Target Spectrum vs. Simulated Spectra', y=0.95)

            for i in range(self.nGM):
                ax[0].loglog(self.T, np.exp(self.sim_spec[i, :]), color='gray', lw=1, label='Selected')

            ax[0].loglog(self.T, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                             label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                             label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

            ax[0].loglog(self.T, np.exp(np.mean(self.sim_spec, axis=0)), color='blue', lw=2,
                         label='Selected - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T, np.exp(np.mean(self.sim_spec, axis=0) + 2 * np.std(self.sim_spec, axis=0)),
                             color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T, np.exp(np.mean(self.sim_spec, axis=0) - 2 * np.std(self.sim_spec, axis=0)),
                             color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

            ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
            ax[0].set_xlabel('Period [sec]')
            ax[0].set_ylabel('Spectral Acceleration [g]')
            ax[0].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
            ax[0].set_xlim([self.T[0], self.T[-1]])
            if self.cond == 1:
                ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

            if self.useVar == 1:
                # Sample and target standard deviations
                ax[1].semilogx(self.T, self.sigma_ln, color='red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
                ax[1].semilogx(self.T, np.std(self.sim_spec, axis=0), color='black', linestyle='--', lw=2,
                               label='Selected - $\sigma_{ln}$')
                ax[1].set_xlabel('Period [sec]')
                ax[1].set_ylabel('Dispersion')
                ax[1].grid(True)
                ax[1].legend(frameon=False)
                ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
                ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax[1].set_xlim([self.T[0], self.T[-1]])
                ax[1].set_ylim(bottom=0)
                if self.cond == 1:
                    ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

            if save == 1:
                plt.savefig(os.path.join(self.outdir, 'Simulated.pdf'))

        if rec == 1:
            # Plot Target spectrum vs. Selected response spectra
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            plt.suptitle('Target Spectrum vs. Spectra of Selected Records', y=0.95)

            for i in range(self.nGM):
                ax[0].loglog(self.T, np.exp(self.rec_spec[i, :]), color='gray', lw=1, label='Selected')

            ax[0].loglog(self.T, np.exp(self.mu_ln), color='red', lw=2, label='Target - $e^{\mu_{ln}}$')
            if self.useVar == 1:
                ax[0].loglog(self.T, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                             label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
                ax[0].loglog(self.T, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=2,
                             label='Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

            ax[0].loglog(self.T, np.exp(np.mean(self.rec_spec, axis=0)), color='blue', lw=2,
                         label='Selected - $e^{\mu_{ln}}$')
            ax[0].loglog(self.T, np.exp(np.mean(self.rec_spec, axis=0) + 2 * np.std(self.rec_spec, axis=0)),
                         color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
            ax[0].loglog(self.T, np.exp(np.mean(self.rec_spec, axis=0) - 2 * np.std(self.rec_spec, axis=0)),
                         color='blue', linestyle='--', lw=2, label='Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')

            ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])
            ax[0].set_xlabel('Period [sec]')
            ax[0].set_ylabel('Spectral Acceleration [g]')
            ax[0].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
            ax[0].set_xlim([self.T[0], self.T[-1]])
            if self.cond == 1:
                ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

            # Sample and target standard deviations
            ax[1].semilogx(self.T, self.sigma_ln, color='red', linestyle='--', lw=2, label='Target - $\sigma_{ln}$')
            ax[1].semilogx(self.T, np.std(self.rec_spec, axis=0), color='black', linestyle='--', lw=2,
                           label='Selected - $\sigma_{ln}$')
            ax[1].set_xlabel('Period [sec]')
            ax[1].set_ylabel('Dispersion')
            ax[1].grid(True)
            ax[1].legend(frameon=False)
            ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 4])
            ax[0].set_xlim([self.T[0], self.T[-1]])
            ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax[1].set_ylim(bottom=0)
            if self.cond == 1:
                ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)

            if save == 1:
                plt.savefig(os.path.join(self.outdir, 'Selected.pdf'))

        # Show the figure
        if show == 1:
            plt.show()


#############################################################################################
#############################################################################################

class tbdy_2018(downloader, file_manager):
    """
    This class is used to
        1) Create target spectrum based on TBDY2018
        2) Selecting and scaling suitable ground motion sets for target spectrum in accordance with TBDY2018
            - Currently, only supports the record selection from NGA_W2 record database
    """

    def __init__(self, database='NGA_W2', outdir='Outputs'):
        """
        Details
        -------
        Loads the record database to use

        Parameters
        ----------
        database : str, optional
            database to use: e.g. NGA_W2.
            The default is NGA_W2.
        outdir : str, optional
            output directory
            The default is 'Outputs'

        Returns
        -------
        None.
        """

        # Add the input the ground motion database to use
        matfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Meta_Data', database)
        self.database = loadmat(matfile, squeeze_me=True)
        self.database['Name'] = database
        # create the output directory and add the path to self
        cwd = os.getcwd()
        outdir_path = os.path.join(cwd, outdir)
        self.outdir = outdir_path
        self.create_dir(self.outdir)

        # initialize the objects being used
        downloader.__init__(self)
        file_manager.__init__(self)

    @staticmethod
    def get_Sae(T, Lat, Long, DD, Soil):
        """
        Details
        -------
        This method creates the target spectrum

        References
        ----------

        Notes
        -----

        Parameters
        ----------
        Lat: float
            Site latitude
        Long: float
            Site longitude
        DD:  int
            Earthquake ground motion intensity level (1,2,3,4)
        Soil: str
            Site soil class
        T:  numpy.array
            period array in which target spectrum is calculated

        Returns
        -------
        Sae: numpy.array
            Elastic acceleration response spectrum
        """
        excel_file = 'Parameters_TBDY2018.xlsx'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Meta_Data', excel_file)
        data = pd.read_excel(file_path, sheet_name = 'Parameters', engine='openpyxl') 
        
        # Check if the coordinates are within the limits
        if Long > np.max(data['Longitude']) or Long < np.min(data['Longitude']):
            raise ValueError('Longitude value must be within the limits: [24.55,45.95]')
        if Lat > np.max(data['Latitude']) or Lat < np.min(data['Latitude']):
            raise ValueError('Latitude value must be within the limits: [34.25,42.95]')
            
        # Targeted probability of exceedance in 50 years
        if DD == 1: PoE = '2'
        elif DD == 2: PoE = '10'
        elif DD == 3: PoE = '50'
        elif DD == 4: PoE = '68'
        
        # Determine Peak Ground Acceleration PGA [g]
        PGA_col = 'PGA (g) - %'+PoE
        data_pga = np.array([data['Longitude'], data['Latitude'],data[PGA_col]]).T
        PGA = interpolate.griddata(data_pga[:,0:2], data_pga[:,2], [(Long, Lat)], method='linear')
        
        # Short period map spectral acceleration coefficient [dimensionless]
        SS_col = 'SS (g) - %'+PoE
        data_ss = np.array([data['Longitude'], data['Latitude'],data[SS_col]]).T
        SS = interpolate.griddata(data_ss[:,0:2], data_ss[:,2], [(Long, Lat)], method='linear')
        
        # Map spectral acceleration coefficient for a 1.0 second period [dimensionless]
        S1_col = 'S1 (g) - %'+PoE        
        data_s1 = np.array([data['Longitude'], data['Latitude'],data[S1_col]]).T
        S1 = interpolate.griddata(data_s1[:,0:2], data_s1[:,2], [(Long, Lat)], method='linear')

        SoilParam={
            'FS' : {
               'ZA': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
               'ZB': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
               'ZC': [1.3, 1.3, 1.2, 1.2, 1.2, 1.2],
               'ZD': [1.6, 1.4, 1.2, 1.1, 1.0, 1.0],
               'ZE': [2.4, 1.7, 1.3, 1.1, 0.9, 0.8]
               },
        
            'SS' : [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        
            'F1' : {
               'ZA': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
               'ZB': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
               'ZC': [1.5, 1.5, 1.5, 1.5, 1.5, 1.4],
               'ZD': [2.4, 2.2, 2.0, 1.9, 1.8, 1.7],
               'ZE': [4.2, 3.3, 2.8, 2.4, 2.2, 2.0]
               },
        
            'S1' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        
            }
        
        # Local soil response coefficient for the short period region
        if SS <= SoilParam['SS'][0]:
            FS = SoilParam['FS'][Soil][0]
        elif SS > SoilParam['SS'][0] and SS <= SoilParam['SS'][1]:
            FS=(SoilParam['FS'][Soil][1]-SoilParam['FS'][Soil][0]) \
            *(SS-SoilParam['SS'][0])/(SoilParam['SS'][1]-SoilParam['SS'][0]) \
            +SoilParam['FS'][Soil][0]
        elif SS > SoilParam['SS'][1] and SS <= SoilParam['SS'][2]:
            FS=(SoilParam['FS'][Soil][2]-SoilParam['FS'][Soil][1]) \
            *(SS-SoilParam['SS'][1])/(SoilParam['SS'][2]-SoilParam['SS'][1]) \
            +SoilParam['FS'][Soil][1]
        elif SS > SoilParam['SS'][2] and SS <= SoilParam['SS'][3]:
            FS=(SoilParam['FS'][Soil][3]-SoilParam['FS'][Soil][2]) \
            *(SS-SoilParam['SS'][2])/(SoilParam['SS'][3]-SoilParam['SS'][2]) \
            +SoilParam['FS'][Soil][2]
        elif SS > SoilParam['SS'][3] and SS <= SoilParam['SS'][4]:
            FS=(SoilParam['FS'][Soil][4]-SoilParam['FS'][Soil][3]) \
            *(SS-SoilParam['SS'][3])/(SoilParam['SS'][4]-SoilParam['SS'][3]) \
            +SoilParam['FS'][Soil][3]
        elif SS > SoilParam['SS'][4] and SS <= SoilParam['SS'][5]:
            FS=(SoilParam['FS'][Soil][5]-SoilParam['FS'][Soil][4]) \
            *(SS-SoilParam['SS'][4])/(SoilParam['SS'][5]-SoilParam['SS'][4]) \
            +SoilParam['FS'][Soil][4]
        elif SS >= SoilParam['SS'][5]:
            FS = SoilParam['FS'][Soil][5]
        
        # Local soil response coefficient for 1.0 second period
        if S1 <= SoilParam['S1'][0]:
            F1 = SoilParam['F1'][Soil][0]
        elif S1 > SoilParam['S1'][0] and S1 <= SoilParam['S1'][1]:
            F1=(SoilParam['F1'][Soil][1]-SoilParam['F1'][Soil][0]) \
            *(S1-SoilParam['S1'][0])/(SoilParam['S1'][1]-SoilParam['S1'][0]) \
            +SoilParam['F1'][Soil][0]
        elif S1 > SoilParam['S1'][1] and S1 <= SoilParam['S1'][2]:
            F1=(SoilParam['F1'][Soil][2]-SoilParam['F1'][Soil][1]) \
            *(S1-SoilParam['S1'][1])/(SoilParam['S1'][2]-SoilParam['S1'][1]) \
            +SoilParam['F1'][Soil][1]
        elif S1 > SoilParam['S1'][2] and S1 <= SoilParam['S1'][3]:
            F1=(SoilParam['F1'][Soil][3]-SoilParam['F1'][Soil][2]) \
            *(S1-SoilParam['S1'][2])/(SoilParam['S1'][3]-SoilParam['S1'][2]) \
            +SoilParam['F1'][Soil][2]
        elif S1 > SoilParam['S1'][3] and S1 <= SoilParam['S1'][4]:
            F1=(SoilParam['F1'][Soil][4]-SoilParam['F1'][Soil][3]) \
            *(S1-SoilParam['S1'][3])/(SoilParam['S1'][4]-SoilParam['S1'][3]) \
            +SoilParam['F1'][Soil][3]
        elif S1 > SoilParam['S1'][4] and S1 <= SoilParam['S1'][5]:
            F1=(SoilParam['F1'][Soil][5]-SoilParam['F1'][Soil][4]) \
            *(S1-SoilParam['S1'][4])/(SoilParam['S1'][5]-SoilParam['S1'][4]) \
            +SoilParam['F1'][Soil][4]
        elif S1 >= SoilParam['S1'][5]:
            F1 = SoilParam['F1'][Soil][5]
        
        SDS = SS*FS # short period spectral acceleration coefficient
        SD1 = S1*F1 # spectral acceleration coefficient for 1.0
        
        Sae = np.zeros(len(T))

        TA = 0.2 * SD1 / SDS
        TB = SD1 / SDS
        TL = 6

        for i in range(len(T)):
            if T[i] == 0:
                Sae[i] = PGA
            elif T[i] <= TA:
                Sae[i] = (0.4 + 0.6 * T[i] / TA) * SDS
            elif TA < T[i] <= TB:
                Sae[i] = SDS
            elif TB < T[i] <= TL:
                Sae[i] = SD1 / T[i]
            elif T[i] > TL:
                Sae[i] = SD1 * TL / T[i] ** 2

        return Sae

    def search_database(self):
        """
        Details
        -------
        Searches the record database and does the filtering.

        Parameters
        ----------
        None.

        Returns
        -------
        sampleBig : numpy.array
            An array which contains the IMLs from filtered database.
        soil_Vs30 : numpy.array
            An array which contains the Vs30s from filtered database.
        magnitude : numpy.array
            An array which contains the magnitudes from filtered database.
        Rjb : numpy.array
            An array which contains the Rjbs from filtered database.
        mechanism : numpy.array
            An array which contains the fault type info from filtered database.
        Filename_1 : numpy.array
            An array which contains the filename of 1st gm component from filtered database.
            If selection is set to 1, it will include filenames of both components.
        Filename_2 : numpy.array
            An array which contains the filenameof 2nd gm component filtered database.
            If selection is set to 1, it will be None value.
        NGA_num : numpy.array
            If NGA_W2 is used as record database, record sequence numbers from filtered
            database will be saved, for other databases this variable is None.
        """

        if self.database['Name'] == "NGA_W2":

            if self.selection == 1:  # SaKnown = Sa_arb
                SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
                soil_Vs30 = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
                Mw = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)
                Rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
                fault = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
                Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
                NGA_num = np.append(self.database['NGA_num'], self.database['NGA_num'], axis=0)
                eq_ID = np.append(self.database['EQID'], self.database['EQID'], axis=0)

            elif self.selection == 2:  # SaKnown = (Sa_1**2+Sa_2**2)**0.5
                SaKnown = np.sqrt(self.database['Sa_1'] ** 2 + self.database['Sa_2'] ** 2)
                soil_Vs30 = self.database['soil_Vs30']
                Mw = self.database['magnitude']
                Rjb = self.database['Rjb']
                fault = self.database['mechanism']
                Filename_1 = self.database['Filename_1']
                Filename_2 = self.database['Filename_2']
                NGA_num = self.database['NGA_num']
                eq_ID = self.database['EQID']

            else:
                raise ValueError('Selection can only be performed for one or two components at the moment, exiting...')

        else:
            raise ValueError('Selection can only be performed using NGA_W2 database at the moment, exiting...')

        perKnown = self.database['Periods']

        # Limiting the records to be considered using the `notAllowed' variable
        # Sa cannot be negative or zero, remove these.
        notAllowed = np.unique(np.where(SaKnown <= 0)[0]).tolist()

        if self.Vs30_lim is not None:  # limiting values on soil exist
            mask = (soil_Vs30 > min(self.Vs30_lim)) * (soil_Vs30 < max(self.Vs30_lim) * np.invert(np.isnan(soil_Vs30)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Mw_lim is not None:  # limiting values on magnitude exist
            mask = (Mw > min(self.Mw_lim)) * (Mw < max(self.Mw_lim) * np.invert(np.isnan(Mw)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.Rjb_lim is not None:  # limiting values on Rjb exist
            mask = (Rjb > min(self.Rjb_lim)) * (Rjb < max(self.Rjb_lim) * np.invert(np.isnan(Rjb)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if self.fault_lim is not None:  # limiting values on mechanism exist
            mask = (fault == self.fault_lim * np.invert(np.isnan(fault)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        # get the unique values
        notAllowed = (list(set(notAllowed)))
        Allowed = [i for i in range(SaKnown.shape[0])]
        for i in notAllowed:
            Allowed.remove(i)

        # Use only allowed records
        SaKnown = SaKnown[Allowed, :]
        soil_Vs30 = soil_Vs30[Allowed]
        Mw = Mw[Allowed]
        Rjb = Rjb[Allowed]
        fault = fault[Allowed]
        eq_ID = eq_ID[Allowed]
        Filename_1 = Filename_1[Allowed]

        if self.selection == 1:
            Filename_2 = None
        else:
            Filename_2 = Filename_2[Allowed]

        if self.database['Name'] == "NGA_W2":
            NGA_num = NGA_num[Allowed]
        else:
            NGA_num = None

        # Match periods (known periods and periods for error computations)
        self.T = perKnown[(perKnown >= 0.2 * self.Tp) * (perKnown <= 1.5 * self.Tp)]

        # Arrange the available spectra for error computations
        recPer = []
        for i in range(len(self.T)):
            recPer.append(np.where(perKnown == self.T[i])[0][0])
        sampleBig = SaKnown[:, recPer]

        # Check for invalid input
        if np.any(np.isnan(sampleBig)):
            raise Warning('NaNs found in input response spectra.',
                          'Fix the response spectra of database.')

        # Check if enough records are available
        if self.nGM > len(NGA_num):
            raise Warning('There are not enough records which satisfy',
                          'the given record selection criteria...',
                          'Please use broaden your selection criteria...')

        return sampleBig, soil_Vs30, Mw, Rjb, fault, eq_ID, Filename_1, Filename_2, NGA_num

    def select(self, Lat=41.0582, Long=29.00951, DD=2, Soil='ZC', nGM=11, selection=1, Tp=1,
               Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, opt=1,
               maxScale=2, weights=[1, 1]):
        """
        Details
        -------
        Select the suitable ground motion set
        in accordance with TBDY 2018.
        
        Rule 1: Mean of selected records should remain above the lower bound target spectra.
            For selection = 1: Sa_rec = (Sa_1 or Sa_2) - lower bound = 1.0 * SaTarget(0.2Tp-1.5Tp) 
            For Selection = 2: Sa_rec = (Sa_1**2+Sa_2**2)**0.5 - lower bound = 1.3 * SaTarget(0.2Tp-1.5Tp) 

        Rule 2: 
            No more than 3 records can be selected from the same event! In other words,
            rec_eqID cannot be the same for more than 3 of the selected records.      

        Rule 3: 
            At least 11 records (or pairs) must be selected.

        Parameters
        ----------
        Lat: float, optional, the default is 41.0582.
            Site latitude
        Long: float, optional, the default is 29.00951.
            Site longitude
        DD:  int, optional, the default is 2.
            Earthquake ground motion intensity level (1,2,3,4)
        Soil: str, optional, the default is 'ZC'.
            Site soil class
        nGM : int, optional, the default is 11.
            Number of records to be selected. 
        selection : int, optional, the default is 1.
            Number of ground motion components to select. 
        Tp : float, optional, the default is 1.
            Predominant period of the structure. 
        Mw_lim : list, optional, the default is None.
            The limiting values on magnitude. 
        Vs30_lim : list, optional, the default is None.
            The limiting values on Vs30. 
        Rjb_lim : list, optional, the default is None.
            The limiting values on Rjb. 
        fault_lim : int, optional, the default is None.
            The limiting fault mechanism. 
            0 for unspecified fault 
            1 for strike-slip fault
            2 for normal fault
            3 for reverse fault
        opt : int, optional, the default is 1.
            If equal to 0, the record set is selected using
            method of “least squares”.
            If equal to 1, the record set selected such that 
            scaling factor is closer to 1.
            If equal to 2, the record set selected such that 
            both scaling factor and standard deviation is lowered.
        maxScale : float, optional, the default is 2.
            Maximum allowed scaling factor, used with opt=2 case.
        weights = list, optional, the default is [1,1].
            Error weights (mean,std), used with opt=2 case.
        
        Returns
        -------

        """

        # Add selection settings to self
        self.nGM = nGM
        self.selection = selection
        self.Mw_lim = Mw_lim
        self.Vs30_lim = Vs30_lim
        self.Rjb_lim = Rjb_lim
        self.fault_lim = fault_lim
        self.Tp = Tp
        self.Lat = Lat
        self.Long = Long
        self.DD = DD
        self.Soil = Soil

        weights = np.array(weights, dtype=float)

        # Search the database and filter
        sampleBig, Vs30, Mw, Rjb, fault, eq_ID, Filename_1, Filename_2, NGA_num = self.search_database()

        # Determine the lower bound spectra
        target_spec = self.get_Sae(self.T, Lat, Long, DD, Soil)
        if selection == 1:
            target_spec = 1.0 * target_spec
        elif selection == 2:
            target_spec = 1.3 * target_spec

        # Sample size of the filtered database
        nBig = sampleBig.shape[0]

        # Find best matches to the target spectrum from ground-motion database
        mse = ((np.matlib.repmat(target_spec, nBig, 1) - sampleBig) ** 2).mean(axis=1)

        recID_sorted = np.argsort(mse)
        recIDs = np.ones(self.nGM, dtype=int) * (-1)
        eqIDs = np.ones(self.nGM, dtype=int) * (-1)
        idx1 = 0
        idx2 = 0
        while idx1 < self.nGM:  # not more than 3 of the records should be from the same event
            tmp1 = recID_sorted[idx2]
            idx2 += 1
            tmp2 = eq_ID[tmp1]
            recIDs[idx1] = tmp1
            eqIDs[idx1] = tmp2
            if np.sum(eqIDs == tmp2) <= 3:
                idx1 += 1

        # Initial selection results - based on MSE
        sampleSmall = sampleBig[recIDs.tolist(), :]
        scaleFac = np.max(target_spec / sampleSmall.mean(axis=0))

        @njit
        def opt_method1(sampleSmall, scaleFac, target_spec, recIDs, eqIDs, minID):
            # Optimize based on scaling factor
            def mean_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].mean())

                return np.array(res)

            def std_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].std())

                return np.array(res)

            for j in range(nBig):
                tmp = eq_ID[j]
                # record should not be repeated and number of eqs from the same event should not exceed 3
                if not np.any(recIDs == j) and np.sum(eqIDs == tmp) <= 2:
                    # Add to the sample the scaled spectra
                    temp = np.zeros((1, len(sampleBig[j, :])))
                    temp[:, :] = sampleBig[j, :]  # get the trial spectra
                    tempSample = np.concatenate((sampleSmall, temp), axis=0)  # add the trial spectra to subset list
                    tempScale = np.max(target_spec / mean_numba(tempSample))  # compute new scaling factor

                    # Should cause improvement
                    if abs(tempScale - 1) <= abs(scaleFac - 1):
                        minID = j
                        scaleFac = tempScale

            return minID, scaleFac

        @njit
        def opt_method2(sampleSmall, scaleFac, target_spec, recIDs, eqIDs, minID, DevTot):
            # Optimize based on dispersion
            def mean_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].mean())

                return np.array(res)

            def std_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].std())

                return np.array(res)

            for j in range(nBig):
                tmp = eq_ID[j]

                # record should not be repeated and number of eqs from the same event should not exceed 3
                if not np.any(recIDs == j) and np.sum(eqIDs == tmp) <= 2:
                    # Add to the sample the scaled spectra
                    temp = np.zeros((1, len(sampleBig[j, :])))
                    temp[:, :] = sampleBig[j, :]  # get the trial spectra
                    tempSample = np.concatenate((sampleSmall, temp), axis=0)  # add the trial spectra to subset list
                    tempScale = np.max(target_spec / mean_numba(tempSample))  # compute new scaling factor
                    tempSig = std_numba(tempSample*tempScale) # Compute standard deviation
                    tempMean = mean_numba(tempSample*tempScale) # Compute mean
                    devSig = np.max(tempSig) # Compute maximum standard deviation
                    devMean = np.max(np.abs(target_spec - tempMean)) # Compute maximum difference in mean
                    tempDevTot = devMean*weights[0] + devSig*weights[1]

                    # Should cause improvement
                    if maxScale > tempScale > 1 / maxScale and tempDevTot <= DevTot:
                        minID = j
                        scaleFac = tempScale
                        DevTot = tempDevTot

            return minID, scaleFac

        # Apply Greedy subset modification procedure to improve selection
        # Use njit to speed up the optimization algorithm
        if opt != 0:
            for i in range(self.nGM):  # Loop for nGM
                minID = recIDs[i]
                devSig = np.max(np.std(sampleSmall * scaleFac, axis=0))  # Compute standard deviation
                devMean = np.max(np.abs(target_spec - np.mean(sampleSmall, axis=0)) * scaleFac)
                DevTot = devMean * weights[0] + devSig * weights[1]
                sampleSmall = np.delete(sampleSmall, i, 0)
                recIDs = np.delete(recIDs, i)
                eqIDs = np.delete(eqIDs, i)

                # Try to add a new spectra to the subset list
                if opt == 1:  # try to optimize scaling factor only (closest to 1)
                    minID, scaleFac = opt_method1(sampleSmall, scaleFac, target_spec, recIDs, eqIDs, minID)
                if opt == 2:  # try to optimize the error (max(mean-target) + max(std))
                    minID, scaleFac = opt_method2(sampleSmall, scaleFac, target_spec, recIDs, eqIDs, minID, DevTot)

                # Add new element in the right slot
                sampleSmall = np.concatenate(
                    (sampleSmall[:i, :], sampleBig[minID, :].reshape(1, sampleBig.shape[1]), sampleSmall[i:, :]),
                    axis=0)
                recIDs = np.concatenate((recIDs[:i], np.array([minID]), recIDs[i:]))
                eqIDs = np.concatenate((eqIDs[:i], np.array([eq_ID[minID]]), eqIDs[i:]))

        recIDs = recIDs.tolist()
        # Add selected record information to self
        self.rec_scale = float(scaleFac)
        self.rec_Vs30 = Vs30[recIDs]
        self.rec_Rjb = Rjb[recIDs]
        self.rec_Mw = Mw[recIDs]
        self.rec_fault = fault[recIDs]
        self.rec_eqID = eq_ID[recIDs]
        self.rec_h1 = Filename_1[recIDs]

        if self.selection == 1:
            self.rec_h2 = None
        elif self.selection == 2:
            self.rec_h2 = Filename_2[recIDs]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recIDs]
        else:
            self.rec_rsn = None

        rec_idxs = []
        if self.selection == 1:
            SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_h1:
                rec_idxs.append(np.where(Filename_1 == rec)[0][0])
            rec_spec = SaKnown[rec_idxs, :]
        elif self.selection == 2:
            for rec in self.rec_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            Sa_1 = self.database['Sa_1'][rec_idxs, :]
            Sa_2 = self.database['Sa_2'][rec_idxs, :]
            rec_spec = (Sa_1 ** 2 + Sa_2 ** 2) ** 0.5

        # Save the results for whole spectral range
        self.rec_spec = rec_spec
        self.T = self.database['Periods']
        if selection == 1:
            self.target = self.get_Sae(self.T, Lat, Long, DD, Soil)
        elif selection == 2:
            self.target = self.get_Sae(self.T, Lat, Long, DD, Soil) * 1.3

        print('Ground motion selection is finished scaling factor is %.3f' % self.rec_scale)

    def plot(self, save=0, show=1):
        """
        Details
        -------
        Plots the target spectrum and spectra 
        of selected records.

        Parameters
        ----------
        save   : int, optional
            Flag to save plotted figures in pdf format.
            The default is 0.
        show  : int, optional
            Flag to show figures
            The default is 1.
            
        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """

        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIG_SIZE = 16
        BIGGER_SIZE = 18

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.ioff()

        hatch = [self.Tp * 0.2, self.Tp * 1.5]
        # Plot Target spectrum vs. Selected response spectra
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for i in range(self.rec_spec.shape[0]):
            ax.plot(self.T, self.rec_spec[i, :] * self.rec_scale, color='gray', lw=1, label='Selected')
        ax.plot(self.T, np.mean(self.rec_spec, axis=0) * self.rec_scale, color='black', lw=2, label='Selected Mean')
        ax.plot(self.T, self.target, color='red', lw=2, label='Target')
        ax.axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
        ax.set_xlabel('Period [sec]')
        ax.set_ylabel('Spectral Acceleration [g]')
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False)
        ax.set_xlim([self.T[0], self.Tp * 3])
        plt.suptitle('Target Spectrum vs. Spectra of Selected Records', y=0.95)

        if save == 1:
            plt.savefig(os.path.join(self.outdir, 'Selected.pdf'))

        # Show the figure
        if show == 1:
            plt.show()


#############################################################################################
#############################################################################################


class ec8_part1(downloader, file_manager):
    """
    This class is used to
        1) Create target spectrum based on EC8 - Part 1
        2) Selecting and scaling suitable ground motion sets for target spectrum in accordance with EC8 - Part 1
            - Currently, only supports the record selection from NGA_W2 record database
    """

    def __init__(self, database='NGA_W2', outdir='Outputs'):
        """
        Details
        -------
        Loads the record database to use

        Parameters
        ----------
        database : str, optional
            database to use: e.g. NGA_W2.
            The default is NGA_W2.
        outdir : str, optional
            output directory
            The default is 'Outputs'

        Returns
        -------
        None.
        """

        # initialize the objects being used
        downloader.__init__(self)
        file_manager.__init__(self)

        # Add the input the ground motion database to use
        matfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Meta_Data', database)
        self.database = loadmat(matfile, squeeze_me=True)
        self.database['Name'] = database
        # create the output directory and add the path to self
        cwd = os.getcwd()
        outdir_path = os.path.join(cwd, outdir)
        self.outdir = outdir_path
        self.create_dir(self.outdir)
        
    @staticmethod
    def get_EC804_spectrum_Sa_el(ag,xi,T,I,Type,Soil):
        """
        Details:
        Get the elastic response spectrum for EN 1998-1:2004
    
        Notes:
        Requires get_EC804_spectrum_props
    
        References:
        CEN. Eurocode 8: Design of Structures for Earthquake Resistance -
        Part 1: General Rules, Seismic Actions and Rules for Buildings
        (EN 1998-1:2004). Brussels, Belgium: 2004.
    
        Inputs:
        ag: PGA
        xi: Damping
        T: Period
        I: Importance factor
        Type: Type of spectrum (Option: 'Type1' or 'Type2')
        Soil: Soil Class (Options: 'A', 'B', 'C', 'D' or 'E')
    
        Returns:
        Sa: Spectral acceleration
    
        """
        SpecProp={
            'Type1' : {
               'A': {'S':  1.00, 'Tb': 0.15, 'Tc': 0.4, 'Td': 2.0},
               'B': {'S':  1.20, 'Tb': 0.15, 'Tc': 0.5, 'Td': 2.0},
               'C': {'S':  1.15, 'Tb': 0.20, 'Tc': 0.6, 'Td': 2.0},
               'D': {'S':  1.35, 'Tb': 0.20, 'Tc': 0.8, 'Td': 2.0},
               'E': {'S':  1.40, 'Tb': 0.15, 'Tc': 0.5, 'Td': 2.0},
               },
    
            'Type2' : {
               'A': {'S':  1.00, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
               'B': {'S':  1.35, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
               'C': {'S':  1.50, 'Tb': 0.10, 'Tc': 0.25, 'Td': 1.2},
               'D': {'S':  1.80, 'Tb': 0.10, 'Tc': 0.30, 'Td': 1.2},
               'E': {'S':  1.60, 'Tb': 0.05, 'Tc': 0.25, 'Td': 1.2},
            }
        }
    
        S=SpecProp[Type][Soil]['S']
        Tb=SpecProp[Type][Soil]['Tb']
        Tc=SpecProp[Type][Soil]['Tc']
        Td=SpecProp[Type][Soil]['Td']
    
        eta=max(np.sqrt(0.10/(0.05+xi)),0.55)

        ag = ag*I

        Sa = []
        for i in range(len(T)):
            if T[i] >= 0 and T[i] <= Tb:
                Sa_el=ag*S*(1.0+T[i]/Tb*(2.5*eta-1.0))
            elif T[i] >= Tb and T[i] <= Tc:
                Sa_el=ag*S*2.5*eta
            elif T[i] >= Tc and T[i] <= Td:
                Sa_el=ag*S*2.5*eta*(Tc/T[i])
            elif T[i] >= Td:
                Sa_el=ag*S*2.5*eta*(Tc*Td/T[i]/T[i])
            else:
                print('Error! Cannot compute a value of Sa_el')
            
            Sa.append(Sa_el)
            
        return np.array(Sa)


    def search_database(self):
        """
        Details
        -------
        Searches the record database and does the filtering.

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

        if self.database['Name'] == "NGA_W2":

            if self.selection == 1:  # SaKnown = Sa_arb
                SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
                soil_Vs30 = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
                Mw = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)
                Rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
                fault = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
                Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
                NGA_num = np.append(self.database['NGA_num'], self.database['NGA_num'], axis=0)
                eq_ID = np.append(self.database['EQID'], self.database['EQID'], axis=0)

            elif self.selection == 2:  # SaKnown = (Sa_1**2+Sa_2**2)**0.5
                SaKnown = (self.database['Sa_1'] + self.database['Sa_2'])/2
                soil_Vs30 = self.database['soil_Vs30']
                Mw = self.database['magnitude']
                Rjb = self.database['Rjb']
                fault = self.database['mechanism']
                Filename_1 = self.database['Filename_1']
                Filename_2 = self.database['Filename_2']
                NGA_num = self.database['NGA_num']
                eq_ID = self.database['EQID']

            else:
                print('Selection can only be performed for one or two components at the moment, exiting...')
                sys.exit()

        else:
            print('Selection can only be performed using NGA_W2 database at the moment, exiting...')
            sys.exit()

        perKnown = self.database['Periods']

        # Limiting the records to be considered using the `notAllowed' variable
        # Sa cannot be negative or zero, remove these.
        notAllowed = np.unique(np.where(SaKnown <= 0)[0]).tolist()

        if not self.Vs30_lim is None:  # limiting values on soil exist
            mask = (soil_Vs30 > min(self.Vs30_lim)) * (soil_Vs30 < max(self.Vs30_lim) * np.invert(np.isnan(soil_Vs30)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if not self.Mw_lim is None:  # limiting values on magnitude exist
            mask = (Mw > min(self.Mw_lim)) * (Mw < max(self.Mw_lim) * np.invert(np.isnan(Mw)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if not self.Rjb_lim is None:  # limiting values on Rjb exist
            mask = (Rjb > min(self.Rjb_lim)) * (Rjb < max(self.Rjb_lim) * np.invert(np.isnan(Rjb)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        if not self.fault_lim is None:  # limiting values on mechanism exist
            mask = (fault == self.fault_lim * np.invert(np.isnan(fault)))
            temp = [i for i, x in enumerate(mask) if not x]
            notAllowed.extend(temp)

        # get the unique values
        notAllowed = (list(set(notAllowed)))
        Allowed = [i for i in range(SaKnown.shape[0])]
        for i in notAllowed:
            Allowed.remove(i)

        # Use only allowed records
        SaKnown = SaKnown[Allowed, :]
        soil_Vs30 = soil_Vs30[Allowed]
        Mw = Mw[Allowed]
        Rjb = Rjb[Allowed]
        fault = fault[Allowed]
        eq_ID = eq_ID[Allowed]
        Filename_1 = Filename_1[Allowed]

        if self.selection == 1:
            Filename_2 = None
        else:
            Filename_2 = Filename_2[Allowed]

        if self.database['Name'] == "NGA_W2":
            NGA_num = NGA_num[Allowed]
        else:
            NGA_num = None

        # Match periods (known periods and periods for error computations)
        # Add Sa(T=0) or PGA, approximated as Sa(T=0.01)
        self.T = np.append(perKnown[0],perKnown[(perKnown >= 0.2 * self.Tp) * (perKnown <= 2.0 * self.Tp)])

        # Arrange the available spectra for error computations
        recPer = []
        for i in range(len(self.T)):
            recPer.append(np.where(perKnown == self.T[i])[0][0])
        sampleBig = SaKnown[:, recPer]

        # Check for invalid input
        if np.any(np.isnan(sampleBig)):
            print('NaNs found in input response spectra.',
                  'Fix the response spectra of database.')
            sys.exit()

        # Check if enough records are available
        if self.nGM > len(NGA_num):
            print('There are not enough records which satisfy',
                  'the given record selection criteria...',
                  'Please use broaden your selection criteria...')
            sys.exit()

        return sampleBig, soil_Vs30, Mw, Rjb, fault, eq_ID, Filename_1, Filename_2, NGA_num

    def select(self, ag=0.2,xi=0.05, I=1.0, Type='Type1',Soil='A', nGM=3, selection=1, Tp=1,
               Mw_lim=None, Vs30_lim=None, Rjb_lim=None, fault_lim=None, opt=1, 
               maxScale=2, weights = [1,1]):
        """
        Details
        -------
        Select the suitable ground motion set
        in accordance with EC8 - PART 1
        
        Mean of selected records should remain above the lower bound target spectra.
            For selection = 1: Sa_rec = (Sa_1 or Sa_2) - lower bound = 0.9 * SaTarget(0.2Tp-1.5Tp) 
            For Selection = 2: Sa_rec = (Sa_1**2+Sa_2**2)**0.5 - lower bound = 0.9 * SaTarget(0.2Tp-1.5Tp) 
            Always Sa(T=0) > Sa(T=0)_target
            
        Parameters
        ----------
        ag:  float, optional, the default is 0.25.
            Peak ground acceleration [g]
        xi: float, optional, the default is 0.05.
            Damping
        I:  float, optional, the default is 1.2.
            importance factor
        Type: str, optional, the default is 'Type1'
            Type of spectrum (Option: 'Type1' or 'Type2')
        Soil: str, optional, the default is 'B'
            Soil Class (Options: 'A', 'B', 'C', 'D' or 'E')
        nGM : int, optional, the default is 11.
            Number of records to be selected. 
        selection : int, optional, the default is 1.
            Number of ground motion components to select. 
        Tp : float, optional, the default is 1.
            Predominant period of the structure. 
        Mw_lim : list, optional, the default is None.
            The limiting values on magnitude. 
        Vs30_lim : list, optional, the default is None.
            The limiting values on Vs30. 
        Rjb_lim : list, optional, the default is None.
            The limiting values on Rjb. 
        fault_lim : int, optional, the default is None.
            The limiting fault mechanism. 
            0 for unspecified fault 
            1 for strike-slip fault
            2 for normal fault
            3 for reverse fault
        opt : int, optional, the default is 1.
            If equal to 0, the record set is selected using
            method of “least squares”.
            If equal to 1, the record set selected such that 
            scaling factor is closer to 1.
            If equal to 2, the record set selected such that 
            both scaling factor and standard deviation is lowered.
        maxScale : float, optional, the default is 2.
            Maximum allowed scaling factor, used with opt=2 case.
        weights = list, optional, the default is [1,1].
            Error weights (mean,std), used with opt=2 case.

        Returns
        -------

        """

        # Add selection settings to self
        self.nGM = nGM
        self.selection = selection
        self.Mw_lim = Mw_lim
        self.Vs30_lim = Vs30_lim
        self.Rjb_lim = Rjb_lim
        self.fault_lim = fault_lim
        self.Tp = Tp
        self.ag = ag
        self.I = I
        self.Type = Type
        self.Soil = Soil

        weights = np.array(weights, dtype=float)
        # Search the database and filter
        sampleBig, Vs30, Mw, Rjb, fault, eq_ID, Filename_1, Filename_2, NGA_num = self.search_database()

        # Determine the lower bound spectra
        target_spec = self.get_EC804_spectrum_Sa_el(ag,xi,self.T,I,Type,Soil)
        target_spec[1:] = 0.9 * target_spec[1:] # lower bound spectra except PGA

        # Sample size of the filtered database
        nBig = sampleBig.shape[0]

        # Find best matches to the target spectrum from ground-motion database
        mse = ((np.matlib.repmat(target_spec, nBig, 1) - sampleBig) ** 2).mean(axis=1)
        recID_sorted = np.argsort(mse)
        recIDs = recID_sorted[:self.nGM]

        # Initial selection results - based on MSE
        sampleSmall = sampleBig[recIDs.tolist(), :]
        scaleFac = np.max(target_spec / sampleSmall.mean(axis=0))

        # Apply Greedy subset modification procedure to improve selection
        # Use njit to speed up the optimization algorithm
        @njit
        def opt_method1(sampleSmall, scaleFac, target_spec, recIDs, minID):

            def mean_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].mean())

                return np.array(res)

            for j in range(nBig):
                if not np.any(recIDs == j):
                    # Add to the sample the scaled spectra
                    temp = np.zeros((1, len(sampleBig[j, :])));
                    temp[:, :] = sampleBig[j, :]  # get the trial spectra
                    tempSample = np.concatenate((sampleSmall, temp), axis=0)  # add the trial spectra to subset list
                    tempScale = np.max(target_spec / mean_numba(tempSample))  # compute new scaling factor

                    # Should cause improvement and record should not be repeated
                    if abs(tempScale - 1) <= abs(scaleFac - 1):
                        minID = j
                        scaleFac = tempScale

            return minID, scaleFac

        @njit
        def opt_method2(sampleSmall, scaleFac, target_spec, recIDs, minID, DevTot):
            # Optimize based on dispersion
            def mean_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].mean())

                return np.array(res)
            
            def std_numba(a):

                res = []
                for i in range(a.shape[1]):
                    res.append(a[:, i].std())

                return np.array(res)

            for j in range(nBig):
                
                # record should not be repeated
                if not np.any(recIDs == j):
                    # Add to the sample the scaled spectra
                    temp = np.zeros((1, len(sampleBig[j, :])));
                    temp[:, :] = sampleBig[j, :]  # get the trial spectra
                    tempSample = np.concatenate((sampleSmall, temp), axis=0)  # add the trial spectra to subset list
                    tempScale = np.max(target_spec / mean_numba(tempSample))  # compute new scaling factor
                    tempSig = std_numba(tempSample*tempScale) # Compute standard deviation
                    tempMean = mean_numba(tempSample*tempScale) # Compute mean
                    devSig = np.max(tempSig) # Compute maximum standard deviation
                    devMean = np.max(np.abs(target_spec - tempMean)) # Compute maximum difference in mean
                    tempDevTot = devMean*weights[0] + devSig*weights[1]
                    
                    # Should cause improvement
                    if tempScale<maxScale and tempScale>1/maxScale and tempDevTot <= DevTot:
                        minID = j
                        scaleFac = tempScale
                        DevTot = tempDevTot

            return minID, scaleFac
        
        # Apply Greedy subset modification procedure to improve selection
        # Use njit to speed up the optimization algorithm
        if opt != 0:
            for i in range(self.nGM):  # Loop for nGM
                minID = recIDs[i]
                devSig = np.max(np.std(sampleSmall*scaleFac,axis=0)) # Compute standard deviation
                devMean = np.max(np.abs(target_spec - np.mean(sampleSmall,axis=0))*scaleFac)
                DevTot = devMean*weights[0] + devSig*weights[1]
                sampleSmall = np.delete(sampleSmall, i, 0)
                recIDs = np.delete(recIDs, i)
                
                # Try to add a new spectra to the subset list
                if opt == 1: # try to optimize scaling factor only (closest to 1)
                    minID, scaleFac = opt_method1(sampleSmall, scaleFac, target_spec, recIDs, minID)
                if opt == 2: # try to optimize the error (max(mean-target) + max(std))
                    minID, scaleFac = opt_method2(sampleSmall, scaleFac, target_spec, recIDs, minID, DevTot)

                # Add new element in the right slot
                sampleSmall = np.concatenate(
                    (sampleSmall[:i, :], sampleBig[minID, :].reshape(1, sampleBig.shape[1]), sampleSmall[i:, :]),
                    axis=0)
                recIDs = np.concatenate((recIDs[:i], np.array([minID]), recIDs[i:]))

        recIDs = recIDs.tolist()
        # Add selected record information to self
        self.rec_scale = float(scaleFac)
        self.rec_Vs30 = Vs30[recIDs]
        self.rec_Rjb = Rjb[recIDs]
        self.rec_Mw = Mw[recIDs]
        self.rec_fault = fault[recIDs]
        self.rec_eqID = eq_ID[recIDs]
        self.rec_h1 = Filename_1[recIDs]

        if self.selection == 1:
            self.rec_h2 = None
        elif self.selection == 2:
            self.rec_h2 = Filename_2[recIDs]

        if self.database['Name'] == 'NGA_W2':
            self.rec_rsn = NGA_num[recIDs]
        else:
            self.rec_rsn = None

        rec_idxs = []
        if self.selection == 1:
            SaKnown = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            Filename_1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            for rec in self.rec_h1:
                rec_idxs.append(np.where(Filename_1 == rec)[0][0])
            self.rec_spec = SaKnown[rec_idxs, :]
            
        elif self.selection == 2:
            for rec in self.rec_h1:
                rec_idxs.append(np.where(self.database['Filename_1'] == rec)[0][0])
            self.rec_spec1 = self.database['Sa_1'][rec_idxs, :]
            self.rec_spec2 = self.database['Sa_2'][rec_idxs, :]

        # Save the results for whole spectral range
        self.T = self.database['Periods']
        self.design_spectrum = self.get_EC804_spectrum_Sa_el(ag,xi,self.T,I,Type,Soil)
            
        print('Ground motion selection is finished scaling factor is %.3f' % self.rec_scale)

    def plot(self, save=0, show=1):
        """
        Details
        -------
        Plots the target spectrum and spectra 
        of selected records.

        Parameters
        ----------
        save   : int, optional
            Flag to save plotted figures in pdf format.
            The default is 0.
        show  : int, optional
            Flag to show figures
            The default is 1.
            
        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """

        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIG_SIZE = 16
        BIGGER_SIZE = 18

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.ioff()

        hatch = [self.Tp * 0.2, self.Tp * 2.0]
        # Plot Target spectrum vs. Selected response spectra
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if self.selection == 1:
            for i in range(self.rec_spec.shape[0]):
                ax.plot(self.T, self.rec_spec[i, :] * self.rec_scale, color='gray', lw=1, label='Selected')
            ax.plot(self.T, np.mean(self.rec_spec, axis=0) * self.rec_scale, color='black', lw=2, label='Selected Mean')
        if self.selection == 2:
            for i in range(self.rec_spec1.shape[0]):
                ax.plot(self.T, self.rec_spec1[i, :] * self.rec_scale, color='gray', lw=1, label='Selected')
                ax.plot(self.T, self.rec_spec2[i, :] * self.rec_scale, color='gray', lw=1, label='Selected')
            ax.plot(self.T, np.mean((self.rec_spec1+self.rec_spec2)/2, axis=0) * self.rec_scale, color='black', lw=2, label='Selected Mean')
        ax.plot(self.T, self.design_spectrum, color='blue', lw=2, label='Design Spectrum')
        ax.plot(self.T, 0.9*self.design_spectrum, color='blue', lw=2, ls='--', label='0.9 x Design Spectrum')
        ax.axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
        ax.set_xlabel('Period [sec]')
        ax.set_ylabel('Spectral Acceleration [g]')
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False)
        ax.set_xlim([self.T[0], self.Tp * 3])
        plt.suptitle('Target Spectrum vs. Spectra of Selected Records', y=0.95)

        if save == 1:
            plt.savefig(os.path.join(self.outdir, 'Selected.pdf'))

        # Show the figure
        if show == 1:
            plt.show()
