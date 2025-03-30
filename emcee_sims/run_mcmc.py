import emcee
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# Import relevant packages 
import emcee 
import phoebe as pb
from phoebe import u
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as un
import scipy as sp 
import warnings 

pb.atmospheres.passbands._url_tables_server = 'https://staging.phoebe-project.org'
pb.list_online_passbands(refresh=True)
pb.list_all_update_passbands_available()
pb.update_all_passbands()



### for a cadence of 5 mins:
def sample_number(t):
    return 1.87*24*60/t # [days] * [hours/day] * [mins/hour] / [mins]

def upsample_lightcurve(x_upsampled,x,y): 
    '''
    Upsample lightcurve to new time resolution
    '''
    interpolated_lc = sp.interpolate.CubicSpline(x, y)
    return interpolated_lc(x_upsampled)

# Define PHOEBE model 
def get_phoebe_lightcurve(b,t,mask,Teff_ratio,requiv, ecc=0.0137, incl=77.7, period=1.87, gravb=0.9, irrad=0.9,nperiods=3):
    
    #setting chosen binary parameters
    b["ecc@binary"] = ecc
    b["incl@binary"] = incl
    b["period@binary"] = period
    b['teffratio@binary'] = Teff_ratio
    b["requivsumfrac@binary@orbit@component"] = requiv
    
    b["gravb_bol@primary"] = gravb
    b["irrad_frac_refl_bol@primary"] = irrad

    b.run_compute()
    fluxes = b.get_value('fluxes@lc01@model') # normalized flux units
    _fluxes = b.get_value('fluxes@lc01@model') # normalized flux units
    times = b.get_value('times@lc01@model') # days 
    _times = b.get_value('times@lc01@model') # days 
    
        
    for i in range(nperiods-1): 
        fluxes = np.concatenate([fluxes,_fluxes[1:]])
        times = np.concatenate([times, _times[1:] + (i+1)*period])

    flux_model = upsample_lightcurve(t,times,fluxes)

    return flux_model

def log_likelihood(xuse, mask, flux, flux_err, Teff_ratio,requiv):    
    # Define period 
    period = 1.87
    nperiods = 3
    # Define sampling rate
    tsamp = 100 #units?
    nsamp = sample_number(tsamp)
    logger = pb.logger()
    b = pb.default_binary(force_build=True)
    lcnum='lc01'
    b.add_dataset('lc', times=pb.linspace(0,period,int(nsamp)), dataset=lcnum, overwrite=True)
    ### comment out these two lines after running once (if re-running cell)
    b.flip_constraint("requivsumfrac@binary", solve_for="requiv@primary")
    b.flip_constraint("teffratio@binary", solve_for = "teff@primary")

    model_flux = get_phoebe_lightcurve(b,xuse,mask,Teff_ratio,requiv, ecc=0.0137, incl=77.7, period=period, gravb=0.9, irrad=0.9,nperiods=nperiods)
    residual = flux - model_flux
    logL = -0.5 * np.sum((residual / flux_err) ** 2 + np.log(2 * np.pi * flux_err ** 2))
    return logL

# Define the log-prior function with a Gaussian prior on f
def log_prior(Teff_ratio, requiv):

    # Assert uniform priors on ratio of Teff_ratio and requiv
    if (Teff_ratio<0) or (Teff_ratio>1): 
        return -np.inf 

    if (requiv<0) or (requiv>1): 
        return -np.inf

    return 0
        

# Define the log-posterior function
def log_posterior(params, xuse, mask, flux, flux_err):
    Teff_ratio,requiv = params
    lp = log_prior(Teff_ratio,requiv)

    # If outside of uniform prior, reject step
    if not np.isfinite(lp):
        return -np.inf

    # Phoebe has its own internal checks. If it raises an error, reject step 
    try: 
        llhd = log_likelihood(xuse, mask, flux, flux_err, Teff_ratio,requiv)
    except: 
        return -np.inf
        
    return lp + llhd


def guess_initial_pars(nwalkers, ndim=2): 
    # below is setting the intial starting point, it has to be an accept to run
    initial_pos = np.empty((nwalkers,ndim))
    Teff_guess = 0.5 + 0.1*np.random.randn(nwalkers)
    requiv_guess = 0.5 + 0.1*np.random.randn(nwalkers)
    # incl = 90 - 4.5*np.abs(np.randpm.randn(walkers))

    initial_pos[:,0] = Teff_guess
    initial_pos[:,1] = requiv_guess

    return initial_pos


def get_confidence_interval(percentiles): 

    median = percentiles[1]
    upper = percentiles[-1] - median 
    lower = median - percentiles[0]

    return median, upper, lower 



if __name__=='__main__': 
    ndim = 2 
    nwalkers = 4 
    nsteps = 200
    
    p0 = guess_initial_pars(nwalkers,ndim)
    # rename the files for whatever you are running
    # Import simulated light curve to act as "true data" 
    yfull = np.load('fluxes_95percent.npy') # Normalized
    xfull = np.load('t_arr_95percent.npy') # units of days 
    mask = np.load('mask_95percent.npy') 
    xuse = xfull[mask]
    fluxes = yfull[mask]
    
    # For uncertainties, let's assume the same for all data points, based off of our added gaussian noise (mean 0, std = 0.01)
    flux_err = np.ones_like(fluxes)*0.05 # CHANGE WITH SIGMA FROM NOISY FILES
    
    # Initialize mcmc 
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_posterior,args=(xuse,mask,fluxes,flux_err))

    # Run mcmc 
    warnings.filterwarnings(action="ignore") 
    
    sampler.run_mcmc(p0, nsteps, progress=True)
    

    print(
        "Mean acceptance fraction: {0}".format(
            np.mean(sampler.acceptance_fraction)
        )
    )

    samples = sampler.get_chain()
    
    # Discard first 10% of sample to omit burn in period
    flat_samples = sampler.get_chain(discard=int(nsteps*0.1), flat=True)
    
    #Save data to disk for future analysis
    path = 'chains_95percent.npy'
    np.save(path, flat_samples)


    # Plotting 
    # Load in chains 
    import corner 
    # flat_samples = np.load('chains1.npy')
    flat_samples = np.load('chains_95percent.npy')
    
    # Define labels for corner plot
    labels = [r"$T$", r"$requiv$"]
    
    # Generate the corner plot
    fig = corner.corner(flat_samples, labels=labels, show_titles=True,title_fmt=0.1, quantiles=[0.16, 0.5, 0.84])
    plt.show()

    Teff_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    Requiv_mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])

    Teff_fit, upper_Teff, lower_Teff = get_confidence_interval(Teff_mcmc)
    requiv_fit,upper_req,lower_req = get_confidence_interval(Requiv_mcmc)


    # Compute best fit model
        # Define period 
    period = 1.87
    nperiods = 3
    # Define sampling rate
    tsamp = 100 #units? # can update this to run faster/slower
    nsamp = sample_number(tsamp)
    logger = pb.logger()
    b = pb.default_binary(force_build=True)
    lcnum='lc01'
    b.add_dataset('lc', times=pb.linspace(0,period,int(nsamp)), dataset=lcnum, overwrite=True)
    ### comment out these two lines after running once (if re-running cell)
    b.flip_constraint("requivsumfrac@binary", solve_for="requiv@primary")
    b.flip_constraint("teffratio@binary", solve_for = "teff@primary") 
    best_model_flux = get_phoebe_lightcurve(b,xfull,mask,Teff_fit,requiv_fit)

    print(r'''Best fit parameters: 
Teff = {0}(upper err): {1}, (lower err): {2}
Requiv = {3}(upper err): {4}, (lower err): {5}
      '''.format(
          Teff_fit,upper_Teff,lower_Teff,
         requiv_fit,upper_req,lower_req
                
                ))


    # Plot resulting fit on top of data 
    xuse = xfull[mask]
    yuse = yfull[mask]
    
    # Compute residuals 
    residuals = yuse - best_model_flux[mask]
    
    # Plot fit, data and residuals 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2.5, 1.5]})
    
    ax1.plot(xuse,yuse,'.',mew=1,mfc='none',color='black',ls='none')
    ax1.errorbar(xuse,yuse,yerr=flux_err,capsize=1,alpha=1,color='black', fmt='none')
    ax1.plot(xfull, best_model_flux,ls='-',color='cyan')
    
    
    ax2.plot(xuse, residuals, '.', mew=1,mfc='none',color='black',ls='none')
    ax2.errorbar(xuse, residuals, yerr=flux_err,capsize=1,alpha=1,color='black', fmt='none')
    ax2.set_ylabel('Residuals',fontsize=18)
    
    ax2.set_xlabel('Time (days)',fontsize=18)
    ax1.set_ylabel('Normalized Flux (arb.)',fontsize=18)
    ax1.grid(alpha=0.1)
    ax2.grid(alpha=0.1)
    plt.savefig('bestfit_95percent.png')
    
    plt.show()

        # Save best fit model to disk
    np.save('bestfit_95percent.npy',best_model_flux)
    # save best fit parameters to disk
    np.save('bestfit_pars_95percent.npy',np.array([Teff_fit,requiv_fit]))
    




