import numpy as np
from scipy.special import erf
from astropy.modeling.models import Gaussian1D, custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.convolution import Box1DKernel, convolve
from astropy.stats import sigma_clip

__all__ = ['clip_outliers', 'gauss_removal']
 
def clip_outliers(data, log, wavelength, sigma=10, box_width=5, maxiters=5, boundary='extend', fill_value='mask', verbose=False):
    '''Find outliers in 1D time series.
  
    Be careful when using this function on a time-series with known astrophysical variations. The variable
    box_width should be set to be significantly smaller than any astrophysical variation timescales otherwise
    these signals may be clipped.
   
    Parameters
    ----------
    data: ndarray (1D, float)
        The input array in which to identify outliers
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    wavelength: float
        The wavelength currently under consideration.
    sigma: float
        The number of sigmas a point must be from the rolling mean to be considered an outlier
    box_width: int
        The width of the box-car filter (used to calculated the rolling median) in units of number of data points
    maxiters: int
        The number of iterations of sigma clipping that should be performed.
    fill_value: string or float
        Either the string 'mask' to mask the outlier values, 'boxcar' to replace data with the mean from the box-car filter, or a constant float-type fill value.
  
    Returns
    -------
    data:   ndarray (1D, boolean)
        An array with the same dimensions as the input array with outliers replaced with fill_value.
  
    Notes
    -----
    History:
  
    - Jan 29-31, 2022 Taylor Bell
        Initial version, added logging
    '''
    kernel = Box1DKernel(box_width)
    # Compute the moving mean
    bound_val = np.ma.median(data) # Only used if boundary=='fill'
    smoothed_data = convolve(data, kernel, boundary=boundary, fill_value=bound_val)
    # Compare data to the moving mean (to remove astrophysical signals)
    residuals = data-smoothed_data
    # Sigma clip residuals to find bad points in data
    residuals = sigma_clip(residuals, sigma=sigma, maxiters=maxiters, cenfunc=np.ma.median)
    outliers = np.ma.getmaskarray(residuals)
  
    if np.any(outliers):
        log.writelog('Identified {} outliers for wavelength {}'.format(np.sum(outliers), wavelength), mute=(not verbose))
  
    # Replace clipped data
    if fill_value=='mask':
        data = np.ma.masked_where(outliers, data)
    elif fill_value=='boxcar':
        data = replace_moving_mean(data, outliers, kernel)
    else:
        data[outliers] = fill_value
  
    return data, np.sum(outliers)
 
def replace_moving_mean(data, outliers, kernel):
    '''Replace clipped values with the mean from a moving mean.
   
    Parameters
    ----------
    data: ndarray (1D, float)
        The input array in which to replace outliers
    outliers: ndarray (1D, bool)
        The input array in which to replace outliers
    kernel: astropy.convolution.Kernel1D
        The kernel used to compute the moving mean.
  
    Returns
    -------
    data:   ndarray (boolean)
        An array with the same dimensions as the input array with outliers replaced with fill_value.
  
    Notes
    -----
    History:
  
    - Jan 29, 2022 Taylor Bell
        Initial version
    '''
    # First set outliers to NaN so they don't bias moving mean
    data[outliers] = np.nan
    smoothed_data = convolve(data, kernel, boundary='extend')
    # Replace outliers with value of moving mean
    data[outliers] = smoothed_data[outliers]
  
    return data


def skewed_gaussian(x, eta=0, omega=1, alpha=0,scale=1):
    """                     
    A skewed gaussian model.
    """
    t = alpha * (x - eta) / omega
    Psi = 0.5 * (1 + erf(t / np.sqrt(2)))
    psi = 2.0 / (omega * np.sqrt(2 * np.pi)) * np.exp(- (x-eta)**2 / (2.0 * omega**2))
    return (psi * Psi)*scale
 

def gauss_removal(img, mask, linspace, where='bkg'):
    """
    An additional step to remove cosmic rays. This fits a Gaussian to
    the background (or a skewed Gaussian to the orders) and masks data
    points which are above a certain sigma.   

    Parameters
    ----------
    img : np.ndarray
       Single exposure image.
    mask : np.ndarray       
       An approximate mask for the orders.
    linspace : array               
       Sets the lower and upper bin bounds for the
       pixel values. Should be of length = 2.    
    where : str, optional                        
       Sets where the mask is covering. Default is `bkg`.
       Other option is `order`.  

    Returns    
    -------    
    img : np.ndarray
       The same input image, now masked for newly identified
       outliers.   
    """
    n, bins = np.histogram((img*mask).flatten(),
                                    bins=np.linspace(linspace[0],linspace[1],100))
    bincenters = (bins[1:]+bins[:-1])/2

    if where=='bkg':
        g = Gaussian1D(mean=0,amplitude=100,stddev=10)
        rmv = np.where(np.abs(bincenters)<=5)[0]
    elif where=='order':
        GaussianSkewed = custom_model(skewed_gaussian)
        g = GaussianSkewed(eta=0,omega=20,alpha=4, scale=100)
        rmv = np.where(np.abs(bincenters)==0)[0]

    # finds bin centers and removes bincenter = 0 (because this bin
    #   seems to be enormous and we don't want to skew the best-fit
    bincenters, n = np.delete(bincenters, rmv), np.delete(n,rmv)

    # fit the model to the histogram bins
    fitter = LevMarLSQFitter()
    gfit = fitter(g, bincenters, n)

    if where=='bkg':
        xcr, ycr = np.where(np.abs(img*mask)>=gfit.mean+2*gfit.stddev)
    elif where=='order':
        xcr, ycr = np.where(img*mask<=gfit.eta-1*gfit.omega)

    # returns an image that is nan-masked  
    img[xcr,ycr] = np.nan
    return img
