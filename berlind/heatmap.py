""" Inference code for heat map pixel inference of population posterior distributionp.  """

import numpy as np


def lnlike(alpha, s, errs):
    """ Likelihood function for hyper parameters alpha.
        :alpha:
        Flattened 2D numpy array with NxM-1 heat map values
        :s:
        Flattened array of samples binned onto NxM grid - Nj samples drawn from each, Ni, galaxy
        RETURNS:
        One value of lnlike for given alphas having summed over all the pixels, i.
        """
    #a = np.append(alpha, [1-np.sum(alpha)], axis=0)
    a = alpha
    #return np.sum(a - s - a*np.log(a/s))
    return -0.5*np.sum( ((s-a)**2) )

def lnprior(alpha):
    """
        Prior probabilty on the hyper parameters alpha, describing an NxM heat map.
        :alpha:
        Flattened 2D numpy array with NxM-1 heat map values
        """
    if np.all(alpha >= 0) and np.sum(alpha <= 1):
        return 0.0
    else:
        return -np.inf

def lnprob(alpha, s, errs):
    """
        Posterior function for alpha and binned samples drawn from each galaxy posterior. 
        :alpha:
        Flattened 2D numpy array with NxM-1 heat map values
        :s:
        Flattened array of samples binned onto NxM grid - Nj samples drawn from each, Ni, galaxy
        """
    lp = lnprior(alpha)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(alpha, s, errs)



