""" Inference code for heat map pixel inference of population posterior distributionp.  """

import numpy as np


def lnlike(pi, N, pv, ndraw):
        """ Likelihood function for hyper parameters pi.
            :pi:
            [N, M] pixel values flattened into vector [N*M]
            :N:
            Drawn samples from each k galaxy binned onto NxM grid which has been flattened to a vector- [Ngal, N*M]
            RETURNS:
            One value of lnlike for given pis having summed over all galaxies, Ngal.
            """
        # np.dot(N, pi) gives vector of shape (Ngal,) we take the log and then sum over all galaxies, Ngal
        # could we also times by GZ vote fraction p - shape (Ngal,) - here before the log? 
        pis = np.append(pi, [1-np.sum(pi)], axis=0)
        return np.sum(np.log(pv*(np.dot(N, pis))/float(ndraw)))

def lnprior(pi):
    """
        Prior probabilty on the hyper parameters pi, describing an NxM heat map.
        :pi:
        [N, M] pixel values flattened into vector [N*M]
        """
    if np.all(pi >= 0) and np.sum(pi) < 1:
        #return np.sum(pi*np.log(pi))
        return 0.0
    else:
        return -np.inf

def lnprob(pi, N, pv, ndraw):
    """
        Posterior function for pixels pi and binned samples drawn from each galaxy posterior. 
        :pi:
        [N, M] pixel values flattened into vector [N*M]
        :N:
        Drawn samples from each k galaxy binned onto NxM grid which has been flattened to a vector- [Ngal, N*M]
        RETURNS:
        One value of lnprob for given pis having summed over all galaxies k
        """
    lp = lnprior(pi)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pi, N, pv, ndraw)



