"""Back up plan: produce plots of many panels with grid of R_p with (i) increasing stellar mass, (ii) increasing stellar mass of central and (iii) number in cluster using the original method. Eventually these will be produced using hyper PopStarPy method. For now this file will load in all the samples and data file and split into bins and then collate and save."""

import numpy as np
import pylab as plt
from astropy import Table, Column, vstack
from astropy import units as un
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 75.0, Om0 = 0.3)


