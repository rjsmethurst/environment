import numpy as np 
import pylab as P 
from astropy.tables import Table, vstack, Column
from astropy import units as u 
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)

