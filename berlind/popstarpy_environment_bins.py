"""Back up plan: produce plots of many panels with grid of R_p with (i) increasing stellar mass, (ii) increasing stellar mass of central and (iii) number in cluster using the original method. Eventually these will be produced using hyper PopStarPy method. For now this file will load in all the samples and data file and split into bins and then collate and save."""

import numpy as np
import pylab as plt
from scipy import stats
from functools import partial
from astropy import Table, Column, vstack
Pfrom astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 75.0, Om0 = 0.3)

data = Table.read('berlind_data_gz2_extra_order_sfh_samples.fits', format='fits')
samples = N.load('all_samples/all_samples_glob.npy')

alows = []
alowp = []
lowpd = N.zeros((1,1))
lowps = N.zeros((1,1))
print 'probs...'
for n in range(len(alow)):
    alows.append(dir1+'samples_*_'+str(alow['col15'][n])+'_'+str(alow['col16'][n])+'.npy')
    alowp.append(dir2+'log_probability_*_'+str(alow['col15'][n])+'_'+str(alow['col16'][n])+'.npy')
    #lowpd = N.append(lowpd, alow['t01_smooth_or_features_a02_features_or_disk_debiased'][n]*N.ones((40000,1)), axis=0)
    #lowps = N.append(lowps, alow['t01_smooth_or_features_a01_smooth_debiased'][n]*N.ones((40000,1)), axis=0)
print 'globbing...'
alowg = map(glob.glob, alows)
alowgp = map(glob.glob, alowp)
print 'low mass agn :', len(alowg)

X = N.linspace(0, 14, 100)
Xs = X[:-1] + N.diff(X)
Y = N.linspace(0, 4, 100)
Ys = Y[:-1] + N.diff(Y)

count = 0

r_bins = stats.mstats.mquantiles(N.nan_to_num(sat['projected cluster centric radius'].to(un.Mpc)/sat['virial radius']).value, [0, 1/6., 2/6., 3/6., 4/6., 5/6., 1], limit=(0.001, 15))
mc_bins = stats.mstats.mquantiles(sat['stellar mass of group central'], [0, 0.25, 0.5, 0.75, 1], limit=(9, 12))
ms_bins = stats.mstats.mquantiles(sat['stellar mass'], [0, 0.25, 0.5, 0.75, 1], limit=(7, 13))
N_bins = stats.mstats.mquantiles(sat['number in cluster'], [0, 0.25, 0.5, 0.75, 1], limit=(2, 250))

for i in range(len(r_bins)-1):
	for j in range(len(mc_bins)-1):
	    sums=N.zeros((99,99))
	    sumd=N.zeros((99,99))
	    s = # samples where bins
	    p = # prob where bins and where p[exp(p) < 0.2]=0
	    dd = # data where bins
	    mapfunc = partial(N.histogram2d, bins=(X,Y), normed=True, weights=N.log(p))
	    binned = N.array(map(mapfunc, s[:,:,0], s[:,:,1]))[:,0]
	    sumd = N.sum(binned*dd['t01_smooth_or_features_a02_features_or_disk_debiased'].astype(float))
	    sums = N.sum(binned*dd['t01_smooth_or_features_a01_smooth_debiased'].astype(float))
	    N.save('environment_bin_disc_'+str(r_bins[i])+'_'+str(r_bins[i+1])+'_'+str(mc_bins[j])+'_'+str(mc_bins[j+1])+'.npy', sumd)
	    N.save('environment_bin_smooth_'+str(r_bins[i])+'_'+str(r_bins[i+1])+'_'+str(mc_bins[j])+'_'+str(mc_bins[j+1])+'.npy', sums)