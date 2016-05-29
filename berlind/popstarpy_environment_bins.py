"""Back up plan: produce plots of many panels with grid of R_p with (i) increasing stellar mass, (ii) increasing stellar mass of central and (iii) number in cluster using the original method. Eventually these will be produced using hyper PopStarPy method. For now this file will load in all the samples and data file and split into bins and then collate and save."""
var = []
alls = [var for var in globals() if var[0] != '_']
for var in alls:
	del globals()[var]

import numpy as N
from scipy import stats
from functools import partial
from astropy.table import Table, Column, vstack
from astropy import units as un
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 75.0, Om0 = 0.3)
from pdb import set_trace as st 
import gc
gc.collect()

data = Table.read('berlind_lim_data_gz2_extra_order_sfh_samples_MPA_JHU.fits', format='fits')
idx = N.where(data['AVG_SFR']< data['peng'])
data = data[idx]
print 'loading samples'
samples = N.empty((16767, 40000, 2))
samples[:,:,:] = N.load('all_samples_glob.npy')
samples = samples[idx]
print 'loading prob'
prob = N.load('all_lnprob_env_exp_lt_0.2_1_reshape_glob.npy')[idx]
# prob[prob < 0.2] = 0

print 'loaded everything'

X = N.linspace(0, 14, 100)
Xs = X[:-1] + N.diff(X)
Y = N.linspace(0, 4, 100)
Ys = Y[:-1] + N.diff(Y)

#r_bins = stats.mstats.mquantiles(N.nan_to_num(data['projected cluster centric radius'].to(un.Mpc)/data['virial radius']).value, [0, 1/6., 2/6., 3/6., 4/6., 5/6., 1], limit=(0.001, 15))
r_bins = N.array([0.02202517, 0.5, 1.0, 2.0, 14.98747191])
mc_bins = stats.mstats.mquantiles(N.nan_to_num(data['stellar mass of group central']), [0, 1/3., 2/3., 1], limit=(9, 12))
#ms_bins = stats.mstats.mquantiles(data['stellar mass'], [0, 0.25, 0.5, 0.75, 1], limit=(7, 13))
#N_bins = stats.mstats.mquantiles(data['number in cluster'], [0, 0.25, 0.5, 0.75, 1], limit=(2, 250))

print 'digitizing'

dr = N.digitize(N.nan_to_num(data['projected cluster centric radius'].to(un.Mpc)/data['virial radius']).value, r_bins)
dm = N.digitize(N.nan_to_num(data['stellar mass of group central']), mc_bins)
#dms = N.digitize(data['stellar mass'], ms_bins)
#dn = N.digitize(data['number in cluster'], N_bins)


def maphist(x, y, p):
	return N.nan_to_num(N.histogram2d(x, y, bins=(X,Y), normed=True, weights=p)[0])

for i in range(1, len(r_bins)):
	for j in range(1, len(mc_bins)):
		print i, j
		sums=N.zeros((99,99))
		sumd=N.zeros((99,99))
		idx = N.logical_and(dr==i,dm==j)
		ss = samples[idx]
		p = prob[idx]
		dd = data[idx]
		#mapfunc = partial(N.histogram2d, bins=(X,Y), normed=True, weights=N.log(p[:,:,0]))
		binned = N.array(map(maphist, ss[:,:,0], ss[:,:,1], p[:,:,0]))
		sumd = N.sum(binned.T*dd['t01_smooth_or_features_a02_features_or_disk_debiased'].data.data, axis=2)
		sums = N.sum(binned.T*dd['t01_smooth_or_features_a01_smooth_debiased'].data.data, axis=2)
		N.save('environment_quenched_less_bins_disc_'+str(r_bins[i-1])+'_'+str(r_bins[i])+'_'+str(mc_bins[j-1])+'_'+str(mc_bins[j])+'.npy', sumd)
		N.save('environment_quenched_less_bins_smooth_'+str(r_bins[i-1])+'_'+str(r_bins[i])+'_'+str(mc_bins[j-1])+'_'+str(mc_bins[j])+'.npy', sums)