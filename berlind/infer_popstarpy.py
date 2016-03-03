import numpy as np
from heatmap import * 
import emcee
import pylab as plt
from astropy.table import Table
from itertools import product
import time
import triangle


print 'loading...'
highs = np.load('all_samples/all_sat_samples_high_mass_glob.npy')

data = Table.read('berlind_data_gz2_extra_order_sfh_samples.fits', format='fits')
ahigh = data[np.where(data['stellar mass'] > 10.75)]

del data 

ndraw = 100
nwalkers = 10
burnin = 200
nsteps = 200
ndim = (50**2) - 1

def histone(s):
	H, X, Y = np.histogram2d(s[:,0], s[:,1], bins=50, range=[[0,14], [0,4]])
	idx = np.random.choice(np.arange(50*50), ndraw, p=(H/np.sum(H)).flatten())
	return s[idx,:]

print 'drawing samples...'
samp = np.empty((len(highs), ndraw, 2))
samp1 = np.empty((len(highs), ndraw, 2))
samp2 = np.empty((len(highs), ndraw, 2))
for i in range(len(highs)):
	samp[i,:,:] = histone(highs[i,:,:])
	samp1[i,:,:] = histone(highs[i,:,:])
	samp2[i,:,:] = histone(highs[i,:,:])


samp3 = np.mean(np.append(samp1.reshape(1,len(highs), ndraw, -1), samp2.reshape(1,len(highs), ndraw, -1), axis=0), axis=0)
samp4 = samp3/np.sum(samp3)
errs = np.abs((samp/np.sum(samp))-samp3)

pd = np.outer(ahigh['t01_smooth_or_features_a02_features_or_disk_debiased'], np.ones(ndraw)).reshape(-1,)
ps = np.outer(ahigh['t01_smooth_or_features_a01_smooth_debiased'], np.ones(ndraw)).reshape(-1,)

print 'binning...'
sd, xs, ys = np.histogram2d(samp[:,:,0].reshape(-1,), samp[:,:,1].reshape(-1,), bins=np.sqrt(ndim+1), range=([0,14],[0,4]), weights=pd)
ss, xs, ys = np.histogram2d(samp[:,:,0].reshape(-1,), samp[:,:,1].reshape(-1,), bins=np.sqrt(ndim+1), range=([0,14],[0,4]), weights=ps)

sd1, xs, ys = np.histogram2d(samp1[:,:,0].reshape(-1,), samp1[:,:,1].reshape(-1,), bins=np.sqrt(ndim+1), range=([0,14],[0,4]), weights=pd)
sd2, xs, ys = np.histogram2d(samp2[:,:,0].reshape(-1,), samp2[:,:,1].reshape(-1,), bins=np.sqrt(ndim+1), range=([0,14],[0,4]), weights=pd)

sd3 = np.mean(np.append(sd1.reshape(1,np.sqrt(ndim+1), -1), sd2.reshape(1, np.sqrt(ndim+1), -1), axis=0), axis=0)
sd4 = sd3/np.sum(sd3)
errs = np.abs((sd/np.sum(sd))-sd4)
errs[errs==0] = np.mean(errs)

#start = (sd/np.sum(sd)).reshape(-1,) - np.random.randn(2500)
#start[-1] = 1 - np.sum(start[:-1])
st = np.random.rand(ndim+1).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))
start = (st/np.sum(st)).flatten()
import scipy.optimize as op
nll = lambda *args: -lnprob(*args)
result = op.minimize(nll, start, args=((sd/np.sum(sd)).reshape(-1,), errs.flatten()))

# print 'starting emcee...'
# p0 = [start +1e-3*np.random.randn(ndim) for i in range(nwalkers)]
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2, args=([(sd/np.sum(sd)).reshape(-1,)]))
# pos, prob, state = sampler.run_mcmc(p0, burnin)
# samples = sampler.chain[:,:,:].reshape((-1,ndim))
# samples_save = 'samples_burnin_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
# np.save(samples_save, samples)
# biprob = sampler.flatlnprobability
# print 'max prob...', np.max(biprob)
# np.save('samples_burnin_lnprob_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', biprob)
# #walker_plot(samples, nwalkers, burnin)
# sampler.reset()
# print 'RESET samples...'
# # main sampler run
# sampler.run_mcmc(pos, nsteps)
# samples = sampler.chain[:,:,:].reshape((-1,ndim))
# samples_save = 'samples_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
# np.save(samples_save, samples)
# prob = sampler.flatlnprobability
# np.save('samples_lnprob_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', prob)
# print 'acceptance fraction', sampler.acceptance_fraction
# emcee_alpha = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))
# print 'emcee found values for high mass disc heat map...', emcee_alpha
# fig = triangle.corner(samples[:,:10], labels=list((np.arange(10).astype(str))))
# fig.savefig('triangle_heatmap_alpha_high_mass_group_disc.pdf')

# alpha = np.percentile(samples, 50, axis=0)
# alphas = np.append(alpha, [1-np.sum(alpha)], axis=0).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))

print result 

alphas= result['x'].reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))
#alphas = np.append(alpha, [1-np.sum(alpha)], axis=0).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))


plt.figure()
plt.imshow(alphas.T, origin='lower', cmap=plt.cm.cubehelix_r, interpolation='nearest', extent=(0,14,0,4), aspect='auto')
plt.savefig('heatmap_high_mass_group_disc.png')
