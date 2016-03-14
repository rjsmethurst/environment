import numpy as np
from heatmap import * 
import emcee
import pylab as plt
from astropy.table import Table
from itertools import product
import time
import triangle
import os


print 'loading...'
highs = np.load('first_5000_samples_green.npy')
prob = np.load('first_5000_lnprob_green.npy')
probs = np.append(prob.reshape(prob.shape[0], prob.shape[1], 1), prob.reshape(prob.shape[0], prob.shape[1], 1), axis=2)

data = Table.read('galaxy_data_extra.fits', format='fits')
green = data[np.where(np.logical_and(data['MU_MR'] < data['upper_GV'], data['MU_MR'] > data['lower_GV']))]
del data 

gsk =[1967, 2328, 2498, 4748, 6439, 8202, 11121, 11592, 12053, 13680, 14337, 16607, 17055, 18062, 18697, 18803, 18808, 20272, 20289, 20338, 20426, 20429, 20468, 20488, 21075, 21351]

green = np.delete(green, gsk)

greens = green[:5000]

ndraw = 2000
nwalkers = 500
burnin = 2000
nsteps = 500
ndim = (10**2) - 1

def histone(s, ps):
	#H, X, Y = np.histogram2d(s[:,0], s[:,1], bins=50, range=[[0,14], [0,4]], weights=ps)
	#idx = np.random.choice(np.arange(50*50), ndraw, p=(H/np.sum(H)).flatten())
	idx = np.random.choice(np.arange(s.shape[0]), ndraw, p =ps)
	return s[idx,:]

def vector(s):
	return np.histogram2d(s[:,0], s[:,1], bins=np.sqrt(ndim+1), range=([0,14],[0,4]))[0].flatten()


print 'drawing samples...'
samp = np.empty((len(highs), ndraw, 2))
empty = []
for i in range(len(highs)):
	s = highs[i,:,:][np.where(np.exp(probs[i,:,:])>0.2)].reshape(-1,2)
	p = probs[i,:,:][np.where(np.exp(probs[i,:,:])>0.2)].reshape(-1,2)
	if len(s)>0:
		samp[i,:,:] = histone(s, np.exp(p[:,0]))
	else:
		empty.append(i)

print 'empty rows... ', empty
print samp.shape
samp.delete(empty)
print samp.shape

print 'binning samples...'
Ns = np.array([vector(slice) for slice in samp])

Ns[np.where(Ns==0)]=0.1

pd = greens['t01_smooth_or_features_a02_features_or_disk_debiased']
pd[np.where(pd==0)] = 1E-5
ps = greens['t01_smooth_or_features_a01_smooth_debiased']
ps[np.where(ps==0)] = 1E-5

if os.path.exists('green_valley_pi_ps.npy') == True:
	start = np.load('green_valley_pi_ps.npy')
	if len(start) == ndim:
		pass
	elif ndim+1 % len(start) == 0:
		start = np.repeat(np.repeat(start, ndim+1/float(len(start)), axis=0), ndim+1/float(len(start)), axis=1)
	else:
		start = np.ones(ndim)/(ndim+1)
else:
	start = np.ones(ndim)/(ndim+1)


del samp
del highs
del green
del greens

#start = np.ones(ndim)/(ndim+1)

print 'optimizing...'

import scipy.optimize as op
nll = lambda *args: -lnprob(*args)
result = op.minimize(nll, start, args=(Ns, ps, ndraw), method = 'Nelder-Mead')

while result['message'] != 'Optimization terminated successfully.':
	start = result['x']
	result = op.minimize(nll, start, args=(Ns, ps, ndraw), method = 'Nelder-Mead')

pi = result['x']
np.save('green_valley_pi_ps_20x20.npy', pi)


# print 'starting emcee...'
# p0 = [start +1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2, args=([Ns]))
# pos, prob, state = sampler.run_mcmc(p0, burnin)
# samples = sampler.chain[:,:,:].reshape((-1,ndim))
# samples_save = 'samples_burnin_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
# np.save(samples_save, samples)
# biprob = sampler.flatlnprobability
# print 'max prob...', np.max(biprob)
# if np.max(biprob) == -np.inf:
# 	os.system('exit')
# np.save('samples_burnin_lnprob_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', biprob)
# #walker_plot(samples, nwalkers, burnin)
# sampler.reset()
# print 'RESET samples...'
# # main sampler run
# sampler.run_mcmc(pos, nsteps)
# samples = sampler.chain[:,:,:].reshape((-1,ndim))
# samples_save = 'samples_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
# np.save(samples_save, samples)
# prob = sampler.flatlnprobability
# np.save('samples_lnprob_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', prob)
# print 'acceptance fraction', sampler.acceptance_fraction
# emcee_alpha = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))

#pi = np.percentile(samples, 50, axis=0)
pis = np.append(pi, [1-np.sum(pi)], axis=0).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))

print 'pi found values for green heat map...', pis


plt.figure()
plt.imshow(pis.T, origin='lower', cmap=plt.cm.binary, interpolation='nearest', extent=(0,14,0,4), aspect='auto')
plt.savefig('heatmap_green_ps_20x20.png')

# fig = triangle.corner(samples[:,:10])
# fig.savefig('triangle_heatmap_alpha_green.pdf')

