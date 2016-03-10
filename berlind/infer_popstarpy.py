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

data = Table.read('galaxy_data_extra.fits', format='fits')
green = data[np.where(np.logical_and(data['MU_MR'] < data['upper_GV'], data['MU_MR'] > data['lower_GV']))]
del data 

gsk =[1967, 2328, 2498, 4748, 6439, 8202, 11121, 11592, 12053, 13680, 14337, 16607, 17055, 18062, 18697, 18803, 18808, 20272, 20289, 20338, 20426, 20429, 20468, 20488, 21075, 21351]

green = np.delete(green, gsk)

greens = green[:5000]

ndraw = 1000
nwalkers = 500
burnin = 2000
nsteps = 500
ndim = (10**2) - 1

def histone(s):
	H, X, Y = np.histogram2d(s[:,0], s[:,1], bins=50, range=[[0,14], [0,4]])
	idx = np.random.choice(np.arange(50*50), ndraw, p=(H/np.sum(H)).flatten())
	return s[idx,:]

def vector(s):
	return np.histogram2d(s[:,0], s[:,1], bins=np.sqrt(ndim+1), range=([0,14],[0,4]))[0].flatten()

print 'drawing samples...'
samp = np.empty((len(highs), ndraw, 2))
for i in range(len(highs)):
	samp[i,:,:] = histone(highs[i,:,:])

print 'binning samples...'
Ns = np.array([vector(slice) for slice in samp])

Ns[np.where(Ns==0)]=0.1

#st = np.random.rand(ndim)
#start = (st/np.sum(st))

# if os.path.exists('last_pis.npy') == True:
# 	start = np.load('last_pis.npy')
# 	if len(start) == ndim:
# 		pass
# 	else:
# 		start = np.ones(ndim)/(ndim+1)
# else:
# 	start = np.ones(ndim)/(ndim+1)

start = np.ones(ndim)/(ndim+1)

import scipy.optimize as op
nll = lambda *args: -lnprob(*args)
result = op.minimize(nll, start, args=(Ns), method = 'Nelder-Mead')

start = result['x']
N.save('start_pis_scipy.npy', start)

print 'starting emcee...'
p0 = [start +1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2, args=([Ns]))
pos, prob, state = sampler.run_mcmc(p0, burnin)
samples = sampler.chain[:,:,:].reshape((-1,ndim))
samples_save = 'samples_burnin_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
np.save(samples_save, samples)
biprob = sampler.flatlnprobability
print 'max prob...', np.max(biprob)
if np.max(biprob) == -np.inf:
	os.system('exit')
np.save('samples_burnin_lnprob_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', biprob)
#walker_plot(samples, nwalkers, burnin)
sampler.reset()
print 'RESET samples...'
# main sampler run
sampler.run_mcmc(pos, nsteps)
samples = sampler.chain[:,:,:].reshape((-1,ndim))
samples_save = 'samples_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
np.save(samples_save, samples)
prob = sampler.flatlnprobability
np.save('samples_lnprob_heatmap_green_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', prob)
print 'acceptance fraction', sampler.acceptance_fraction
emcee_alpha = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))

pi = np.percentile(samples, 50, axis=0)
pis = np.append(pi, [1-np.sum(pi)], axis=0).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))

print 'pi found values for green heat map...', pis


plt.figure()
plt.imshow(pis.T, origin='lower', cmap=plt.cm.binary, interpolation='nearest', extent=(0,14,0,4), aspect='auto')
plt.savefig('heatmap_green.png')

fig = triangle.corner(samples[:,:10])
fig.savefig('triangle_heatmap_alpha_green.pdf')

