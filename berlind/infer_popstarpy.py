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
nwalkers = 200
burnin = 2500
nsteps = 1000
ndim = (5**2) - 1

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

#st = np.random.rand(ndim)
#start = (st/np.sum(st))
start = np.ones(ndim)/(ndim+1)

print 'starting emcee...'
p0 = [start +1e-3*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2, args=([Ns]))
pos, prob, state = sampler.run_mcmc(p0, burnin)
samples = sampler.chain[:,:,:].reshape((-1,ndim))
samples_save = 'samples_burnin_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
np.save(samples_save, samples)
biprob = sampler.flatlnprobability
print 'max prob...', np.max(biprob)
if np.max(biprob) == -np.inf:
	os.system('exit')
np.save('samples_burnin_lnprob_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', biprob)
#walker_plot(samples, nwalkers, burnin)
sampler.reset()
print 'RESET samples...'
# main sampler run
sampler.run_mcmc(pos, nsteps)
samples = sampler.chain[:,:,:].reshape((-1,ndim))
samples_save = 'samples_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy'
np.save(samples_save, samples)
prob = sampler.flatlnprobability
np.save('samples_lnprob_heatmap_high_mass_group_disc_'+str(len(samples))+'_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.npy', prob)
print 'acceptance fraction', sampler.acceptance_fraction
emcee_alpha = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))

pi = np.percentile(samples, 50, axis=0)
pis = np.append(pi, [1-np.sum(pi)], axis=0).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))

print 'pi found values for high mass disc heat map...', pis


plt.figure()
plt.imshow(pis.T, origin='lower', cmap=plt.cm.binary, interpolation='nearest', extent=(0,14,0,4), aspect='auto')
plt.savefig('heatmap_high_mass_group_disc.png')

fig = triangle.corner(samples[:,:10])
fig.savefig('triangle_heatmap_alpha_high_mass_group_disc.pdf')

