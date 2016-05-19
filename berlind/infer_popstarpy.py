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
highs = np.load('first_5000_samples_red.npy')
prob = np.load('first_5000_lnprob_red.npy')
probs = np.append(prob.reshape(prob.shape[0], prob.shape[1], 1), prob.reshape(prob.shape[0], prob.shape[1], 1), axis=2)

data = Table.read('galaxy_data_extra.fits', format='fits')
green = data[np.where(np.logical_and(data['MU_MR'] < data['upper_GV'], data['MU_MR'] > data['lower_GV']))]
del data 

greens = green[:5000]

gsk=[482, 1488, 1515, 1740, 1894, 1998, 2090, 2115, 2703, 2805, 2904, 2990, 4441, 4593, 4701]

greens = np.delete(greens, gsk)


ndraw = 2000
nwalkers = 500
burnin = 2000
nsteps = 500
ndim = (10**2) - 1

def histone(s, ps):
	#H, X, Y = np.histogram2d(s[:,0], s[:,1], bins=50, range=[[0,14], [0,4]], weights=ps)
	#idx = np.random.choice(np.arange(50*50), ndraw, p=(H/np.sum(H)).flatten())
	idx = np.random.choice(np.arange(s.shape[0]), ndraw, p =ps/np.sum(ps))
	return s[idx,:], idx

def vector(s):
	return np.histogram2d(s[:,0], s[:,1], bins=np.sqrt(ndim+1), range=([0,14],[0,4]), normed=True, weights=s[:,2])[0].flatten()


print 'drawing samples...'
samp = np.empty((len(highs), ndraw, 2))
idx = np.empty((len(highs), ndraw))
empty = []
for i in range(len(highs)):
	s = highs[i,:,:][np.where((probs[i,:,:])>0.2)].reshape(-1,2)
	p = probs[i,:,0][np.where((probs[i,:,0])>0.2)].reshape(-1,)
	if len(s)>0:
		pn = (p - np.min(p))/(np.max(p)-np.min(p))
		samp[i,:,:], idx[i,:] = histone(s, pn)
		#samp[i,:,:] = histone(s, np.ones(len(s)))
	else:
		empty.append(i)

print 'empty rows... ', empty
print samp.shape
samps = np.delete(samp,empty, axis=0)
idxes = np.delete(idx, empty, axis=0)
a = np.arange(len(samps)).reshape(-1,1)
idxs = np.repeat(a, ndraw, axis=1)
ps = probs[idxs, idxes.astype(int), :]
samp = np.append(samps, ps[:,:,:], axis=2)
print samp.shape
greens = np.delete(greens, empty, axis=0)
print 'binning samples...'
Ns = np.array([vector(slice) for slice in samp])
np.save('binned_first_5000_red_seq_drawn_2000_samples.npy', Ns)

Ns[np.where(Ns<=0)]=np.nan
mins = 0.01*np.nanmin(Ns)
Ns = np.nan_to_num(Ns)
Ns[np.where(Ns==0)]= 0.01*mins

pd = greens['t01_smooth_or_features_a02_features_or_disk_debiased']
pd[np.where(pd==0)] = 1E-5
ps = greens['t01_smooth_or_features_a01_smooth_debiased']
ps[np.where(ps==0)] = 1E-5

print len(pd)
print Ns.shape

if os.path.exists('red_sequence_pi_pd_'+str((np.sqrt(ndim+1)))+'x'+str((np.sqrt(ndim+1)))+'_normed_log_weight.npy') == True:
	start = np.load('red_sequence_pi_pd_'+str((np.sqrt(ndim+1)))+'x'+str((np.sqrt(ndim+1)))+'_normed_log_weight.npy')
	if len(start) == ndim:
		print 'loaded previous solution'
		pass
	elif (ndim+1) % (len(start)+1) == 0:
		starts = np.append(start, 1-np.sum(start)).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))
		start = np.repeat(np.repeat(starts, (ndim+1)/float(len(start)+1), axis=0), (ndim+1)/float(len(start)+1), axis=1).reshape(-1,)[:-1]
		print 'loaded previous solution and interpolated onto new ndim grid'
	else:
		start = np.ones(ndim)/(ndim+1)
		print 'starting with equal values across the grid'
else:
	start = np.ones(ndim)/(ndim+1)
	print 'starting with equal values across the grid'

#start = np.ones(ndim)/(ndim+1)

print 'optimizing disc...'

import scipy.optimize as op
nll = lambda *args: -lnprob(*args)
result = op.minimize(nll, start, args=(Ns, pd, ndraw), method = 'Nelder-Mead')

while result['message'] != 'Optimization terminated successfully.':
	start = result['x']
	result = op.minimize(nll, start, args=(Ns, pd, ndraw), method = 'Nelder-Mead')

pid = result['x']
np.save('red_sequence_pi_pd_'+str(np.sqrt(ndim+1))+'x'+str(np.sqrt(ndim+1))+'_normed_log_weight.npy', pid)

#pid = np.load('red_sequence_pi_pd_15.0x15.0_normed_log_weight.npy')
pids = np.append(pid, [1-np.sum(pid)], axis=0).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))

print 'pi found values for green heat map...', pids

Xs = np.linspace(0, 14, np.sqrt(ndim+1)+1)
Ys = np.linspace(0, 4, np.sqrt(ndim+1)+1)

pidsm = pids/np.min(pids)
pidsm[pidsm<=1]=0

pidsy = np.log(pidsm)
pidsy[pidsy < 0] = 0
# xs = 1./np.abs(np.sum(np.log(pids), axis=1))
# ys = 1./np.abs(np.sum(np.log(pids), axis=0))
xsdn = np.sum((pidsy), axis=1)
ysdn = np.sum((pidsy), axis=0)

# mx = np.max(xsdn)/np.min(xsdn)
# minx = (1+mx)*np.max(xsdn)
# my = np.max(ysdn)/np.min(ysdn)
# miny = (1+my)*np.max(ysdn)

# xsd =  (xsdn-minx)/(np.max(xsdn)-minx)
# ysd =  (ysdn-miny)/(np.max(ysdn)-miny)
xsd =  (xsdn-np.min(xsdn))/(np.max(xsdn)-np.min(xsdn))
ysd =  (ysdn-np.min(ysdn))/(np.max(ysdn)-np.min(ysdn))

ysl = 100 * np.sum(ysd[Ys[:-1]+np.diff(Ys)[0] <= 1.0])/np.sum(ysd)
ysm = 100 * np.sum(ysd[np.where(np.logical_and(Ys[:-1]+np.diff(Ys)[0] < 2.0, Ys[:-1]+np.diff(Ys)[0] > 1.0))])/np.sum(ysd)
ysh = 100 * np.sum(ysd[Ys[:-1]+np.diff(Ys)[0] >= 2.0])/np.sum(ysd)

fig = plt.figure(figsize=(6.25,6.25))
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
ax2.set_xlabel(r'$t_q$')
ax2.set_ylabel(r'$\tau$')
ax2.imshow(np.log(pids.T), origin='lower', cmap=plt.cm.binary, interpolation='nearest', extent=(0,14,0,4), aspect='auto', vmin=-7)
ax2.tick_params(axis='x', labeltop='off')
ax1 = plt.subplot2grid((3,3), (0,0),colspan=2)
ax1.hist(Xs[:-1], weights=xsd, bins=Xs, histtype='step', color='k')
ax1.set_xlim(0, 14)
ax1.set_ylim(-0.1, 1.1)
ax1.tick_params(axis='x', labelbottom='off', labeltop='off')
ax1.tick_params(axis='y', labelleft='off')
ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax3.tick_params(axis='x', labelbottom='off')
ax3.tick_params(axis='y', labelleft='off')
ax3.hist(Ys[:-1], weights=ysd, bins=Ys, histtype='step', color='k', orientation='horizontal')
ax3.set_xlim(-0.1, 1.1)
ax3.text(0.2, 0.125, str("%.1f"%ysl)+'%', color='k', transform=ax3.transAxes)
ax3.text(0.2, 0.375, str("%.1f"%ysm)+'%', color='k', transform=ax3.transAxes)
ax3.text(0.2, 0.75, str("%.1f"%ysh)+'%', color='k', transform=ax3.transAxes)
ax3.axhline(y = 1.0, xmin=0, xmax=1, color='0.3', linestyle='dashed')
ax3.axhline(y = 2.0, xmin=0, xmax=1, color='0.3', linestyle='dashed')
#fig.tight_layout()
plt.subplots_adjust(wspace=0.0)
plt.subplots_adjust(hspace=0.0)
plt.savefig('heatmap_red_pd_'+str(np.sqrt(ndim+1))+'x'+str(np.sqrt(ndim+1))+'_normed_log_weight_flat.png')


print 'optimizing smooth...'

if os.path.exists('red_sequence_pi_ps_'+str((np.sqrt(ndim+1)))+'x'+str((np.sqrt(ndim+1)))+'_normed_log_weight.npy') == True:
	start = np.load('red_sequence_pi_ps_'+str((np.sqrt(ndim+1)))+'x'+str((np.sqrt(ndim+1)))+'_normed_log_weight.npy')
	if len(start) == ndim:
		print 'loaded previous solution'
		pass
	elif (ndim+1) % (len(start)+1) == 0:
		starts = np.append(start, 1-np.sum(start)).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))
		start = np.repeat(np.repeat(starts, (ndim+1)/float(len(start)+1), axis=0), (ndim+1)/float(len(start)+1), axis=1).reshape(-1,)[:-1]
		print 'loaded previous solution and interpolated onto new ndim grid'
	else:
		start = np.ones(ndim)/(ndim+1)
		print 'starting with equal values across the grid'
else:
	start = np.ones(ndim)/(ndim+1)
	print 'starting with equal values across the grid'

import scipy.optimize as op
nll = lambda *args: -lnprob(*args)
result = op.minimize(nll, start, args=(Ns, ps, ndraw), method = 'Nelder-Mead')

while result['message'] != 'Optimization terminated successfully.':
	start = result['x']
	result = op.minimize(nll, start, args=(Ns, ps, ndraw), method = 'Nelder-Mead')

pis = result['x']
np.save('red_sequence_pi_ps_'+str(np.sqrt(ndim+1))+'x'+str(np.sqrt(ndim+1))+'_normed_log_weight.npy', pis)

piss = np.append(pis, [1-np.sum(pis)], axis=0).reshape(np.sqrt(ndim+1), np.sqrt(ndim+1))

print 'pi found values for green heat map...', piss

Xss = np.linspace(0, 14, np.sqrt(ndim+1)+1)
Yss = np.linspace(0, 4, np.sqrt(ndim+1)+1)

pissm = piss/np.min(piss)
pissm[pissm<=1]=0

#xss = 1./np.abs(np.sum(np.log(piss), axis=1))
#yss = 1./np.abs(np.sum(np.log(piss), axis=0))
pissy = np.nan_to_num(np.log(pissm))
pissy[pissy<0]=0

xssn = np.sum((pissy), axis=1)
yssn = np.sum((pissy), axis=0)


# mx = np.max(xssn)/np.min(xssn)
# minx = (1+mx)*np.max(xssn)
# my = np.max(yssn)/np.min(yssn)
# miny = (1+my)*np.max(yssn)

# xss =  (xssn-minx)/(np.max(xssn)-minx)
# yss =  (yssn-miny)/(np.max(yssn)-miny)
xss = (xssn-np.min(xssn))/(np.max(xssn)-np.min(xssn))
yss = (yssn-np.min(yssn))/(np.max(yssn)-np.min(yssn))

yssl = 100 * np.sum(yss[Yss[:-1]+np.diff(Yss)[0] <= 1.0])/np.sum(yss)
yssm = 100 * np.sum(yss[np.where(np.logical_and(Yss[:-1]+np.diff(Yss)[0] < 2.0, Yss[:-1]+np.diff(Yss)[0] > 1.0))])/np.sum(yss)
yssh = 100 * np.sum(yss[Yss[:-1]+np.diff(Yss)[0] >= 2.0])/np.sum(yss)

fig = plt.figure(figsize=(6.25,6.25))
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
ax2.set_xlabel(r'$t_q$')
ax2.set_ylabel(r'$\tau$')
ax2.imshow(np.log(piss.T), origin='lower', cmap=plt.cm.binary, interpolation='nearest', extent=(0,14,0,4), aspect='auto', vmin=-7)
ax2.tick_params(axis='x', labeltop='off')
ax1 = plt.subplot2grid((3,3), (0,0),colspan=2)
ax1.hist(Xss[:-1], weights=xss, bins=Xss, histtype='step', color='k')
ax1.set_xlim(0, 14)
ax1.set_ylim(-0.1, 1.1)
ax1.tick_params(axis='x', labelbottom='off', labeltop='off')
ax1.tick_params(axis='y', labelleft='off')
ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax3.tick_params(axis='x', labelbottom='off')
ax3.tick_params(axis='y', labelleft='off')
ax3.hist(Yss[:-1], weights=yss, bins=Yss, histtype='step', color='k', orientation='horizontal')
ax3.set_xlim(-0.1, 1.1)
ax3.text(0.2, 0.125, str("%.1f"%yssl)+'%', color='k', transform=ax3.transAxes)
ax3.text(0.2, 0.375, str("%.1f"%yssm)+'%', color='k', transform=ax3.transAxes)
ax3.text(0.2, 0.75, str("%.1f"%yssh)+'%', color='k', transform=ax3.transAxes)
ax3.axhline(y = 1.0, xmin=0, xmax=1, color='0.3', linestyle='dashed')
ax3.axhline(y = 2.0, xmin=0, xmax=1, color='0.3', linestyle='dashed')
#fig.tight_layout()
plt.subplots_adjust(wspace=0.0)
plt.subplots_adjust(hspace=0.0)
plt.savefig('heatmap_red_ps_'+str(np.sqrt(ndim+1))+'x'+str(np.sqrt(ndim+1))+'_normed_log_weight_flat.png')


