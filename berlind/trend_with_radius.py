import numpy as N
import pylab as P
from astropy.table import Table, vstack, Column
from astropy import units as u


# In[3]:

from astropy.cosmology import FlatLambdaCDM
#cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)
cosmo = FlatLambdaCDM(H0 = 75.0, Om0 = 0.3)
c = 299792.458 * u.km/u.s

s = N.load('all_samples/all_samples_glob.npy')
# In[6]:

data = Table.read('berlind_data_gz2_extra_order_sfh_samples.fits', format='fits')


ahigh = data[N.where(data['halo mass'] > 13)]
amed = data[N.where(N.logical_and(data['halo mass'] < 13, data['halo mass']>12))]
alow = data[N.where(data['halo mass'] < 12)]


highs = s[N.where(data['halo mass'] > 13)]
meds = s[N.where(N.logical_and(data['halo mass'] < 13, data['halo mass']>12))]
lows = s[N.where(data['halo mass'] < 12)]

hd = highs[N.where(ahigh['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.8)] 
md = meds[N.where(amed['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.8)]
ld = lows[N.where(alow['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.8)]
hs = highs[N.where(ahigh['t01_smooth_or_features_a01_smooth_debiased'] > 0.8)] 
ms = meds[N.where(amed['t01_smooth_or_features_a01_smooth_debiased'] > 0.8)]
ls = lows[N.where(alow['t01_smooth_or_features_a01_smooth_debiased'] > 0.8)]

ahd = ahigh[N.where(ahigh['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.8)] 
amd = amed[N.where(amed['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.8)]
ald = alow[N.where(alow['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.8)]
ahs = ahigh[N.where(ahigh['t01_smooth_or_features_a01_smooth_debiased'] > 0.8)] 
ams = amed[N.where(amed['t01_smooth_or_features_a01_smooth_debiased'] > 0.8)]
als = alow[N.where(alow['t01_smooth_or_features_a01_smooth_debiased'] > 0.8)]

# In[19]:

pccrhs = N.outer(ahs['projected cluster centric radius'].to(u.Mpc)/ahs['virial radius'], N.ones(40000))
pccrhd = N.outer(ahd['projected cluster centric radius'].to(u.Mpc)/ahd['virial radius'], N.ones(40000))

# In[20]:

pccrms = N.outer(ams['projected cluster centric radius'].to(u.Mpc)/ams['virial radius'], N.ones(40000))
pccrmd = N.outer(amd['projected cluster centric radius'].to(u.Mpc)/amd['virial radius'], N.ones(40000))

# In[21]:

pccrls = N.outer(als['projected cluster centric radius'].to(u.Mpc)/als['virial radius'], N.ones(40000))
pccrld = N.outer(ald['projected cluster centric radius'].to(u.Mpc)/ald['virial radius'], N.ones(40000))


# In[24]:

from scipy.stats import binned_statistic as bs


# In[26]:

logbins = N.append(N.linspace(0, 1, 6),  N.linspace(2, 15, 8), axis=0)


# In[27]:

tshs = bs(pccrhs.flatten(), hs[:,:,0].flatten(), statistic='mean', bins=logbins)
tshd = bs(pccrhd.flatten(), hd[:,:,0].flatten(), statistic='mean', bins=logbins)


# # In[31]:

# #tsh1 = bs(pccrhs.flatten(), highs[:,:,0].flatten(), statistic=lambda y: N.percentile(y, 16), bins=logbins)
# #tsh2 = bs(pccrhs.flatten(), highs[:,:,0].flatten(), statistic=lambda y: N.percentile(y, 84), bins=logbins)


# # In[32]:

tsms = bs(pccrms.flatten(), ms[:,:,0].flatten(), statistic='mean', bins=logbins)
tsmd = bs(pccrmd.flatten(), md[:,:,0].flatten(), statistic='mean', bins=logbins)


# # In[ ]:

# #tsm1 = bs(pccrms.flatten(), meds[:,:,0].flatten(), statistic=lambda y: N.percentile(y, 16), bins=logbins)
# #tsm2 = bs(pccrms.flatten(), meds[:,:,0].flatten(), statistic=lambda y: N.percentile(y, 84), bins=logbins)


# # In[33]:

tsls = bs(pccrls.flatten(), ls[:,:,0].flatten(), statistic='mean', bins=logbins)
tsld = bs(pccrld.flatten(), ld[:,:,0].flatten(), statistic='mean', bins=logbins)


# # In[ ]:

# #tsl1 = bs(pccrls.flatten(), lows[:,:,0].flatten(), statistic=lambda y: N.percentile(y, 16), bins=logbins)
# #tsl2 = bs(pccrls.flatten(), lows[:,:,0].flatten(), statistic=lambda y: N.percentile(y, 84), bins=logbins)

import os

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'


P.rc('figure', facecolor='none', edgecolor='none', autolayout=True)
P.rc('path', simplify=True)
P.rc('text', usetex=True)
P.rc('font', family='serif')
P.rc('axes', labelsize='large', facecolor='none', linewidth=0.7, color_cycle = ['k', '#900000', 'g', 'b', 'c', 'm', 'y'])
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('lines', markersize=4, linewidth=1, markeredgewidth=0.2)
P.rc('legend', numpoints=1, frameon=False, handletextpad=0.3, scatterpoints=1, handlelength=2, handleheight=0.1)
P.rc('savefig', facecolor='none', edgecolor='none', frameon='False')

params =   {'font.size' : 11,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.size': 1,
            'ytick.minor.size': 1,
            }
P.rcParams.update(params) 

# # In[44]:

P.figure(figsize=(5,12))
ax = P.subplot(311)
ax2 = P.subplot(312)
ax3 = P.subplot(313)
ax.scatter(tshs[1][:-1] + N.diff(tshs[1])[0]/2, tshs[0], marker='x', color='r')
ax.scatter(tshd[1][:-1] + N.diff(tshd[1])[0]/2, tshd[0], marker='x', color='b')
ax.set_xscale('log')
#ax12 = ax.twinx()
#ax12.scatter(taush[1][:-1] + N.diff(taush[1])[0]/2, taush[0], marker='o', color='k')
#ax.fill_between(tsh[1][:-1] + N.diff(tsh[1])[0]/2, tsh1[0], tsh2[0], color='k', alpha=0.15, edgecolor='None')
ax2.scatter(tsms[1][:-1] + N.diff(tsms[1])[0]/2, tsms[0], marker='x', color='r')
ax2.scatter(tsmd[1][:-1] + N.diff(tsmd[1])[0]/2, tsmd[0], marker='x', color='b')

ax2.set_xscale('log')
#ax22 = ax2.twinx()
#ax22.scatter(tausm[1][:-1] + N.diff(tausm[1])[0]/2, tausm[0], marker='o', color='k')
#ax2.fill_between(tsm[1][:-1] + N.diff(tsm[1])[0]/2, tsm1[0], tsm2[0], color='k', alpha=0.15, edgecolor='None')
ax3.scatter(tsls[1][:-1] + N.diff(tsls[1])[0]/2, tsls[0], marker='x', color='r')
ax3.scatter(tsld[1][:-1] + N.diff(tsld[1])[0]/2, tsld[0], marker='x', color='b')

ax3.set_xscale('log')
#ax32 = ax3.twinx()
#ax32.scatter(tausl[1][:-1] + N.diff(tausl[1])[0]/2, tausl[0], marker='o', color='k')
#ax3.fill_between(tsl[1][:-1] + N.diff(tsl[1])[0]/2, tsl1[0], tsl2[0], color='k', alpha=0.15, edgecolor='None')
ax.minorticks_on()
ax.set_xlim(0.09, 15)
ax.text(0.1, 0.9, r'$\log[M_h/M_{\odot}] > 13$', transform=ax.transAxes)
#ax.set_xlabel(r'$R/R_{200}$')
ax.set_ylabel(r'$\overline{t_q}$')
ax2.set_ylabel(r'$\overline{t_q}$')
ax3.set_ylabel(r'$\overline{t_q}$')
#ax12.set_ylabel(r'$\overline{\tau}$')
#ax22.set_ylabel(r'$\overline{\tau}$')
#ax32.set_ylabel(r'$\overline{\tau}$')
ax.set_xticks([0.1, 1, 10])
ax.set_xticklabels([0.1, 1, 10])
ax2.minorticks_on()
ax2.set_xlim(0.09, 15)
ax2.text(0.1, 0.9, r'$12 < \log[M_h/M_{\odot}] < 13$', transform=ax2.transAxes)
#ax2.set_xlabel(r'$R/R_{200}$')
ax2.set_xticks([0.1, 1, 10])
ax2.set_xticklabels([0.1, 1, 10])
ax3.minorticks_on()
ax3.set_xlim(0.09, 15)
ax3.set_xlabel(r'$R/R_{200}$')
ax3.set_xticks([0.1, 1, 10])
ax3.set_xticklabels([0.1, 1, 10])
ax3.text(0.1, 0.9, r'$\log[M_h/M_{\odot}] < 12$', transform=ax3.transAxes)

#P.subplots_adjust(hspace=0.0)
P.savefig('t_q_tau_trend_with_radius_split_halo_mass_disc_smooth.png')



# In[5]:

taushs = bs(pccrhs.flatten(), hs[:,:,1].flatten(), statistic='median', bins=logbins)
taushd = bs(pccrhd.flatten(), hd[:,:,1].flatten(), statistic='median', bins=logbins)

tausms = bs(pccrms.flatten(), ms[:,:,1].flatten(), statistic='median', bins=logbins)
tausmd = bs(pccrmd.flatten(), md[:,:,1].flatten(), statistic='median', bins=logbins)

tausls = bs(pccrls.flatten(), ls[:,:,1].flatten(), statistic='median', bins=logbins)
tausld = bs(pccrld.flatten(), ld[:,:,1].flatten(), statistic='median', bins=logbins)


# In[ ]:

#tausl1 = bs(pccrls.flatten(), lows[:,:,1].flatten(), statistic=lambda y: N.percentile(y, 16), bins=logbins)
#tausl2 = bs(pccrls.flatten(), lows[:,:,1].flatten(), statistic=lambda y: N.percentile(y, 84), bins=logbins)


# In[ ]:

P.figure(figsize=(4.5,12))
ax = P.subplot(311)
ax2 = P.subplot(312)
ax3 = P.subplot(313)
ax.scatter(taushs[1][:-1] + N.diff(taushs[1])[0]/2, taushs[0], marker='x', color='r')
ax.scatter(taushd[1][:-1] + N.diff(taushd[1])[0]/2, taushd[0], marker='x', color='b')

#ax.fill_between(taush[1][:-1] + N.diff(taush[1])[0]/2, taush1[0], tsh2[0], color='k', alpha=0.15, edgecolor='None')
ax2.scatter(tausms[1][:-1] + N.diff(tausms[1])[0]/2, tausms[0], marker='x', color='r')
ax2.scatter(tausmd[1][:-1] + N.diff(tausmd[1])[0]/2, tausmd[0], marker='x', color='b')

#ax2.fill_between(tausm[1][:-1] + N.diff(tausm[1])[0]/2, tausm1[0], tausm2[0], color='k', alpha=0.15, edgecolor='None')
ax3.scatter(tausls[1][:-1] + N.diff(tausls[1])[0]/2, tausls[0], marker='x', color='r')
ax3.scatter(tausld[1][:-1] + N.diff(tausld[1])[0]/2, tausld[0], marker='x', color='b')

#ax3.fill_between(tausl[1][:-1] + N.diff(tausl[1])[0]/2, tausl1[0], tausl2[0], color='k', alpha=0.15, edgecolor='None')
ax.set_xscale('log')
ax.minorticks_on()
ax.set_xlim(0.09, 15)
#ax.tick_params('x', labelbottom='off')
#ax.set_xlabel(r'$R/R_{200}$')
ax2.set_ylabel(r'$\tau$')
ax.set_xticks([0.1, 1, 10])
ax.set_xticklabels([0.1, 1, 10])
ax2.set_xscale('log')
ax2.minorticks_on()
ax2.set_xlim(0.09, 15)
#ax2.tick_params('x', labelbottom='off')
#ax2.set_xlabel(r'$R/R_{200}$')
ax2.set_xticks([0.1, 1, 10])
ax2.set_xticklabels([0.1, 1, 10])
ax3.set_xscale('log')
ax3.minorticks_on()
ax3.set_xlim(0.09, 15)
ax3.set_xlabel(r'$R/R_{200}$')
ax3.set_xticks([0.1, 1, 10])
ax3.set_xticklabels([0.1, 1, 10])
P.subplots_adjust(hspace=0.0)
P.savefig('tau_trend_with_radius_split_halo_mass_disc_smooth.png')


# In[ ]:



