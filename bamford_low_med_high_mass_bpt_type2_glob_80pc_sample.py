import numpy as N
import os
import time
import glob
from astropy.table import Table, vstack

dir1='/usersVol1/smethurst/hyper/samples/'
# l11 = os.listdir(dir1)
dir2 = '/usersVol1/smethurst/hyper/prob/'
# l22 = os.listdir(dir2)
# l22 = l22[2:]
h = Table.read('bpt_identified_type2_agn_seyfert_gz2_match.fits', format='fits')

print len(h)

d = h[h['t01_smooth_or_features_a02_features_or_disk_debiased'] >= 0.8]
s = h[h['t01_smooth_or_features_a01_smooth_debiased'] >=0.8]

h = vstack([d, s])

abs_m_gal = h['MR']
ur_gal = h['MU_MR']
print abs_m_gal
log_m_l = N.zeros(len(ur_gal))
m_msun = N.zeros_like(log_m_l)

for j in range(len(log_m_l)):
    if ur_gal[j] <=2.1:
        log_m_l[j] = -0.95 + 0.56 * ur_gal[j]
    else:
        log_m_l[j] = -0.16 + 0.18 * ur_gal[j]
    m_msun[j] = (((4.62 - abs_m_gal[j])/2.5) + log_m_l[j])

print m_msun

print 'defining p values'
pm = h

alow = pm[N.where(m_msun < 10.25)]
N.save('low_bpt_type2_seyf_hardcut_80pc_agreement_sample.npy', alow)
alows = []
alowp = []
for n in range(len(alow)):
    alows.append('samples_*_'+str(alow['col15'][n])+'_'+str(alow['col16'][n])+'.npy')
    alowp.append('log_probability_*_'+str(alow['col15'][n])+'_'+str(alow['col16'][n])+'.npy')

print 'low mass agn :', len(alow)


amed = pm[N.where(N.logical_and(m_msun > 10.25, m_msun<10.75))]
N.save('med_bpt_type2_seyf_hardcut_80pc_agreement_sample.npy', amed)
ameds = []
amedp = []
for n in range(len(amed)):
    ameds.append('samples_*_'+str(amed['col15'][n])+'_'+str(amed['col16'][n])+'.npy')
    amedp.append('log_probability_*_'+str(amed['col15'][n])+'_'+str(amed['col16'][n])+'.npy')

print 'med mass agn :', len(amed)


ahigh = pm[N.where(m_msun > 10.75)]
N.save('high_bpt_type2_seyf_hardcut_80pc_agreement_sample.npy', ahigh)
ahighs = []
ahighp = []
for n in range(len(ahigh)):
    ahighs.append('samples_*_'+str(ahigh['col15'][n])+'_'+str(ahigh['col16'][n])+'.npy')
    ahighp.append('log_probability_*_'+str(ahigh['col15'][n])+'_'+str(ahigh['col16'][n])+'.npy')

print 'high mass agn :', len(ahigh)

X = N.linspace(0, 14, 100)
Y = N.linspace(0, 4, 100)
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

count = 0

bf = N.zeros((1,6))

for j in range(len(alows)):
    print 'low mass agn: ', (j/float(len(alows)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+alows[j])[0])
        p = N.exp(N.load(glob.glob(dir2+alowp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*alow['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*alow['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_low_mass_bpt_type2_seyf_hardcut_80pc_agreement_sample.npy', bf)
N.save('sum_weight_low_mass_bpt_type2_seyf_hardcut_80pc_agreement_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_low_mass_bpt_type2_seyf_hardcut_80pc_agreement_sample_smooth_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(ameds)):
    print 'med mass agn: ',(j/float(len(ameds)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+ameds[j])[0])
        p = N.exp(N.load(glob.glob(dir2+amedp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*amed['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*amed['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_med_mass_seyf.npy', bf)
N.save('sum_weight_med_mass_bpt_type2_seyf_hardcut_80pc_agreement_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_med_mass_bpt_type2_seyf_hardcut_80pc_agreement_sample_smooth_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(ahighs)):
    print 'high mass agn: ',(j/float(len(ahighs)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+ahighs[j])[0])
        p = N.exp(N.load(glob.glob(dir2+ahighp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
            pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*ahigh['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*ahigh['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_high_mass_seyf.npy', bf)
N.save('sum_weight_high_mass_bpt_type2_seyf_hardcut_80pc_agreement_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_high_mass_bpt_type2_seyf_hardcut_80pc_agreement_sample_smooth_log_weight.npy', sums)

s