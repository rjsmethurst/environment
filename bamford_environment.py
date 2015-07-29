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
h = Table.read('gz2_gz1_extra_galex_matched_data.fits', format='fits')

h = h[h['IVAN_DENSITY'] > -50]

d = h[h['t01_smooth_or_features_a02_features_or_disk_debiased'] >= 0.8]
s = h[h['t01_smooth_or_features_a01_smooth_debiased'] >=0.8]

h = vstack([d, s])

pc = h[N.where(h['IVAN_DENSITY'] > 0.8)]


abs_m_gal = pc['MR']
ur_gal = pc['MU_MR']
log_m_l = N.zeros(len(ur_gal))
m_msun = N.zeros_like(log_m_l)

for j in range(len(log_m_l)):
    if ur_gal[j] <=2.1:
        log_m_l[j] = -0.95 + 0.56 * ur_gal[j]
    else:
        log_m_l[j] = -0.16 + 0.18 * ur_gal[j]
    m_msun[j] = (((4.62 - abs_m_gal[j])/2.5) + log_m_l[j])


clow = pc[N.where(m_msun < 10.25)]
idx = N.random.randint(0, len(clow), 482)
clow = clow[idx]
clow.write('low_mass_gal_cluster_clean.fits', format='fits')
clows = []
clowp = []
for n in range(len(clow)):
    clows.append('samples_*_'+str(clow['col15'][n])+'_'+str(clow['col16'][n])+'.npy')
    clowp.append('log_probability_*_'+str(clow['col15'][n])+'_'+str(clow['col16'][n])+'.npy')

print 'low mass cluster :', len(clow)


cmed = pc[N.where(N.logical_and(m_msun > 10.25, m_msun<10.75))]
idx = N.random.randint(0, len(cmed), 482)
cmed = cmed[idx]
cmed.write('med_mass_gal_cluster_clean.fits', format='fits')
cmeds = []
cmedp = []
for n in range(len(cmed)):
    cmeds.append('samples_*_'+str(cmed['col15'][n])+'_'+str(cmed['col16'][n])+'.npy')
    cmedp.append('log_probability_*_'+str(cmed['col15'][n])+'_'+str(cmed['col16'][n])+'.npy')

print 'med mass cluster :', len(cmed)


chigh = pc[N.where(m_msun > 10.75)]
idx = N.random.randint(0, len(chigh), 482)
chigh = chigh[idx]
chigh.write('high_mass_gal_cluster_clean.fits', format='fits')
chighs = []
chighp = []
for n in range(len(chigh)):
    chighs.append('samples_*_'+str(chigh['col15'][n])+'_'+str(chigh['col16'][n])+'.npy')
    chighp.append('log_probability_*_'+str(chigh['col15'][n])+'_'+str(chigh['col16'][n])+'.npy')

print 'high mass cluster :', len(chigh)

print 'defining p values'
pm = h[N.where(h['IVAN_DENSITY'] < -0.8)]

abs_m_gal = pm['MR']
ur_gal = pm['MU_MR']
log_m_l = N.zeros(len(ur_gal))
m_msun = N.zeros_like(log_m_l)

for j in range(len(log_m_l)):
    if ur_gal[j] <=2.1:
        log_m_l[j] = -0.95 + 0.56 * ur_gal[j]
    else:
        log_m_l[j] = -0.16 + 0.18 * ur_gal[j]
    m_msun[j] = (((4.62 - abs_m_gal[j])/2.5) + log_m_l[j])


alow = pm[N.where(m_msun < 10.25)]
idx = N.random.randint(0, len(alow), 482)
alow = alow[idx]
alow.write('low_mass_gal_field_clean.fits', format='fits')
alows = []
alowp = []
for n in range(len(alow)):
    alows.append('samples_*_'+str(alow['col15'][n])+'_'+str(alow['col16'][n])+'.npy')
    alowp.append('log_probability_*_'+str(alow['col15'][n])+'_'+str(alow['col16'][n])+'.npy')

print 'low mass field :', len(alow)


amed = pm[N.where(N.logical_and(m_msun > 10.25, m_msun<10.75))]
idx = N.random.randint(0, len(amed), 482)
amed = amed[idx]
amed.write('med_mass_gal_field_clean.fits', format='fits')
ameds = []
amedp = []
for n in range(len(amed)):
    ameds.append('samples_*_'+str(amed['col15'][n])+'_'+str(amed['col16'][n])+'.npy')
    amedp.append('log_probability_*_'+str(amed['col15'][n])+'_'+str(amed['col16'][n])+'.npy')

print 'med mass field :', len(amed)


ahigh = pm[N.where(m_msun > 10.75)]
# idx = N.random.randint(0, len(ahigh), len(chigh))
# ahigh = ahigh[idx]
ahigh.write('high_mass_gal_field_clean.fits', format='fits')
ahighs = []
ahighp = []
for n in range(len(ahigh)):
    ahighs.append('samples_*_'+str(ahigh['col15'][n])+'_'+str(ahigh['col16'][n])+'.npy')
    ahighp.append('log_probability_*_'+str(ahigh['col15'][n])+'_'+str(ahigh['col16'][n])+'.npy')

print 'high mass field :', len(ahigh)




X = N.linspace(0, 14, 100)
Y = N.linspace(0, 4, 100)
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

count = 0

bf = N.zeros((1,6))

for j in range(len(alows)):
    print 'low mass field: ', (j/float(len(alows)))*100, '% complete'
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
N.save('best_fit_zlt01_low_mass_field_clean.npy', bf)
N.save('sum_weight_low_mass_field_disc_zlt01_clean_log_weight.npy', sumd)
N.save('sum_weight_low_mass_field_smooth_zlt01_clean_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(ameds)):
    print 'med mass field: ',(j/float(len(ameds)))*100, '% complete'
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
N.save('best_fit_zlt01_med_mass_field_clean.npy', bf)
N.save('sum_weight_med_mass_field_disc_zlt01_clean_log_weight.npy', sumd)
N.save('sum_weight_med_mass_field_smooth_zlt01_clean_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(ahighs)):
    print 'high mass field: ',(j/float(len(ahighs)))*100, '% complete'
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
N.save('best_fit_zlt01_high_mass_field_clean.npy', bf)
N.save('sum_weight_high_mass_field_disc_zlt01_clean_log_weight.npy', sumd)
N.save('sum_weight_high_mass_field_smooth_zlt01_clean_log_weight.npy', sums)




for j in range(len(clows)):
    print 'low mass cluster: ', (j/float(len(clows)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+clows[j])[0])
        p = N.exp(N.load(glob.glob(dir2+clowp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*clow['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*clow['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_zlt01_low_mass_cluster_clean.npy', bf)
N.save('sum_weight_low_mass_cluster_disc_zlt01_clean_log_weight.npy', sumd)
N.save('sum_weight_low_mass_cluster_smooth_zlt01_clean_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(cmeds)):
    print 'med mass cluster: ',(j/float(len(cmeds)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+cmeds[j])[0])
        p = N.exp(N.load(glob.glob(dir2+cmedp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*cmed['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*cmed['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_zlt01_med_mass_cluster_clean.npy', bf)
N.save('sum_weight_med_mass_cluster_disc_zlt01_clean_log_weight.npy', sumd)
N.save('sum_weight_med_mass_cluster_smooth_zlt01_clean_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(chighs)):
    print 'high mass cluster: ',(j/float(len(chighs)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+chighs[j])[0])
        p = N.exp(N.load(glob.glob(dir2+chighp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
            pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*chigh['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*chigh['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_zlt01_high_mass_cluster_clean.npy', bf)
N.save('sum_weight_high_mass_cluster_disc_zlt01_clean_log_weight.npy', sumd)
N.save('sum_weight_high_mass_cluster_smooth_zlt01_clean_log_weight.npy', sums)