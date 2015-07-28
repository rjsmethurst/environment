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

d = h[h['t01_smooth_or_features_a02_features_or_disk_debiased'] >= 0.8]
s = h[h['t01_smooth_or_features_a01_smooth_debiased'] >=0.8]

h = vstack([d, s])

pc = h[N.where(h['IVAN_DENSITY'] > 0.8)]

C_dash = 2.06 - 0.244*N.tanh((pc['MR'] + 20.07)/1.09)
lower = C_dash - 0.128
upper = C_dash + 0.128

cblue= pc[N.where(pc['MU_MR'] < lower)]
idx = N.random.randint(0, len(cblue), 500)
cblue = cblue[idx]
N.save('blue_gal_cluster_80pc_sample.npy', cblue)
cblues = []
cbluep = []
for n in range(len(cblue)):
    cblues.append('samples_*_'+str(cblue['col15'][n])+'_'+str(cblue['col16'][n])+'.npy')
    cbluep.append('log_probability_*_'+str(cblue['col15'][n])+'_'+str(cblue['col16'][n])+'.npy')

print 'blue cluster :', len(cblue)


cgreen = pc[N.where(N.logical_and(pc['MU_MR'] < upper, pc['MU_MR'] > lower))]
N.save('green_gal_cluster_80pc_sample.npy', cgreen)
cgreens = []
cgreenp = []
for n in range(len(cgreen)):
    cgreens.append('samples_*_'+str(cgreen['col15'][n])+'_'+str(cgreen['col16'][n])+'.npy')
    cgreenp.append('log_probability_*_'+str(cgreen['col15'][n])+'_'+str(cgreen['col16'][n])+'.npy')

print 'green cluster :', len(cgreen)


cred = pc[N.where(pc['MU_MR'] > upper)]
idx = N.random.randint(0, len(cred), 500)
cred = cred[idx]
N.save('red_gal_field_80pc_sample.npy', cred)
creds = []
credp = []
for n in range(len(cred)):
    creds.append('samples_*_'+str(cred['col15'][n])+'_'+str(cred['col16'][n])+'.npy')
    credp.append('log_probability_*_'+str(cred['col15'][n])+'_'+str(cred['col16'][n])+'.npy')

print 'red cluster :', len(cred)

print 'defining p values'
pm = h[N.where(h['IVAN_DENSITY'] < -0.8)]

C_dash = 2.06 - 0.244*N.tanh((pm['MR'] + 20.07)/1.09)
lower = C_dash - 0.128
upper = C_dash + 0.128

fblue = pm[N.where(pm['MU_MR'] < lower)]
idx = N.random.randint(0, len(fblue), 500)
fblue = fblue[idx]
N.save('blue_gal_field_80pc_sample.npy', fblue)
fblues = []
fbluep = []
for n in range(len(fblue)):
    fblues.append('samples_*_'+str(fblue['col15'][n])+'_'+str(fblue['col16'][n])+'.npy')
    fbluep.append('log_probability_*_'+str(fblue['col15'][n])+'_'+str(fblue['col16'][n])+'.npy')

print 'blue field :', len(fblue)


fgreen = pm[N.where(N.logical_and(pm['MU_MR'] < upper, pm['MU_MR'] > lower))]
idx = N.random.randint(0, len(fgreen), 500)
fgreen = fgreen[idx]
N.save('green_gal_field_80pc_sample.npy', fgreen)
fgreens = []
fgreenp = []
for n in range(len(fgreen)):
    fgreens.append('samples_*_'+str(fgreen['col15'][n])+'_'+str(fgreen['col16'][n])+'.npy')
    fgreenp.append('log_probability_*_'+str(fgreen['col15'][n])+'_'+str(fgreen['col16'][n])+'.npy')

print 'green field :', len(fgreen)


fred = pm[N.where(pm['MU_MR'] > upper)]
idx = N.random.randint(0, len(fred), 500)
fred = fred[idx]
N.save('red_gal_field_80pc_sample.npy', fred)
freds = []
fredp = []
for n in range(len(fred)):
    freds.append('samples_*_'+str(fred['col15'][n])+'_'+str(fred['col16'][n])+'.npy')
    fredp.append('log_probability_*_'+str(fred['col15'][n])+'_'+str(fred['col16'][n])+'.npy')

print 'red field :', len(fred)




X = N.linspace(0, 14, 100)
Y = N.linspace(0, 4, 100)
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

count = 0

bf = N.zeros((1,6))

for j in range(len(fblues)):
    print 'blue field: ', (j/float(len(fblues)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+fblues[j])[0])
        p = N.exp(N.load(glob.glob(dir2+fbluep[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*fblue['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*fblue['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_blue_field_80pc_sample.npy', bf)
N.save('sum_weight_blue_field_80pc_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_blue_field_80pc_sample_smooth_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(fgreens)):
    print 'green field: ',(j/float(len(fgreens)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+fgreens[j])[0])
        p = N.exp(N.load(glob.glob(dir2+fgreenp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*fgreen['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*fgreen['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_green_field_80pc_sample.npy', bf)
N.save('sum_weight_green_field_80pc_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_green_field_80pc_sample_smooth_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(freds)):
    print 'red field: ',(j/float(len(freds)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+freds[j])[0])
        p = N.exp(N.load(glob.glob(dir2+fredp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
            pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*fred['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*fred['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_red_field_80pc_sample.npy', bf)
N.save('sum_weight_red_field_80pc_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_red_field_80pc_sample_smooth_log_weight.npy', sums)




for j in range(len(cblues)):
    print 'blue cluster: ', (j/float(len(cblues)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+cblues[j])[0])
        p = N.exp(N.load(glob.glob(dir2+cbluep[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*cblue['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*cblue['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_blue_cluster_80pc_sample.npy', bf)
N.save('sum_weight_blue_cluster_80pc_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_blue_cluster_80pc_sample_smooth_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(cgreens)):
    print 'green cluster: ',(j/float(len(cgreens)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+cgreens[j])[0])
        p = N.exp(N.load(glob.glob(dir2+cgreenp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
           pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*cgreen['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*cgreen['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_green_cluster_80pc_sample.npy', bf)
N.save('sum_weight_green_cluster_80pc_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_green_cluster_80pc_sample_smooth_log_weight.npy', sums)

bf = N.zeros((1,6))
sumd=N.zeros((len(X)-1, len(Y)-1))
sums=N.zeros((len(X)-1, len(Y)-1))

for j in range(len(creds)):
    print 'red cluster: ',(j/float(len(creds)))*100, '% complete'
    try:
        s = N.load(glob.glob(dir1+creds[j])[0])
        p = N.exp(N.load(glob.glob(dir2+credp[j])[0]))
        s = s[N.where(p>0.2)]
        p = p[N.where(p>0.2)]
        if len(s) == 0:
            pass
        else:
            bf = N.append(bf, N.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(s, [16,50,84],axis=0)))).reshape(1,6), axis=0)
            Hs, Xs, Ys = N.histogram2d(s[:,0], s[:,1], bins=(X, Y), normed=True, weights=N.log(p))
            sumd += (Hs*cred['t01_smooth_or_features_a02_features_or_disk_debiased'][j].astype(float))
            sums += (Hs*cred['t01_smooth_or_features_a01_smooth_debiased'][j].astype(float))
    except IndexError:
        count+=1
        print count
        pass
N.save('best_fit_red_cluster_80pc_sample.npy', bf)
N.save('sum_weight_red_cluster_80pc_sample_disc_log_weight.npy', sumd)
N.save('sum_weight_red_cluster_80pc_sample_smooth_log_weight.npy', sums)