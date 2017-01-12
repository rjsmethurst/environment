import numpy as N
from astropy.table import Table, hstack, vstack, Column
from astropy import units as u

""" In define_cent_sat.py I loaded up the Berlind catalogs I've somehow got the wrong number 
of galaxies in each group. Instead I should take N_group directly from the 
Berlind catalog along with velocity dispersion.

In this program I will link the group and members catalogs from Berlind for each of the 
mr18,19,20 data files and then vstack them into a single file at the end. 

I will THEN cross match this to GZ-GALEX sample after this via TOPCAT."""

c = 299792.458 * u.km/u.s


# mr18g = Table.read('/Users/becky/Projects/environment/berlind/mr18_group_catalog.dat', format='ascii', names=['ID', 'RA', 'Dec', 'z', 'N group', 'Mr_tot', '(g-r)_tot', 'sigma_v', 'R_perp,rms', 'r_edge']) 
# mr18m = Table.read('/Users/becky/Projects/environment/berlind/mr18_member_catalog.dat', format='ascii', names=['groupID', 'RA', 'Dec', 'z', 'M_r', 'g-r', 'fibcol', 'r_edge'])

# mr18g.sort('ID')
# gidx18 = N.searchsorted(mr18g['ID'], mr18m['groupID'])
# gdeets18 = mr18g[gidx18]
# mr18mg = hstack([mr18m, gdeets18])
# mr18mg.write('/Users/becky/Projects/environment/berlind/mr18_members_catalog_with_berlind_group_data.fits', format='fits', overwrite=True)

# mr19g = Table.read('/Users/becky/Projects/environment/berlind/mr19_group_catalog.dat', format='ascii', names=['ID', 'RA', 'Dec', 'z', 'N group', 'Mr_tot', '(g-r)_tot', 'sigma_v', 'R_perp,rms', 'r_edge'])
# mr19m = Table.read('/Users/becky/Projects/environment/berlind/mr19_member_catalog.dat', format='ascii', names=['groupID', 'RA', 'Dec', 'z', 'M_r', 'g-r', 'fibcol', 'r_edge'])
# mr19g.sort('ID')
# gidx19 = N.searchsorted(mr19g['ID'], mr19m['groupID'])
# gdeets19 = mr19g[gidx19]
# mr19mg = hstack([mr19m, gdeets19])
# mr19mg.write('/Users/becky/Projects/environment/berlind/mr19_members_catalog_with_berlind_group_data.fits', format='fits', overwrite=True)


# mr20m = Table.read('/Users/becky/Projects/environment/berlind/mr20_member_catalog.dat', format='ascii', names=['groupID', 'RA', 'Dec', 'z', 'M_r', 'g-r', 'fibcol', 'r_edge'])
# mr20g = Table.read('/Users/becky/Projects/environment/berlind/mr20_group_catalog.dat', format='ascii', names=['ID', 'RA', 'Dec', 'z', 'N group', 'Mr_tot', '(g-r)_tot', 'sigma_v', 'R_perp,rms', 'r_edge'])
# mr20g.sort('ID')
# gidx20 = N.searchsorted(mr20g['ID'], mr20m['groupID'])
# gdeets20 = mr20g[gidx20]
# mr20mg = hstack([mr20m, gdeets20])
# mr20mg.write('/Users/becky/Projects/environment/berlind/mr20_members_catalog_with_berlind_group_data.fits', format='fits', overwrite=True)

# ### Now stack all these tables and save
# mr19mg['groupID'] = mr19mg['groupID'] + N.max(mr18mg['groupID'])
# mr20mg['groupID'] = mr20mg['groupID'] + N.max(mr19mg['groupID'])

# mrallmg = vstack([mr18mg, mr19mg, mr20mg])
# mrallmg.write('/Users/becky/Projects/environment/berlind/mr181920_members_catalog_with_berlind_group_data_central_or_satellite.fits', format='fits', overwrite=True)


# cs = Column(name='central or satellite', dtype=int, length=len(mrallmg))
# del_v_c = Column(name="delta v from central", data=N.zeros(len(mrallmg)))
# del_v_m = Column(name="delta v from group mean", data=N.zeros(len(mrallmg)))


# for n in range(1, N.max(mrallmg['groupID'])):
# 	idx = N.where(mrallmg['groupID'] == n)
# 	if len(idx[0]) == 0:
# 		pass
# 	if len(idx[0]) > 1:
# 		group_data = mrallmg[idx]
# 		cs_int = N.zeros_like(idx[0])
# 		for j in range(len(idx[0])):
# 			if group_data['M_r'][j] == N.min(group_data['M_r']):
# 				cs_int[j] = 1
# 			else:
# 				cs_int[j] = 0
# 		cent = group_data[N.where(cs_int == 1)]
# 		while len(cent) > 1:
# 			gidx = N.where(cs_int ==1)
# 			ridx = N.random.random_integers(0,1,1)
# 			cs_int[gidx[0][ridx][0]] = 0
# 			cent = group_data[N.where(cs_int == 1)]
# 		cs[idx] = cs_int
# 		cent = group_data[N.where(cs_int == 1)]
# 		vi = c * (group_data['z_1'] - N.mean(group_data['z_1']))/(1 + N.mean(group_data['z_1']))
# 		vic = c * (cent['z_1'] - N.mean(group_data['z_1']))/(1 + N.mean(group_data['z_1']))
# 		del_v_c[idx] = vi - vic
# 		del_v_m[idx] = vi - N.mean(vi)
# 	else:
# 		pass

# mrallmg.add_column(cs)
# mrallmg.add_column(del_v_c)
# mrallmg.add_column(del_v_m)

# mrallmg.write('/Users/becky/Projects/environment/berlind/mr181920_members_catalog_with_berlind_group_data_central_or_satellite.fits', format='fits', overwrite=True)

"""Now Ive crossmatched this table to MPA JHU catalog to get the mass of the 
central galaxy for the satellites. Some centrals won't have been matched to MPA JHU 
so those masses are left blank. I also calculated the projected groupc entric radius
as well, along with a dynamic estimate of R200 from the velocity dispersion
calculated by Berlind. """

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 75.0, Om0 = 0.3)

data = Table.read('mr181920_members_catalog_with_berlind_group_data_central_or_satellite_MPA_JHU.fits', format='fits')

centmass = Column(name='stellar mass of central', dtype=float, length=len(data))
R200 = Column(name='virial radius r200', dtype='float', length=len(data))
kpca = cosmo.kpc_comoving_per_arcmin(data['z_1'])
ka = Column(name='kpc per arcmin', data=kpca.value, dtype=float, unit=u.kpc/u.arcmin)
data.add_column(ka)
kd = Column(name='kpc per degree', data = data['kpc per arcmin'].to(u.kpc/u.degree), dtype=float)
data.add_column(kd)
ccr = Column(name='projected cluster centric radius', length=len(data), dtype=float, unit=u.kpc)

i=0
for n in N.unique(data['groupID']):
	i+=1
	print (i/float(len(N.unique(data['groupID']))))*100,'%' 
	idx = N.where(data['groupID'] == n)
	group_data = data[idx]
	cent = group_data[N.where(group_data['central or satellite']==1)]
	centmass[idx] = cent['AVG_MASS']
	R200[idx] = 1.73 * (group_data['sigma_v']/(1000*u.km/u.s)) * (1/(cosmo.Ode(group_data['z_2']) + cosmo.Om0 * (1+group_data['z_2'])**3)**0.5) * (1/(cosmo.H0/100)*(u.km/u.s))
	idxs = N.where(N.logical_and(data['groupID']==n, data['central or satellite'] == 0))
	idxc = N.where(N.logical_and(data['groupID']==n, data['central or satellite'] == 1))
	gc = data[idxc]
	gs = data[idxs]
	if len(gc) > 0:
	    c1 = SkyCoord(ra=gc['RA_1'], dec=gc['Dec_1'], unit=u.degree)
	    c2 = SkyCoord(ra=gs['RA_1'], dec=gs['Dec_1'], unit=u.degree)
	    s = c1.separation(c2)*gs['kpc per degree']
	    ccr[idxs] = s
	    ccr[idxc] = 0.0


data.add_columns([centmass, R200, ccr])
data.write('mr181920_members_catalog_with_berlind_group_data_central_or_satellite_MPA_JHU_cent_mass_r200_rp.fits', format='fits', overwrite=True)