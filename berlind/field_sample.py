import numpy as np 
import pylab as P 
from astropy.table import Table, vstack, Column
from astropy import units as u 
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)

#load in group data and central data
groups = Table.read('mr181920_member_long_groupID_catalog_central_satellites_with_halo_mass_projected_clustercentric_radius_vel_disp_virial_radius_h_alpha_flux.fits', format='fits')
groupc = groups[np.where(np.logical_and(groups['number in cluster']>=2, groups['virial radius']>0))]
cent = groupc[np.where(groupc['central or satellite']==1)]

#load in gz2-galex catalog
gz2 = Table.read('/Users/becky/Projects/environment/gz2_gz1_extra_galex_matched_data.fits', format='fits')
#remove from the gz2-galex catlog those galaxies already in the group catalog
data = gz2[np.invert(np.in1d(gz2['ra_1'].data, groups['ra_1'].data.data))]

#set up astropy 3d sky coordinate objects
datasky = SkyCoord(ra=data['ra_1'], dec=data['dec_1'], distance=cosmo.luminosity_distance(data['z_1']), unit=(u.degree,u.degree,u.Mpc))
centcatalog = SkyCoord(ra=cent['ra_1'], dec=cent['dec_1'],  distance=cosmo.luminosity_distance(cent['z_1']), unit=(u.degree,u.degree, u.Mpc))

#calculate index of central that is closest in 3d distance to the galaxies in the gz2galex data catalog
idx, d2d, d3d = datasky.match_to_catalog_3d(centcatalog)

# set up a new table the same size as the gz2-galex catalog with the central galaxy details that match the same row
# of the gz2-galex table
mci = Column(name='matched central index', data=idx)
centmatch = cent[idx]
centmatchsky = SkyCoord(ra=centmatch['ra_1'], dec=centmatch['dec_1'],  distance=cosmo.luminosity_distance(centmatch['z_1']), unit=(u.degree,u.degree, u.Mpc))

#calculate the separation betweent the gz2-galex data and the central it is closest to
kpcd = (cosmo.kpc_comoving_per_arcmin(data['z_1'])).to(u.kpc/u.degree)
pccr = centmatchsky.separation(datasky)*kpcd
radius = Column(name = 'projected cluster centric radius', data=pccr.to(u.Mpc)/centmatch['virial radius'])

data.add_columns([mci, radius])

P.figure()
ax = P.subplot(111)
ax.hist(radius, histtype='step', color='k', range=(0, 1000), bins=100)
ax.set_xscale('log')
ax.set_xlabel(r'$R/R_{200}$')
ax.set_ylabel(r'$N$')
ax.minorticks_on()
P.savefig('field_radius_distribution.png')

fieldidx = radius > 25

fieldcand = data[fieldidx]

P.figure()
ax = P.subplot(111)
ax.hist(fieldcand['IVAN_DENSITY'], histtype='step', color='k', range=(-3, 3), bins=100)
ax.set_xlabel(r'$\Sigma$')
ax.set_ylabel(r'$N$')
P.savefig('ivan_density_distribution_of_field_candidates.png')

fieldcand.write('field_candidate_sample_rv_gtr_25_gz2_gz1_extra.fits', format='fits')

fieldcandivan = fieldcand[fieldcand['IVAN_DENSITY'] < -0.8]

fieldcandivan.write('field_candidate_sample_ivan_lt_-0.8_rv_gtr_25_gz2_gz1_extra.fits', format='fits')

f = open('field_ivan_rv_no_ivan_ra_dec.txt', 'a')
f.write('objid ra dec \n')
for n in range(len(fieldcandivan)):
    f.write(str(fieldcand['dr7objid'][n])+' '+str(fieldcand['ra_1'][n])+' '+str(fieldcand['dec_1'][n])+'\n')
f.close()


