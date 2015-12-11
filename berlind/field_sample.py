import numpy as np 
import pylab as P 
from astropy.table import Table, vstack, Column
from astropy import units as u 
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)

groups = Table.read('mr181920_member_long_groupID_catalog_central_satellites_with_halo_mass_projected_clustercentric_radius_vel_disp_virial_radius_h_alpha_flux.fits', format='fits')
groups = groups[N.where(groups['number in cluster']>=2)]
cent = groups[N.where(groups['central or satellite']==1)]

gz2 = Table.read('/Users/becky/Projects/Green-Valley-Project/data/GZ2_all_GALEX_match_GZ1_k_correct_green_valley.fits', format='fits')
data = gz2[np.invert(np.in1d(gz2['ra_1'].data.data, groups['ra_1'].data.data))]
zrange = 0.01 

datasky = SkyCoord(ra=data['ra_1'], dec=data['dec_1'], distance=cosmo.luminosity_distance(data['z_1']), unit=(u.degree,u.degree,u.Mpc))
centcatalog = SkyCoord(ra=cent['ra_1'], dec=cent['dec_1'],  distance=cosmo.luminosity_distance(cent['z_1']), unit=(u.degree,u.degree, u.Mpc))

idx, d2d, d3d = datasky.match_to_catalog_3d(centcatalog)

centmatch = cent[idx]
centmatchsky = SkyCoord(ra=centmatch['ra_1'], dec=centmatch['dec_1'],  distance=cosmo.luminosity_distance(centmatch['z_1']), unit=(u.degree,u.degree, u.Mpc))

kpcd = (cosmo.kpc_comoving_per_arcmin(data['z_1'])).to(u.kpc/u.degree)
pccr = centmatchsky.separation(datasky)*kpcd

radius = pccr/centmatch['virial radius']

P.figure()
P.hist(radius, histtype='step', color='k')
