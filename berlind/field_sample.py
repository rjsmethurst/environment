import numpy as np 
import pylab as P 
from astropy.tables import Table, vstack, Column
from astropy import units as u 
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 71.0, Om0 = 0.26)

groups = Table.read('mr181920_member_long_groupID_catalog_central_satellites_with_halo_mass_projected_clustercentric_radius_vel_disp_virial_radius_h_alpha_flux.fits', format='fits')
cent = groups[N.where(groups['central or satellite']==1)]

gz2 = Table.read('/Users/becky/Projects/Green-Valley-Project/data/GZ2_all_GALEX_match_GZ1_k_correct_green_valley.fits', format='fits')
data = gz2[np.invert(np.in1d(gz2['ra_1'].data.data, groups['ra_1'].data.data))]
zrange = 0.01 

c = SkyCoord(ra=data['ra_1'], dec=data['dec_1'], distance=cosmo.luminosity_distance(data['z_1']), unit=(u.degree,u.degree,u.Mpc))
catalog = SkyCoord(ra=cent['ra_1'], dec=cent['dec_1'],  distance=cosmo.luminosity_distance(cent['z_1']), unit=(u.degree,u.degree, u.Mpc))

idx, d2d, d3d = c.match_to_catalog_3d(catalog)

