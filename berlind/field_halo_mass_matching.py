import numpy as N
import pylab as P
from astropy.table import Table, Column, vstack 
import time

datas =  Table.read('mr181920_member_long_groupID_catalog_central_satellites_with_halo_mass_projected_clustercentric_radius_vel_disp_virial_radius_h_alpha_flux.fits', format='fits')
field = Table.read('field_candidate_sample_ivan_lt_-0.8_rv_gtr_25_gz2_gz1_extra_h_alpha_sfr_halo_stellar_masses.fits', format='fits')

#load central galaxies 
data = datas[N.where(datas['central or satellite']==1)]

#set up table to contain resampled field galaxies 
field_resample = Table()

for j in range(len(data)):
	print j
	bins = N.array([N.log10(0.95*(10**data['halo mass'][j])), N.log10(1.05*(10**data['halo mass'][j]))])
	idx = N.digitize(field['halo mass'], bins)
	field_match = field[idx ==1]
	if len(field_match) == 0:
		bins = N.array([N.log10(0.9*(10**data['halo mass'][j])), N.log10(1.1*(10**data['halo mass'][j]))])
		idx = N.digitize(field['halo mass'], bins)
		field_match = field[idx ==1]
	else:
		pass
	if len(field_match) > 5:
		print 'more than 5 matched'
		index = N.random.randint(0, len(field_match), 5)
		field_resample = vstack([field_resample, field_match[index]])
		ind1 = N.where(field['col15'] == field_match[index[0]]['col15'])
		ind2 = N.where(field['col15'] == field_match[index[1]]['col15'])
		ind3 = N.where(field['col15'] == field_match[index[2]]['col15'])
		ind4 = N.where(field['col15'] == field_match[index[3]]['col15'])
		ind5 = N.where(field['col15'] == field_match[index[4]]['col15'])
		field.remove_rows([ind1[0][0], ind2[0][0], ind3[0][0], ind4[0][0], ind5[0][0]])
	elif len(field_match) > 0:
		print 'less than 5 matched'
		index1 = N.random.randint(0, len(field_match))
		ind1 = N.where(field['col15'] == field_match[index1]['col15'])
		field_resample = vstack([field_resample, field_match[index1]])
		field.remove_rows([ind1[0][0]])
	else: 
		print 'no matches found for group index: ', j
print 'number of matched field galaxies: ', len(field_resample)

field_resample.write('field_cand_sample_matched_cent_groups_halo_mass_pm_5pc_sfr_h_alpha_masses_balrdy_env_lt_-0.8_rv_gtr_25.fits', format='fits')

