#!/usr/bin/env python

## Map Reconstruction for Prediction with the MVR algorithm

#  Instructions and Notes
#  ------------
#  Translated from Octave from the script:
#  /ptfa/prog/unsorted/Vis/oct_m/mapReconstr.m
#
#  Other algorithms may be checked from there too.

# Loading packages

import sys
import csv

import numpy as np
import pandas as pd

from statistics import mean

def handle_zero_division(a, b):
    return 0 if b == 0 else a / b

# Read program arguments
filename_load = sys.argv[1]
dir_pred_subtest = sys.argv[2]
DNNtask = sys.argv[3]
MineralArr = sys.argv[4].split(',')
ElementArr = sys.argv[5].split(',')
MetricsArr = sys.argv[6].split(',')

totMin = len(MineralArr)
totElem = len(ElementArr)
totMetrics = len(MetricsArr)

statsFilename = './tests/' + dir_pred_subtest + '/AI_testSet_stats_MVR.info'

df = pd.read_csv(filename_load, encoding = 'utf-8').fillna(0)

# Define predicted elements column names
Pred_Surf_ElementArr = []
for i in ElementArr:
	Pred_Surf_ElementArr.append(str('Pred_Surf_' + i))

# Remove O_2 and Si in case of elements-modified task
if DNNtask == "elements-modified" and "O_2" in ElementArr:
	Pred_Surf_ElementArr.remove("Pred_Surf_O_2")
	ElementArr.remove("O_2")
if DNNtask == "elements-modified" and "Si" in ElementArr:
	Pred_Surf_ElementArr.remove("Pred_Surf_Si")
	ElementArr.remove("Si")

# Collect useful parts of the data in separate dataframes
MetricsData = df[MetricsArr].to_numpy()
UsefulData = df[['Latitude', 'Longitude', 'Altitude', 'TAA']].to_numpy()

# Similarity split by Day/Night
Sim_DayNight = df[['Sun_Inc_Angle']].to_numpy()
# Similarity split by H+/No H+
Sim_Hplus = df[['SP_Activation']].to_numpy()

# Set filenames
filename_elem_rat_Map = {}
elem_rat_Maps = {}
filename_dif_Map = {}
dif_Maps = {}
filename_relResidual_Map = {}
relResidual_Maps = {}
filename_true_elem_rat_Map = {}
true_elem_rat_Maps = {}
Pred_Surf_Elem = {}
True_Surf_Elem = {}
for i in ElementArr:
	filename_elem_rat_Map[i] = './tests/' + dir_pred_subtest + '/Map_elem_' + i + '_rat_MVR.csv'
	elem_rat_Maps[i] = np.zeros((18,36))
	filename_dif_Map[i] = './tests/' + dir_pred_subtest + '/Map_dif_' + i + '_MVR.csv'
	dif_Maps[i] = np.zeros((18,36))
	filename_relResidual_Map[i] = './tests/' + dir_pred_subtest + '/Map_relResidual_' + i + '_MVR.csv'
	relResidual_Maps[i] = np.zeros((18,36))
	filename_true_elem_rat_Map[i] = './tests/' + dir_pred_subtest + '/Map_true_elem_' + i + '_rat_MVR.csv'
	true_elem_rat_Maps[i] = np.zeros((18,36))

	Pred_Surf_Elem[i] = df[str('Pred_Surf_' + i)].to_numpy()
	True_Surf_Elem[i] = df[str('Surf_' + i)].to_numpy()

filename_passCounts = './tests/' + dir_pred_subtest + '/Map_PassCounts_MVR.csv'
filename_Alti = './tests/' + dir_pred_subtest + '/Map_Altitude_MVR.csv'
filename_DayNight = './tests/' + dir_pred_subtest + '/Map_DayNight_MVR.csv'
filename_Hplus = './tests/' + dir_pred_subtest + '/Map_Hplus_MVR.csv'
passCounts = np.zeros((18,36))
AltiMap = np.zeros((18,36))
DayNightMap = np.zeros((18,36))
HplusMap = np.zeros((18,36))

filenames_Similarities = {}
Similarities = {}
for k in MetricsArr:
	filenames_Similarities[k] = './tests/' + dir_pred_subtest + '/Map_Similarity_' + k + '_MVR.csv'
	Similarities[k] = np.zeros((18,36))


print(' ... Reconstruction of the elemental map ratios... ')

for j in range(0, UsefulData.shape[0]):
	# set the map matrix cell indices according to the calculated latitude and longitude
	Lat = UsefulData[j,0]; l_tile = int(np.ceil(Lat/10) + 9)
	Lon = UsefulData[j,1]; m_tile = int(np.ceil(Lon/10))
	l_tile = l_tile - 1
	m_tile = m_tile - 1
	if m_tile == -1:
		m_tile = 0
	if l_tile == -1: 
		l_tile = 0

	passCounts[l_tile,m_tile] = passCounts[l_tile,m_tile] + 1
	AltiMap[l_tile,m_tile] = AltiMap[l_tile,m_tile] + UsefulData[j,2]

	for k in range(0, len(MetricsArr)):
		Similarities[MetricsArr[k]][l_tile,m_tile] = Similarities[MetricsArr[k]][l_tile,m_tile] + MetricsData[j,k]

	if Sim_DayNight[j, 0] == 0:
		DayNightMap[l_tile,m_tile] = DayNightMap[l_tile,m_tile] + 0
	elif Sim_DayNight[j, 0] > 0:
		DayNightMap[l_tile,m_tile] = DayNightMap[l_tile,m_tile] + 1

	if Sim_Hplus[j, 0] == 0:
		HplusMap[l_tile,m_tile] = HplusMap[l_tile,m_tile] + 0
	elif Sim_Hplus[j, 0] > 0:
		HplusMap[l_tile,m_tile] = HplusMap[l_tile,m_tile] + 1

	for i in ElementArr:
		elem_rat_Maps[i][l_tile, m_tile] = elem_rat_Maps[i][l_tile, m_tile] + Pred_Surf_Elem[i][j]
		dif_Maps[i][l_tile, m_tile] = dif_Maps[i][l_tile, m_tile] + np.absolute(Pred_Surf_Elem[i][j] - True_Surf_Elem[i][j])
		if True_Surf_Elem[i][j] == 0:
			new_relative_residual = handle_zero_division(np.absolute(Pred_Surf_Elem[i][j] - True_Surf_Elem[i][j]), np.mean(True_Surf_Elem[i][:]))
		else:
			new_relative_residual = min(handle_zero_division(np.absolute(Pred_Surf_Elem[i][j] - True_Surf_Elem[i][j]), True_Surf_Elem[i][j]),
										handle_zero_division(np.absolute(Pred_Surf_Elem[i][j] - True_Surf_Elem[i][j]), np.mean(True_Surf_Elem[i][:])))
		relResidual_Maps[i][l_tile, m_tile] = relResidual_Maps[i][l_tile, m_tile] + new_relative_residual
		true_elem_rat_Maps[i][l_tile, m_tile] = true_elem_rat_Maps[i][l_tile, m_tile] + True_Surf_Elem[i][j]

# Calculate mean absolute and relative residuals for each mineral
meanAbsResiduals = {}
meanRelResiduals = {}
for i in ElementArr:
	meanAbsResiduals[i] = np.mean(dif_Maps[i])
	meanRelResiduals[i] = np.mean(relResidual_Maps[i])

print('Done! \n')

# Calculate similarity for different altitudes
Sim425s = {}; c425 = 0
Sim850s = {}; c850 = 0
Sim1700s = {}; c1700 = 0
for k in MetricsArr:
	Sim425s[k] = 0;
	Sim850s[k] = 0;
	Sim1700s[k] = 0;

for j in range(0, UsefulData.shape[0]):
	if UsefulData[j, 2] < 850:
		for k in range(0, len(MetricsArr)):
			Sim425s[MetricsArr[k]] = Sim425s[MetricsArr[k]] + MetricsData[j, k]
		c425 = c425 + 1
	elif UsefulData[j, 2] > 1275:
		for k in range(0, len(MetricsArr)):
			Sim1700s[MetricsArr[k]] = Sim1700s[MetricsArr[k]] + MetricsData[j, k]
		c1700 = c1700 + 1
	else:
		for k in range(0, len(MetricsArr)):
			Sim850s[MetricsArr[k]] = Sim850s[MetricsArr[k]] + MetricsData[j, k]
		c850 = c850 + 1

for k in MetricsArr:
	print('   . Similarity {} for altitude 425-850km = {:4.2f}'.format(k, handle_zero_division(Sim425s[k],c425)*100))
	print('   . Similarity {} for altitude 850-1275km = {:4.2f}'.format(k, handle_zero_division(Sim850s[k],c850)*100))
	print('   . Similarity {} for altitude 1275-1700km = {:4.2f}'.format(k, handle_zero_division(Sim1700s[k],c1700)*100))

# Absolute and relative residuals with altitude
absResiduals425s = {}
absResiduals850s = {}
absResiduals1700s = {}
relResiduals425s = {}
relResiduals850s = {}
relResiduals1700s = {}
for i in ElementArr:
	absResiduals425s[i] = 0
	absResiduals850s[i] = 0
	absResiduals1700s[i] = 0
	relResiduals425s[i] = 0
	relResiduals850s[i] = 0
	relResiduals1700s[i] = 0
	c425fromMap = 0; c850fromMap = 0; c1700fromMap = 0;
	for l in range(0, AltiMap.shape[0]):
		for m in range(0, AltiMap.shape[1]):
			AltiThisTile = AltiMap[l, m]
			if AltiThisTile < 850:
				absResiduals425s[i] = absResiduals425s[i] + dif_Maps[i][l, m]
				relResiduals425s[i] = relResiduals425s[i] + relResidual_Maps[i][l, m]
				c425fromMap = c425fromMap + 1;
			elif AltiThisTile > 1275:
				absResiduals1700s[i] = absResiduals1700s[i] + dif_Maps[i][l, m]
				relResiduals1700s[i] = relResiduals1700s[i] + relResidual_Maps[i][l, m]
				c1700fromMap = c1700fromMap + 1;
			else:
				absResiduals850s[i] = absResiduals850s[i] + dif_Maps[i][l, m]
				relResiduals850s[i] = relResiduals850s[i] + relResidual_Maps[i][l, m]
				c850fromMap = c850fromMap + 1; 

print("\n")

# Calculate the similarity for dayside/nightside and presence/absence of impinging H+ ions (protons)
# Similarity split by Day/Night
SimDays = {}; cDay = 0;
SimNights = {}; cNight = 0;
for k in MetricsArr:
	SimDays[k] = 0;
	SimNights[k] = 0;

for j in range(0, Sim_DayNight.shape[0]):
	if Sim_DayNight[j, 0] == 0:
		for k in range(0, len(MetricsArr)):
			SimNights[MetricsArr[k]] = SimNights[MetricsArr[k]] + MetricsData[j, k]
		cNight = cNight + 1
	elif Sim_DayNight[j, 0] > 0:
		for k in range(0, len(MetricsArr)):
			SimDays[MetricsArr[k]] = SimDays[MetricsArr[k]] + MetricsData[j, k]
		cDay = cDay + 1

for k in MetricsArr:
	print('   . Similarity {} for Dayside = {:4.2f}'.format(k, handle_zero_division(SimDays[k],cDay)*100))
	print('   . Similarity {} for Nightside = {:4.2f}'.format(k, handle_zero_division(SimNights[k],cNight)*100))

print("\n")

# Similarity split by H+/No H+
SimHpluses = {}; cHplus = 0;
SimNoHpluses = {}; cNoHplus = 0;
for k in MetricsArr:
	SimHpluses[k] = 0;
	SimNoHpluses[k] = 0;

for j in range(0, Sim_Hplus.shape[0]):
	if Sim_Hplus[j, 0] == 0:
		for k in range(0, len(MetricsArr)):
			SimNoHpluses[MetricsArr[k]] = SimNoHpluses[MetricsArr[k]] + MetricsData[j, k]
		cNoHplus = cNoHplus + 1
	elif Sim_Hplus[j, 0] > 0:
		for k in range(0, len(MetricsArr)):
			SimHpluses[MetricsArr[k]] = SimHpluses[MetricsArr[k]] + MetricsData[j, k]
		cHplus = cHplus + 1
	
for k in MetricsArr:
	print('   . Similarity {} with H+ = {:4.2f}'.format(k, handle_zero_division(SimHpluses[k],cHplus)*100))
	print('   . Similarity {} without H+ = {:4.2f}'.format(k, handle_zero_division(SimNoHpluses[k],cNoHplus)*100))

# Absolute and relative residuals by day/night or H+/NoH+
absResidualsDays = {}
absResidualsNights = {}
relResidualsDays = {}
relResidualsNights = {}

for i in ElementArr:
	absResidualsDays[i] = 0
	absResidualsNights[i] = 0
	relResidualsDays[i] = 0
	relResidualsNights[i] = 0
	cDayfromMap = 0; cNightfromMap = 0;
	for l in range(0, AltiMap.shape[0]):
		for m in range(0, AltiMap.shape[1]):
			DayNightThisTile = DayNightMap[l, m]
			if DayNightThisTile == 0:
				absResidualsNights[i] = absResidualsNights[i] + dif_Maps[i][l, m]
				relResidualsNights[i] = relResidualsNights[i] + relResidual_Maps[i][l, m]
				cNightfromMap = cNightfromMap + 1;
			else:
				absResidualsDays[i] = absResidualsDays[i] + dif_Maps[i][l, m]
				relResidualsDays[i] = relResidualsDays[i] + relResidual_Maps[i][l, m]
				cDayfromMap = cDayfromMap + 1; 

absResidualsHpluses = {}
absResidualsNoHpluses = {}
relResidualsHpluses = {}
relResidualsNoHpluses = {}

for i in ElementArr:
	absResidualsHpluses[i] = 0
	absResidualsNoHpluses[i] = 0
	relResidualsHpluses[i] = 0
	relResidualsNoHpluses[i] = 0
	cHplusfromMap = 0; cNoHplusfromMap = 0;
	for l in range(0, AltiMap.shape[0]):
		for m in range(0, AltiMap.shape[1]):
			HplusThisTile = HplusMap[l, m]
			if HplusThisTile == 0:
				absResidualsNoHpluses[i] = absResidualsNoHpluses[i] + dif_Maps[i][l, m]
				relResidualsNoHpluses[i] = relResidualsNoHpluses[i] + relResidual_Maps[i][l, m]
				cNoHplusfromMap = cNoHplusfromMap + 1;
			else:
				absResidualsHpluses[i] = absResidualsHpluses[i] + dif_Maps[i][l, m]
				relResidualsHpluses[i] = relResidualsHpluses[i] + relResidual_Maps[i][l, m]
				cHplusfromMap = cHplusfromMap + 1; 

print("\n")

# Output data to files
for i in ElementArr:
	np.savetxt(filename_elem_rat_Map[i], np.multiply(elem_rat_Maps[i], np.reciprocal(passCounts)), delimiter=",")
	np.savetxt(filename_dif_Map[i], np.multiply(dif_Maps[i], np.reciprocal(passCounts)), delimiter=",")
	np.savetxt(filename_relResidual_Map[i], np.multiply(relResidual_Maps[i], np.reciprocal(passCounts)), delimiter=",")
	np.savetxt(filename_true_elem_rat_Map[i], np.multiply(true_elem_rat_Maps[i], np.reciprocal(passCounts)), delimiter=",")

np.savetxt(filename_passCounts, passCounts, delimiter=",")
np.savetxt(filename_Alti, np.multiply(AltiMap, np.reciprocal(passCounts)), delimiter=",")

for k in MetricsArr:
	np.savetxt(filenames_Similarities[k], np.multiply(Similarities[k], np.reciprocal(passCounts)), delimiter=",")

accFile = open(statsFilename,"w")

for k in MetricsArr:
	accFile.write('%s_425_850km=%4.2f\n' % (k, handle_zero_division(Sim425s[k],c425)*100))
	accFile.write('%s_850_1275km=%4.2f\n' % (k, handle_zero_division(Sim850s[k],c850)*100))
	accFile.write('%s_1275_1700km=%4.2f\n' % (k, handle_zero_division(Sim1700s[k],c1700)*100))

for k in MetricsArr:
	accFile.write('%s_Dayside=%4.2f\n' % (k, handle_zero_division(SimDays[k],cDay)*100))
	accFile.write('%s_Nightside=%4.2f\n' % (k, handle_zero_division(SimNights[k],cNight)*100))
	
	accFile.write('%s_Hplus=%4.2f\n' % (k, handle_zero_division(SimHpluses[k],cHplus)*100))
	accFile.write('%s_NoHplus=%4.2f\n' % (k, handle_zero_division(SimNoHpluses[k],cNoHplus)*100))

np.savetxt(filename_DayNight, np.multiply(DayNightMap, np.reciprocal(passCounts)), delimiter=",")
np.savetxt(filename_Hplus, np.multiply(HplusMap, np.reciprocal(passCounts)), delimiter=",")

for i in ElementArr:
	accFile.write('%s_meanAbsRes=%4.2f\n' % (i, meanAbsResiduals[i]*100))
	accFile.write('%s_meanRelRes=%4.2f\n' % (i, meanRelResiduals[i]*100))
	accFile.write('%s_absRes_425_850km=%4.2f\n' % (i, handle_zero_division(absResiduals425s[i],c425fromMap)*100))
	accFile.write('%s_absRes_850_1275km=%4.2f\n' % (i, handle_zero_division(absResiduals850s[i],c850fromMap)*100))
	accFile.write('%s_absRes_1275_1700km=%4.2f\n' % (i, handle_zero_division(absResiduals1700s[i],c1700fromMap)*100))
	accFile.write('%s_relRes_425_850km=%4.2f\n' % (i, handle_zero_division(relResiduals425s[i],c425fromMap)*100))
	accFile.write('%s_relRes_850_1275km=%4.2f\n' % (i, handle_zero_division(relResiduals850s[i],c850fromMap)*100))
	accFile.write('%s_relRes_1275_1700km=%4.2f\n' % (i, handle_zero_division(relResiduals1700s[i],c1700fromMap)*100))

for i in ElementArr:
	accFile.write('%s_absRes_Dayside=%4.2f\n' % (i, handle_zero_division(absResidualsDays[i],cDayfromMap)*100))
	accFile.write('%s_absRes_Nightside=%4.2f\n' % (i, handle_zero_division(absResidualsNights[i],cNightfromMap)*100))
	accFile.write('%s_relRes_Dayside=%4.2f\n' % (i, handle_zero_division(relResidualsDays[i],cDayfromMap)*100))
	accFile.write('%s_relRes_Nightside=%4.2f\n' % (i, handle_zero_division(relResidualsNights[i],cNightfromMap)*100))

	accFile.write('%s_absRes_Hplus=%4.2f\n' % (i, handle_zero_division(absResidualsHpluses[i],cHplusfromMap)*100))
	accFile.write('%s_absRes_NoHplus=%4.2f\n' % (i, handle_zero_division(absResidualsNoHpluses[i],cNoHplusfromMap)*100))
	accFile.write('%s_relRes_Hplus=%4.2f\n' % (i, handle_zero_division(relResidualsHpluses[i],cHplusfromMap)*100))
	accFile.write('%s_relRes_NoHplus=%4.2f\n' % (i, handle_zero_division(relResidualsNoHpluses[i],cNoHplusfromMap)*100))

accFile.write('Mean1DAbsRes=%4.2f\n' % (mean(meanAbsResiduals.values())*100))
accFile.write('Mean1DRelRes=%4.2f\n' % (mean(meanRelResiduals.values())*100))

accFile.close

# Print at bottom the total similarities
for k in MetricsArr:
	print('   . Total Similarity {} = {:4.2f}'.format(k, np.mean(Similarities[k])*100))

print("\n")
print('   . Mean 1D Absolute Residual = {:4.2f}'.format(mean(meanAbsResiduals.values())*100))
print('   . Mean 1D Relative Residual = {:4.2f}'.format(mean(meanRelResiduals.values())*100))
print("\n")