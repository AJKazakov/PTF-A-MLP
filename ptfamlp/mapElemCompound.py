#!/usr/bin/env python

## Compound Map Prediction with the MVR algorithm

import sys
import csv

import numpy as np
import pandas as pd

from statistics import mean

def handle_zero_division(a, b):
    return 0 if b == 0 else a / b

# Read program arguments
testName = sys.argv[1]
testSetNames = sys.argv[2].split(',')
DNNtask = sys.argv[3]
MineralArr = sys.argv[4].split(',')
ElementArr = sys.argv[5].split(',')
MetricsArr = sys.argv[6].split(',')
combineMode = sys.argv[7]

totTests = len(testSetNames)
totMin = len(MineralArr)
totElem = len(ElementArr)
totMetrics = len(MetricsArr)

statsFilename_comp = './tests/' + testName + '/AI_Compound_stats_MVR.info'

# Remove O_2 and Si in case of elements-modified task
if DNNtask == "elements-modified" and "O_2" in ElementArr:
	ElementArr.remove("O_2")
if DNNtask == "elements-modified" and "Si" in ElementArr:
	ElementArr.remove("Si")

# Declare filenames and maps for the compound prediction (also for each element and for each metrics)
filename_passCounts_comp = './tests/' + testName + '/Compound_Map_PassCounts_MVR.csv'
filename_Alti_comp = './tests/' + testName + '/Compound_Map_Altitude_MVR.csv'
filename_DayNight_comp = './tests/' + testName + '/Compound_Map_DayNight_MVR.csv'
filename_Hplus_comp = './tests/' + testName + '/Compound_Map_Hplus_MVR.csv'

passCounts_comp = np.zeros((18,36))
AltiMap_comp = np.zeros((18,36))
DayNightMap_comp = np.zeros((18,36))
HplusMap_comp = np.zeros((18,36))

filename_elem_rat_Map_comp = {}
filename_dif_Map_comp = {}
filename_relResidual_Map_comp = {}
filename_true_elem_rat_Map_comp = {}

elem_rat_Maps_comp = {}
dif_Maps_comp = {}
relResidual_Maps_comp = {}
true_elem_rat_Maps_comp = {}

for i in ElementArr:
	filename_elem_rat_Map_comp[i] = './tests/' + testName + '/Compound_Map_elem_' + i + '_rat_MVR.csv'
	filename_dif_Map_comp[i] = './tests/' + testName + '/Compound_Map_dif_' + i + '_MVR.csv'
	filename_relResidual_Map_comp[i] = './tests/' + testName + '/Compound_Map_relResidual_' + i + '_MVR.csv'
	filename_true_elem_rat_Map_comp[i] = './tests/' + testName + '/Compound_Map_true_elem_' + i + '_rat_MVR.csv'

	elem_rat_Maps_comp[i] = np.zeros((18,36))
	dif_Maps_comp[i] = np.zeros((18,36))
	relResidual_Maps_comp[i] = np.zeros((18,36))
	true_elem_rat_Maps_comp[i] = np.zeros((18,36))

filenames_Similarities_comp = {}
Similarities_comp = {}

for k in MetricsArr:
	filenames_Similarities_comp[k] = './tests/' + testName + '/Compound_Map_Similarity_' + k + '_MVR.csv'
	Similarities_comp[k] = np.zeros((18,36))

# Declare dictionaries for each subtest
testSubtestDirs = {}

filename_passCounts = {}
filename_Alti = {}
filename_DayNight = {}
filename_Hplus = {}

passCounts = {}
AltiMap = {}
DayNightMap = {}
HplusMap = {}

filename_elem_rat_Map = {}
filename_dif_Map = {}
filename_relResidual_Map = {}
filename_true_elem_rat_Map = {}

elem_rat_Maps = {}
dif_Maps = {}
relResidual_Maps = {}
true_elem_rat_Maps = {}

filenames_Similarities = {}
Similarities = {}

for j in testSetNames:
	testSubtestDirs[j] = './tests/' + testName + '/subtest_' + j

	filename_passCounts[j] = testSubtestDirs[j] + '/Map_PassCounts_MVR.csv'
	filename_Alti[j] = testSubtestDirs[j] + '/Map_Altitude_MVR.csv'
	filename_DayNight[j] = testSubtestDirs[j] + '/Map_DayNight_MVR.csv'
	filename_Hplus[j] = testSubtestDirs[j] + '/Map_Hplus_MVR.csv'

	passCounts[j] = np.genfromtxt(filename_passCounts[j], delimiter=",")
	AltiMap[j] = np.genfromtxt(filename_Alti[j], delimiter=",")
	DayNightMap[j] = np.genfromtxt(filename_DayNight[j], delimiter=",")
	HplusMap[j] = np.genfromtxt(filename_Hplus[j], delimiter=",")

	# Declare nested dictionaries for each subtest and each element
	filename_elem_rat_Map[j] = {}
	filename_dif_Map[j] = {}
	filename_relResidual_Map[j] = {}
	filename_true_elem_rat_Map[j] = {}

	elem_rat_Maps[j] = {}
	dif_Maps[j] = {}
	relResidual_Maps[j] = {}
	true_elem_rat_Maps[j] = {}

	# Load the maps as numpy arrays for each subtest and each element
	for i in ElementArr:
		filename_elem_rat_Map[j][i] = testSubtestDirs[j] + '/Map_elem_' + i + '_rat_MVR.csv'
		filename_dif_Map[j][i] = testSubtestDirs[j] + '/Map_dif_' + i + '_MVR.csv'
		filename_relResidual_Map[j][i] = testSubtestDirs[j] + '/Map_relResidual_' + i + '_MVR.csv'
		filename_true_elem_rat_Map[j][i] = testSubtestDirs[j] + '/Map_true_elem_' + i + '_rat_MVR.csv'

		elem_rat_Maps[j][i] = np.genfromtxt(filename_elem_rat_Map[j][i], delimiter=",")
		dif_Maps[j][i] = np.zeros((18,36))
		relResidual_Maps[j][i] = np.zeros((18,36))
		true_elem_rat_Maps[j][i] = np.genfromtxt(filename_true_elem_rat_Map[j][i], delimiter=",")

		# If there is a zero array for the compound true map, copy the first subtest' true map
		if np.sum(true_elem_rat_Maps_comp[i]) == 0:
			true_elem_rat_Maps_comp[i] = true_elem_rat_Maps[j][i]

	# Declare nested dictionaries for each subtest and each metrics
	filenames_Similarities[j] = {}
	Similarities[j] = {}

	# Load the maps as numpy arrays for each subtest and each metrics
	for k in MetricsArr:
		filenames_Similarities[j][k] = testSubtestDirs[j] + '/Map_Similarity_' + k + '_MVR.csv'
		Similarities[j][k] = np.genfromtxt(filenames_Similarities[j][k], delimiter=",")

################################################################################
# Combine the maps from the different subtests in mode defined by "combineMode"
################################################################################

print(' ... Combining the subtest maps... ')

if combineMode == "Day_Pref_Avg":
	# Combine the Day/Night map coverage:
	for j in testSetNames:
		DayNightMap_comp += DayNightMap[j]

	# Loop over each cell and combine the subtest predictions
	# based on the availability of dayside predictions in any test.
	# Note: The results are weighted maps by the pass counts for each cell
	#       These are resolved to averages per pass count further below.
	for Lat in range(0, 18):
		for Lon in range(0, 36):
			if DayNightMap_comp[Lat][Lon] != 0:
				# If there is a dayside prediction, combine only the dayside predictions for the cell
				for j in testSetNames:
					if DayNightMap[j][Lat][Lon] > 0:	# this condition check IS necessary, because not every subtest contributes here
						for i in ElementArr:
							elem_rat_Maps_comp[i][Lat][Lon] += passCounts[j][Lat][Lon]*elem_rat_Maps[j][i][Lat][Lon]
							dif_Maps_comp[i][Lat][Lon] += passCounts[j][Lat][Lon]*(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]))
							if true_elem_rat_Maps_comp[i][Lat][Lon] == 0:
								new_relative_residual = handle_zero_division(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]), np.mean(true_elem_rat_Maps_comp[i][:][:]))
							else:
								new_relative_residual = min(handle_zero_division(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]), true_elem_rat_Maps_comp[i][Lat][Lon]),
															handle_zero_division(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]), np.mean(true_elem_rat_Maps_comp[i][:][:])))
							relResidual_Maps_comp[i][Lat][Lon] += new_relative_residual
						for k in MetricsArr:
							Similarities_comp[k][Lat][Lon] += passCounts[j][Lat][Lon]*Similarities[j][k][Lat][Lon]

						passCounts_comp[Lat][Lon] += passCounts[j][Lat][Lon]
						AltiMap_comp[Lat][Lon] += passCounts[j][Lat][Lon]*AltiMap[j][Lat][Lon]
						HplusMap_comp[Lat][Lon] += passCounts[j][Lat][Lon]*HplusMap[j][Lat][Lon]

			else:
				# If there is no dayside prediction, combine all the nightside predictions for the cell
				for j in testSetNames:
					if DayNightMap[j][Lat][Lon] == 0:	# this condition check IS NOT necessary, but just nice to show explicitly
						for i in ElementArr:
							elem_rat_Maps_comp[i][Lat][Lon] += passCounts[j][Lat][Lon]*elem_rat_Maps[j][i][Lat][Lon]
							dif_Maps_comp[i][Lat][Lon] += passCounts[j][Lat][Lon]*(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]))
							if true_elem_rat_Maps_comp[i][Lat][Lon] == 0:
								new_relative_residual = handle_zero_division(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]), np.mean(true_elem_rat_Maps_comp[i][:][:]))
							else:
								new_relative_residual = min(handle_zero_division(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]), true_elem_rat_Maps_comp[i][Lat][Lon]),
															handle_zero_division(np.absolute(elem_rat_Maps[j][i][Lat][Lon] - true_elem_rat_Maps_comp[i][Lat][Lon]), np.mean(true_elem_rat_Maps_comp[i][:][:])))
							relResidual_Maps_comp[i][Lat][Lon] += new_relative_residual
						for k in MetricsArr:
							Similarities_comp[k][Lat][Lon] += passCounts[j][Lat][Lon]*Similarities[j][k][Lat][Lon]

						passCounts_comp[Lat][Lon] += passCounts[j][Lat][Lon]
						AltiMap_comp[Lat][Lon] += passCounts[j][Lat][Lon]*AltiMap[j][Lat][Lon]
						HplusMap_comp[Lat][Lon] += passCounts[j][Lat][Lon]*HplusMap[j][Lat][Lon]

# Calculate the average maps from the weighed maps
AltiMap_comp = np.multiply(AltiMap_comp, np.reciprocal(passCounts_comp))
DayNightMap_comp = np.multiply(DayNightMap_comp, np.reciprocal(passCounts_comp))
HplusMap_comp = np.multiply(HplusMap_comp, np.reciprocal(passCounts_comp))

for i in ElementArr:
	elem_rat_Maps_comp[i] = np.multiply(elem_rat_Maps_comp[i], np.reciprocal(passCounts_comp))
	dif_Maps_comp[i] = np.multiply(dif_Maps_comp[i], np.reciprocal(passCounts_comp))
	relResidual_Maps_comp[i] = np.multiply(relResidual_Maps_comp[i], np.reciprocal(passCounts_comp))

for k in MetricsArr:
	Similarities_comp[k] = np.multiply(Similarities_comp[k], np.reciprocal(passCounts_comp))
	 
# Calculate mean absolute and relative residuals for each element
meanAbsResiduals_comp = {}
meanRelResiduals_comp = {}
for i in ElementArr:
	meanAbsResiduals_comp[i] = np.mean(dif_Maps_comp[i])
	meanRelResiduals_comp[i] = np.mean(relResidual_Maps_comp[i])

# Calculate similarity for different altitudes
Sim425s = {}; c425 = 0
Sim850s = {}; c850 = 0
Sim1700s = {}; c1700 = 0
for k in MetricsArr:
	Sim425s[k] = 0;
	Sim850s[k] = 0;
	Sim1700s[k] = 0;

for Lat in range(0, 18):
	for Lon in range(0, 36):
		if AltiMap_comp[Lat][Lon] < 850:
			for k in MetricsArr:
				Sim425s[k] += Similarities_comp[k][Lat][Lon]
			c425 += 1
		elif AltiMap_comp[Lat][Lon] > 1275:
			for k in MetricsArr:
				Sim1700s[k] += Similarities_comp[k][Lat][Lon]
			c1700 += 1
		else:
			for k in MetricsArr:
				Sim850s[k] += Similarities_comp[k][Lat][Lon]
			c850 += 1

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

	for Lat in range(0, 18):
		for Lon in range(0, 36):
			if AltiMap_comp[Lat][Lon] < 850:
				absResiduals425s[i] += dif_Maps_comp[i][Lat][Lon]
				relResiduals425s[i] += relResidual_Maps_comp[i][Lat][Lon]
				c425fromMap += 1;
			elif AltiMap_comp[Lat][Lon] > 1275:
				absResiduals1700s[i] += dif_Maps_comp[i][Lat][Lon]
				relResiduals1700s[i] += relResidual_Maps_comp[i][Lat][Lon]
				c1700fromMap += 1;
			else:
				absResiduals850s[i] += dif_Maps_comp[i][Lat][Lon]
				relResiduals850s[i] += relResidual_Maps_comp[i][Lat][Lon]
				c850fromMap += 1; 

print("\n")

# Calculate the similarity for dayside/nightside and presence/absence of impinging H+ ions (protons)
# Similarity split by Day/Night
SimDays = {}; cDay = 0;
SimNights = {}; cNight = 0;
for k in MetricsArr:
	SimDays[k] = 0;
	SimNights[k] = 0;

for Lat in range(0, 18):
	for Lon in range(0, 36):
		if DayNightMap_comp[Lat][Lon] == 0:
			for k in MetricsArr:
				SimNights[k] += Similarities_comp[k][Lat][Lon]
			cNight += 1
		elif DayNightMap_comp[Lat][Lon] != 0:
			for k in MetricsArr:
				SimDays[k] += Similarities_comp[k][Lat][Lon]
			cDay += 1

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

for Lat in range(0, 18):
	for Lon in range(0, 36):
		if HplusMap_comp[Lat][Lon] == 0:
			for k in MetricsArr:
				SimNoHpluses[k] += Similarities_comp[k][Lat][Lon]
			cNoHplus += 1
		elif HplusMap_comp[Lat][Lon] > 0:
			for k in MetricsArr:
				SimHpluses[k] += Similarities_comp[k][Lat][Lon]
			cHplus += 1
	
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
	for Lat in range(0, 18):
		for Lon in range(0, 36):
			if DayNightMap_comp[Lat][Lon] == 0:
				absResidualsNights[i] += dif_Maps_comp[i][Lat][Lon]
				relResidualsNights[i] += relResidual_Maps_comp[i][Lat][Lon]
				cNightfromMap += 1
			else:
				absResidualsDays[i] += dif_Maps_comp[i][Lat][Lon]
				relResidualsDays[i] += relResidual_Maps_comp[i][Lat][Lon]
				cDayfromMap += 1 

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
	for Lat in range(0, 18):
		for Lon in range(0, 36):
			if HplusMap_comp[Lat][Lon] == 0:
				absResidualsNoHpluses[i] += dif_Maps_comp[i][Lat][Lon]
				relResidualsNoHpluses[i] += relResidual_Maps_comp[i][Lat][Lon]
				cNoHplusfromMap += 1
			else:
				absResidualsHpluses[i] += dif_Maps_comp[i][Lat][Lon]
				relResidualsHpluses[i] += relResidual_Maps_comp[i][Lat][Lon]
				cHplusfromMap += 1

print("\n")


# Output data to files
for i in ElementArr:
	np.savetxt(filename_elem_rat_Map_comp[i], elem_rat_Maps_comp[i], delimiter=",")
	np.savetxt(filename_dif_Map_comp[i], dif_Maps_comp[i], delimiter=",")
	np.savetxt(filename_relResidual_Map_comp[i], relResidual_Maps_comp[i], delimiter=",")
	np.savetxt(filename_true_elem_rat_Map_comp[i], true_elem_rat_Maps_comp[i], delimiter=",")

np.savetxt(filename_passCounts_comp, passCounts_comp, delimiter=",")
np.savetxt(filename_Alti_comp, AltiMap_comp, delimiter=",")

for k in MetricsArr:
	np.savetxt(filenames_Similarities_comp[k], Similarities_comp[k], delimiter=",")

accFile = open(statsFilename_comp,"w")

for k in MetricsArr:
	accFile.write('%s_425_850km=%4.2f\n' % (k, handle_zero_division(Sim425s[k],c425)*100))
	accFile.write('%s_850_1275km=%4.2f\n' % (k, handle_zero_division(Sim850s[k],c850)*100))
	accFile.write('%s_1275_1700km=%4.2f\n' % (k, handle_zero_division(Sim1700s[k],c1700)*100))

for k in MetricsArr:
	accFile.write('%s_Dayside=%4.2f\n' % (k, handle_zero_division(SimDays[k],cDay)*100))
	accFile.write('%s_Nightside=%4.2f\n' % (k, handle_zero_division(SimNights[k],cNight)*100))
	
	accFile.write('%s_Hplus=%4.2f\n' % (k, handle_zero_division(SimHpluses[k],cHplus)*100))
	accFile.write('%s_NoHplus=%4.2f\n' % (k, handle_zero_division(SimNoHpluses[k],cNoHplus)*100))

np.savetxt(filename_DayNight_comp, DayNightMap_comp, delimiter=",")
np.savetxt(filename_Hplus_comp, HplusMap_comp, delimiter=",")

for i in ElementArr:
	accFile.write('%s_meanAbsRes=%4.2f\n' % (i, meanAbsResiduals_comp[i]*100))
	accFile.write('%s_meanRelRes=%4.2f\n' % (i, meanRelResiduals_comp[i]*100))
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

accFile.write('Mean1DAbsRes=%4.2f\n' % (mean(meanAbsResiduals_comp.values())*100))
accFile.write('Mean1DRelRes=%4.2f\n' % (mean(meanRelResiduals_comp.values())*100))

for k in MetricsArr:
	accFile.write('%s=%4.2f\n' % (k, np.mean(Similarities_comp[k])*100))

accFile.close

# Print at bottom the total similarities
for k in MetricsArr:
	print('   . Total Similarity {} = {:4.2f}'.format(k, np.mean(Similarities_comp[k])*100))

print("\n")
print('   . Mean 1D Absolute Residual = {:4.2f}'.format(mean(meanAbsResiduals_comp.values())*100))
print('   . Mean 1D Relative Residual = {:4.2f}'.format(mean(meanRelResiduals_comp.values())*100))
print("\n")