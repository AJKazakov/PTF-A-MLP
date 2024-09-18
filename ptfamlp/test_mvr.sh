#!/bin/bash
# this script is an example script to make predictions with a pre-trained AI

# set the home directory to the present working directory
PTFAdir="$PWD"

# Arguments for the python program
DNNtask="elements-modified"
featSet="F11"

# Output of testing
testName="new_test_name"

# The training should be in its original form/directory inside the trainings directory
trainingName="new_training_name"

# Inputs of testing - place the datasets in their original form/directory in the inputs directory
testSetNames="testset_11finer50mid_modello_elem-mod_F11_200_t01,testset_11finer50mid_modello_elem-mod_F11_360_200_t01"

# Put the test set names in an array
IFS=',' read -ra testSetNamesArr <<< "$testSetNames"

# Resolve mineral and element names strings for Python command-line passed arguments
MineralsPy="Albite,Anorthite,Diopside,Enstatite,Ferrosilite,Hedenbergite,Orthoclase,Sphalerite,Water"
ElementsPy="Na,Al,Si,O_2,Ca,Mg,Fe,K,Zn,S,H_2O"
MetricsPy="Cos_Sim,ES1,ES2,ES3,R2,ES4"

# Create tests directories
if [ ! -d "$PTFAdir/tests/" ]; then mkdir $PTFAdir/tests/; fi
	if [ ! -d "$PTFAdir/tests/$testName/" ]; then mkdir $PTFAdir/tests/$testName/; fi

# Perform two subsequent sub-tests each 360 degrees of Mercury TAA apart
for idx in ${!testSetNamesArr[@]} ; do
	testSetName=${testSetNamesArr[$idx]};

	echo "Subtest: ${testSetName}"

	if [ ! -d "$PTFAdir/tests/${testName}/subtest_${testSetName}" ]; then mkdir "$PTFAdir/tests/${testName}/subtest_${testSetName}"; fi

	python3 -u "$PTFAdir/MVR_test.py" $DNNtask $featSet $testName $trainingName ${testSetName} $MineralsPy $ElementsPy
done
