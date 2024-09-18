#!/bin/bash
# this is an AI training example script

# set the home directory to the present working directory
PTFAdir="$PWD"

#### CHANGE INPUTS AND OUTPUTS NAMES ####
# Inputs of training - place the datasets in their original form/directory in the inputs directory
trainSetName="trainset_11finer50mid_modello_elem-mod_F11_300_J2000_MVR001jbr"
devSetName="devset_11finer50mid_modello_elem-mod_F11_500_d01"
# Output of training
trainingName="new_training_name"
#########################################

# Other arguments for the python program
tuneControl="OFF"

learnRate=0.0005
batchSize=512
regCoef=0.00001
layArray=600,500,350,250
numEpochs=40
RNGseed=1

DNNtask="elements-modified"
featSet="F11"

MineralsPy="Albite,Anorthite,Diopside,Enstatite,Ferrosilite,Hedenbergite,Orthoclase,Sphalerite,Water"
ElementsPy="Na,Al,Si,O_2,Ca,Mg,Fe,K,Zn,S,H_2O"

# Create training directories
if [ ! -d "$PTFAdir/trainings/" ]; then mkdir $PTFAdir/trainings/; fi
	if [ ! -d "$PTFAdir/trainings/$trainingName/" ]; then mkdir $PTFAdir/trainings/$trainingName/; fi

# Execute python script to train the algorithm
python3 -u "$PTFAdir/MVR.py" $tuneControl $learnRate $batchSize $regCoef $layArray $numEpochs $RNGseed $DNNtask $featSet $trainingName $trainSetName $devSetName $MineralsPy $ElementsPy
