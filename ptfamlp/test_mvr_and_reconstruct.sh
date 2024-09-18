#!/bin/bash
# this script is an example script to make predictions with a pre-trained AI

# set the home directory to the present working directory
PTFAdir="$PWD"

# Arguments for the python program
DNNtask="elements-modified"
featSet="F11"
combineMode="Day_Pref_Avg"

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
Elements="Na Al Si O_2 Ca Mg Fe K Zn S H_2O"; ElementsArr=($Elements)
ElementsPy=${Elements// /,}
MetricsPy="Cos_Sim,ES1,ES2,ES3,R2,ES4"

# Create tests directories
if [ ! -d "$PTFAdir/tests/" ]; then mkdir $PTFAdir/tests/; fi
	if [ ! -d "$PTFAdir/tests/$testName/" ]; then mkdir $PTFAdir/tests/$testName/; fi


for idx in ${!testSetNamesArr[@]} ; do
	testSetName=${testSetNamesArr[$idx]};

	echo "Subtest: ${testSetName}"

	if [ ! -d "$PTFAdir/tests/${testName}/subtest_${testSetName}" ]; then mkdir "$PTFAdir/tests/${testName}/subtest_${testSetName}"; fi

	python3 -u "$PTFAdir/MVR_test.py" $DNNtask $featSet $testName $trainingName ${testSetName} $MineralsPy $ElementsPy
done


# Reconstruct maps from the AI testing
for idx in ${!testSetNamesArr[@]} ; do
	testSetName=${testSetNamesArr[$idx]};
	dirSubtest="${testName}/subtest_${testSetName}"

	echo "Subtest: ${testSetName}"

	if [[ $DNNtask == "elements" ]] || [[ $DNNtask == "elements-modified" ]]; then
		# create the rebinned maps and write them to .csv files
		python3 "$PTFAdir/mapElemReconstruct.py" "$PTFAdir/tests/${dirSubtest}/AI_testSet_Processed_redAI_Pred_MVR.csv" "${dirSubtest}" $DNNtask "${MineralsPy}" "${ElementsPy}" "${MetricsPy}"

		for idx in ${!ElementsArr[@]} ; do
			Element=${ElementsArr[$idx]}; ElementName=${ElementsArr[$idx]};
			if [[ $DNNtask == "elements-modified" ]] && ([[ $Element == "O_2" ]] || [[ $Element == "Si" ]]); then
				:
			else
				# invoke the gnuplot file to plot the map
				gnuplot -e "mapdir='$PTFAdir/tests/${dirSubtest}/'; mapfile='Map_elem_${Element}_rat_MVR.csv'; plotfile='Map_elem_${Element}_rat_MVR.png'; type='Elemental ${ElementName} surface composition (ratio)(MVR predicted)'" "$PTFAdir/Regolith_element_ratios.gpl";
				gnuplot -e "mapdir='$PTFAdir/tests/${dirSubtest}/'; mapfile='Map_dif_${Element}_MVR.csv'; plotfile='Map_dif_${Element}_MVR.png'; type='Prediction Difference for ${ElementName} (Residual from MVR prediction)'" "$PTFAdir/Regolith_element_ratios.gpl";
				gnuplot -e "mapdir='$PTFAdir/tests/${dirSubtest}/'; mapfile='Map_relResidual_${Element}_MVR.csv'; plotfile='Map_relResidual_${Element}_MVR.png'; type='Relative residual for ${ElementName} (From MVR prediction)'" "$PTFAdir/Regolith_element_ratios_lin.gpl";
				gnuplot -e "mapdir='$PTFAdir/tests/${dirSubtest}/'; mapfile='Map_true_elem_${Element}_rat_MVR.csv'; plotfile='Map_true_elem_${Element}_rat_MVR.png'; type='Elemental ${ElementName} surface composition (ratio)(True)'" "$PTFAdir/Regolith_element_ratios.gpl";
			fi
		done
	fi
done

# Generate compound maps from the AI testing
if [[ $DNNtask == "elements" ]] || [[ $DNNtask == "elements-modified" ]]; then
	# create the rebinned maps and write them to .csv files
	python3 "$PTFAdir/mapElemCompound.py" "${testName}" ${testSetNames} $DNNtask "${MineralsPy}" "${ElementsPy}" "${MetricsPy}" ${combineMode}

	for idx in ${!ElementsArr[@]} ; do
		Element=${ElementsArr[$idx]}; ElementName=${ElementsArr[$idx]};
		if [[ $DNNtask == "elements-modified" ]] && ([[ $Element == "O_2" ]] || [[ $Element == "Si" ]]); then
			:
		else
			# invoke the gnuplot file to plot the map
			gnuplot -e "mapdir='$PTFAdir/tests/${testName}/'; mapfile='Compound_Map_elem_${Element}_rat_MVR.csv'; plotfile='Compound_Map_elem_${Element}_rat_MVR.png'; type='Elemental ${ElementName} surface composition (ratio)(MVR predicted)'" "$PTFAdir/Regolith_element_ratios.gpl";
			gnuplot -e "mapdir='$PTFAdir/tests/${testName}/'; mapfile='Compound_Map_dif_${Element}_MVR.csv'; plotfile='Compound_Map_dif_${Element}_MVR.png'; type='Prediction Difference for ${ElementName} (Residual from MVR prediction)'" "$PTFAdir/Regolith_element_ratios.gpl";
			gnuplot -e "mapdir='$PTFAdir/tests/${testName}/'; mapfile='Compound_Map_relResidual_${Element}_MVR.csv'; plotfile='Compound_Map_relResidual_${Element}_MVR.png'; type='Relative residual for ${ElementName} (From MVR prediction)'" "$PTFAdir/Regolith_element_ratios_lin.gpl";
			gnuplot -e "mapdir='$PTFAdir/tests/${testName}/'; mapfile='Compound_Map_true_elem_${Element}_rat_MVR.csv'; plotfile='Compound_Map_true_elem_${Element}_rat_MVR.png'; type='Elemental ${ElementName} surface composition (ratio)(True)'" "$PTFAdir/Regolith_element_ratios.gpl";
		fi
	done
fi
