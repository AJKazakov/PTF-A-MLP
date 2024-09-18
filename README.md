Readme: PTF-A-MLP
Version: 0.1.0
Module: PTF-A-MLP

Author:		Adrian Kazakov (adrian.kazakov@inaf.it)
Date:		2024-09-18

Modified:	Adrian Kazakov
Date:		2024-09-18

Description:
==============================================================
This software is able to train and test a deep neural network
called a multilayer perceptron to predict/reconstruct the
elemental surface composition below a simulated in-situ exospheric
measurement.

It includes scripts written in Python, Bash shell, and gnuplot.

Requirements:
==============================================================
1.	Requires Python and other dependencies listed in ./requirements.txt
2.	Requires gnuplot

----------------------------------
Built on:
----------------------------------
Windows Subsystem for Linux on Windows 11

Tested on:
----------------------------------
Windows Subsystem for Linux on Windows 11

Installation:
==============================================================
No installation needed.

Configuration:
==============================================================
No configuration needed.

Usage:
==============================================================
The two Python scripts for training and testing of the DNN MLP algorithm
are invoked with arguments from the provided example Bash shell scripts.

To run the DNN MLP training program:
1. Place the inputs to the training (training and dev set) in the ./ptfamlp/inputs/ directory.
2. Change the inputs/outputs in the code of the script train_mvr.sh
3. Run the script: $ bash ./ptfamlp/train_mvr.sh

To run the DNN MLP testing/prediction program [with map reconstructions]:
1. Place the inputs to the testing (testing sets) in the ./ptfamlp/inputs/ directory.
2. Make sure that there is the training available in the ./ptfamlp/trainings/ directory.
3. Change the inputs/outputs in the code of the script test_mvr.sh
4. Run the script: $ bash ./ptfamlp/test_mvr.sh 
                   [or $ bash ./ptfamlp/test_mvr_and_reconstruct.sh]

Uninstallation and Cleaning:
==============================================================
No uninstallation needed.
