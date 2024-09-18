#!/usr/bin/env python

## First DNN Try in Python

#  Instructions and Notes
#  ------------
#  Note that this script is developed from ANN.py and could have some artifacts left from it.

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import h5py
import sys
import csv
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from dnn_app_utils import *

import pandas as pd

## =========== Part 1: Loading and Visualizing Data and Setting Parameters =============
#
#

DNNtask = sys.argv[1]
featSet = sys.argv[2]

testName = sys.argv[3]
trainingName = sys.argv[4]
testSetName = sys.argv[5]

MineralArr = sys.argv[6].split(',')
ElementArr = sys.argv[7].split(',')

totMin = len(MineralArr)
totElem = len(ElementArr)

Pred_Surf_ElementArr = []
# Define predicted surface elements column names
for i in ElementArr:
	Pred_Surf_ElementArr.append(str('Pred_Surf_' + i))

# Remove O_2 and Si in case of elements-modified task
if DNNtask == "elements-modified" and "O_2" in ElementArr:
	Pred_Surf_ElementArr.remove("Pred_Surf_O_2")
if DNNtask == "elements-modified" and "Si" in ElementArr:
	Pred_Surf_ElementArr.remove("Pred_Surf_Si")

Pred_Surf_MineralArr = []
# Define predicted minerals column names
for i in MineralArr:
	Pred_Surf_MineralArr.append(str('Pred_Surf_' + i))

# Input files
testdataFilename = './inputs/' + testSetName + '/AI_testSet_Processed_redAI.csv'
testdataFilenameH5 = './inputs/' + testSetName + '/AI_testSet_Processed_redAI.h5'
jsonFile = "./trainings/" + trainingName + "/Model.json"
paramFile = "./trainings/" + trainingName + "/Parameters.h5"
stdFileName = "./trainings/" + trainingName + "/Std.h5"

# Output files
outFile = "./tests/" + testName + "/subtest_" + testSetName + "/AI_testSet_Processed_redAI_Pred_MVR.csv"
accFilename = "./tests/" + testName + "/subtest_" + testSetName + "/AI_testSet_accuracy_MVR.info"

# Load test data already grouped into features and output by feature engineering module
print ("  .. Loading test data ...             ", end = '')

df_test_data = pd.read_csv(testdataFilename, encoding = 'utf-8').fillna(0)

# read from hdf5 file into dataframes
df_test_x_orig = pd.read_hdf(testdataFilenameH5, 'test_x_orig')
df_test_y_orig = pd.read_hdf(testdataFilenameH5, 'test_y_orig')

# transform dataframes to numpy arrays
test_x_orig = df_test_x_orig.to_numpy()
test_y_orig = df_test_y_orig.to_numpy()

stdFile = h5py.File(stdFileName,"r")

# Normalize output sum in case of modified elements prediction (without Si and O_2)
if DNNtask == "elements-modified":
	test_y_element_sums_by_row = df_test_y_orig.sum(axis=1).to_numpy()
	test_y_for_model = test_y_orig / test_y_element_sums_by_row[:, None]
else:
	test_y_for_model = test_y_orig


# Set the input layer size and number of labels
input_layer_size = test_x_orig.shape[1];
num_labels = test_y_for_model.shape[1];

print ("Done! \n")

### CONSTANTS DEFINING THE MODEL ####
n_x = input_layer_size
n_y = num_labels

# Load the test parameters
mean = np.array(stdFile.get('mean'))
scale = np.array(stdFile.get('scale'))

# load json and create model
json_file = open(jsonFile, 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights(paramFile)

# Reshaping
print ("test_x_orig shape: " + str(test_x_orig.shape))
test_x_std = (test_x_orig - mean) / scale
test_x_flatten = test_x_std.reshape(test_x_std.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x = test_x_flatten/1.
test_y = test_y_for_model.reshape((num_labels, test_y_for_model.shape[0]))

# Explore your dataset
m_test = test_x_orig.shape[0]
num_px = test_x_orig.shape[1]

print ("Number of test examples: " + str(m_test))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))

## ================= Part 2: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

# evaluate loaded model on test data
model.compile(optimizer = "Adam", loss = "kl_divergence", metrics = [euclidean_sim4])

print ("\n  .. Prediction: ...Done!\n");
predictions_test = model.predict(test_x_std)
similarity = np.zeros((1,m_test))
y_mean = np.mean(test_y_for_model, axis=0).reshape(1,test_y_for_model.shape[1])

# other similarity metrics
cos_sim = np.zeros((1,m_test))
sim_e1 = np.zeros((1,m_test))
sim_e2 = np.zeros((1,m_test))
sim_e3 = np.zeros((1,m_test))
sim_e4 = np.zeros((1,m_test))
r2 = np.zeros((1,m_test))

for i in range(predictions_test.shape[0]):
	p_vector = predictions_test[i,:]
	y_vector = test_y_for_model[i,:]
	p_vector = p_vector.reshape((1, p_vector.shape[0]))
	y_vector = y_vector.reshape((1, y_vector.shape[0]))
	
	cos_sim[0,i] = np.dot(p_vector, y_vector.T)/(np.linalg.norm(p_vector)*np.linalg.norm(y_vector)) # cos_sim
	sim_e1[0,i] = (np.sum(y_vector, axis=1) - euclidean(p_vector, y_vector, np.sum(y_vector, axis=1))) / np.sum(y_vector, axis=1) # for euclidean_sim1
	sim_e2[0,i] = 1 - np.square(euclidean(p_vector, y_vector, np.sum(y_vector, axis=1))) / np.sum(np.square(y_vector), axis=1) # for euclidean_sim2
	sim_e3[0,i] = 1 - euclidean(p_vector, y_vector, np.sum(y_vector, axis=1)) / np.sqrt(np.sum(np.square(y_vector), axis=1)) # for euclidean_sim3
	r2[0,i] = 1 - np.sum(np.square(y_vector - p_vector), axis=1) / np.sum(np.square(y_vector - y_mean), axis=1) # for R2
	
	sim_e4[0,i] = sim_e1[0,i] * cos_sim[0,i] # for euclidean_sim4 = euclidean_sim1 * cos_sim

scores = model.evaluate(test_x_std, test_y_for_model, verbose=0)

print("   . Cosine Similarity: "  + str((np.sum(sim_e3)/m_test*100)))
print("   . Similarity E1:     "  + str((np.sum(sim_e1)/m_test*100)))
print("   . Similarity E2:     "  + str((np.sum(sim_e2)/m_test*100)))
print("   . Similarity E3:     "  + str((np.sum(sim_e3)/m_test*100)))
print("   . R-Squared:         "  + str((np.sum(r2)/m_test*100)) + "\n")

print("   . Similarity E4:     "  + str((np.sum(sim_e4)/m_test*100)) + "\n")

## ================ Part 3: Save testdata, Parameters and Metadata ================

# Write predictions and similarity to dataframe and .csv
prediction_columns = []
if DNNtask == "minerals":
	for i in Pred_Surf_MineralArr:
		prediction_columns.append(i)
elif DNNtask == "elements":
	for i in Pred_Surf_ElementArr:
		prediction_columns.append(i)
elif DNNtask == "elements-modified":
	for i in Pred_Surf_ElementArr:
		prediction_columns.append(i)
	predictions_test = predictions_test * test_y_element_sums_by_row[:, None]

df_predictions = pd.DataFrame(predictions_test, columns = prediction_columns)
similiarities_array = np.concatenate((cos_sim.T, sim_e1.T, sim_e2.T, sim_e3.T, r2.T, sim_e4.T), axis=1)
df_similarities = pd.DataFrame(similiarities_array, columns = ['Cos_Sim', 'ES1', 'ES2', 'ES3', 'R2', 'ES4'])
df_all_data = pd.concat([df_test_data, df_predictions, df_similarities], axis = 1)

df_all_data.to_csv(outFile, index=False)

accFile = open(accFilename,"w+")
accFile.write("algorithm=MVR\n")
accFile.write("DNNtask=" + DNNtask + "\n")
accFile.write("featSet=" + featSet + "\n")
accFile.write("cos_sim=%2.2f\n" % (np.sum(cos_sim)/m_test*100))
accFile.write("sim_E1=%2.2f\n" % (np.sum(sim_e1)/m_test*100))
accFile.write("sim_E2=%2.2f\n" % (np.sum(sim_e2)/m_test*100))
accFile.write("sim_E3=%2.2f\n" % (np.sum(sim_e3)/m_test*100))
accFile.write("sim_R2=%2.2f\n" % (np.sum(r2)/m_test*100))
accFile.write("sim_E4=%2.2f\n" % (np.sum(sim_e4)/m_test*100))
accFile.close

print ("  .. Parameters and Data: ...Saved!")
