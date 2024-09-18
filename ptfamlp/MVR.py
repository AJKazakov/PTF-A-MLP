#!/usr/bin/env python

## Multivariate Regression first try

#  Instructions and Notes
#  ------------
#  Note that this script is developed from DNNk.py and could have some artifacts left from it.

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
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import glorot_uniform, HeUniform
# Custom libraries
from dnn_app_utils import *
from hyp_opt_utils import *

import pandas as pd

# Set some random seeds
np.random.seed(1)
tf.random.set_seed(2)

## =========== Part 1: Loading and Visualizing Data and Setting Parameters =============
#
#

# Set Some Parameters
tuneControl = sys.argv[1]

# Hyperparameters

# tunable
learnRate = float(sys.argv[2])
print ("  .. Learning rate              = " + str(learnRate))
batchSize = int(sys.argv[3])
print ("  .. Mini-batches size          = " + str(batchSize))
regCoef   = float(sys.argv[4])
print ("  .. Regularization coefficient = " + str(regCoef))
layerArray = np.fromstring(sys.argv[5], dtype=int, sep=',')
print ("  .. layer Array                = " + str(layerArray[:]))
# always manual
numEpochs = int(sys.argv[6])
print ("  .. Number of epochs           = " + str(numEpochs))
RNGseed   = int(sys.argv[7])
print ("  .. RNG seed number            = " + str(RNGseed))

loss_function = "kl_divergence"
output_unit = "Softmax"

DNNtask = sys.argv[8]
featSet = sys.argv[9]

trainingName = sys.argv[10]
trainSetName = sys.argv[11]
devSetName = sys.argv[12]

MineralArr = sys.argv[13].split(',')
ElementArr = sys.argv[14].split(',')

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
dataFilename = './inputs/' + trainSetName + '/AI_trainSet_Processed_redAI.csv'

dataFilenameH5 = './inputs/' + trainSetName + '/AI_trainSet_Processed_redAI.h5'
devdataFilenameH5 = './inputs/' + devSetName + '/AI_devSet_Processed_redAI.h5'

# Output files
outFileGraph = "./trainings/" + trainingName + "/LearnCurve.png"
outFile = "./trainings/" + trainingName + "/AI_trainSet_Processed_redAI_Pred_MVR.csv"
accFileName = "./trainings/" + trainingName + "/AI_trainSet_accuracy_MVR.info"
jsonFile = "./trainings/" + trainingName + "/Model.json"
paramFile = "./trainings/" + trainingName + "/Parameters.h5"
stdFileName = "./trainings/" + trainingName + "/Std.h5"
hyperParFileName = "./trainings/" + trainingName + "/HyperParams.info"

# Load Training Data already grouped into features and output by feature engineering module
print ("  .. Loading Data ...             ", end = '')

df_train_data = pd.read_csv(dataFilename, encoding = 'utf-8').fillna(0)

# read from hdf5 file into dataframes
df_train_x_orig = pd.read_hdf(dataFilenameH5, 'train_x_orig')
df_train_y_orig = pd.read_hdf(dataFilenameH5, 'train_y_orig')

df_dev_x_orig = pd.read_hdf(devdataFilenameH5, 'dev_x_orig')
df_dev_y_orig = pd.read_hdf(devdataFilenameH5, 'dev_y_orig')

# transform dataframes to numpy arrays
train_x_orig = df_train_x_orig.to_numpy()
train_y_orig = df_train_y_orig.to_numpy()

dev_x_orig = df_dev_x_orig.to_numpy()
dev_y_orig = df_dev_y_orig.to_numpy()

# Normalize output sum in case of modified elements prediction (without Si and O_2)
if DNNtask == "elements-modified":
	train_y_element_sums_by_row = df_train_y_orig.sum(axis=1).to_numpy()
	dev_y_element_sums_by_row = df_dev_y_orig.sum(axis=1).to_numpy()
	train_y_for_model = train_y_orig / train_y_element_sums_by_row[:, None]
	dev_y_for_model = dev_y_orig / dev_y_element_sums_by_row[:, None]
else:
	train_y_for_model = train_y_orig
	dev_y_for_model = dev_y_orig



# Set the input layer size and number of labels
input_layer_size = train_x_orig.shape[1];
num_labels = train_y_for_model.shape[1];

print ("Done! \n")

# Reshaping
print ("train_x_orig shape: " + str(train_x_orig.shape))
train_x_std, mean, scale = mean_normalize(train_x_orig)
train_x_flatten = train_x_std.reshape(train_x_std.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
train_x = train_x_flatten/1.
train_y = train_y_for_model.reshape((num_labels, train_y_for_model.shape[0]))

dev_x_std = (dev_x_orig - mean) / scale
dev_x_flatten = dev_x_std.reshape(dev_x_std.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
dev_x = dev_x_flatten/1.
dev_y = dev_y_for_model.reshape((num_labels, dev_y_for_model.shape[0]))

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]

print ("Number of training examples: " + str(m_train))
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))

## =========== Part 2: Tuning the Hypeparameters =============
#
#

### CONSTANTS DEFINING THE MODEL ####
n_x = input_layer_size
n_y = num_labels

if tuneControl == "ON":
	learnRate, layerArray, batchSize, regCoef = hyperOptMVR([n_x], n_y, 
             learnRate, layerArray, batchSize, regCoef, 
             RNGseed, 
             train_x_std, train_y_for_model, dev_x_std, dev_y_for_model,
             numEpochsOpt = 60, n_callsOpt = 30)


## =========== Part 3: Training the AI Model =============
#
#

# Model to train
def Scenxxx(input_shape, output_shape):
	"""
	Implementation of the Scenxxx.

	Arguments:
	input_shape -- shape of the images of the dataset

	Returns:
	model -- a Model() instance in Keras
	"""

	# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
	X_input = keras.layers.Input(input_shape)

	# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
	X = keras.layers.Flatten()(X_input)
	for l in range(0,layerArray.shape[0]):
		X = keras.layers.Dense(layerArray[l], activation=tf.nn.relu, kernel_initializer = glorot_uniform(seed=RNGseed), kernel_regularizer=regularizers.l2(regCoef)) (X)
		#X = keras.layers.Dropout(0.2) (X)

	if output_unit == "Linear":
		X = keras.layers.Dense(output_shape) (X)
	elif output_unit == "Softmax":
		X = keras.layers.Dense(output_shape, activation=tf.nn.softmax) (X)

	# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	model = keras.models.Model(inputs = X_input, outputs = X, name='Scenxxx')

	return model

model = Scenxxx([n_x], n_y)
optimizer = Adam(learning_rate=learnRate)
model.compile(optimizer = optimizer, loss = loss_function, metrics = [euclidean_sim4])
tic = time.time()
out_epoch = NPeriodicLogger(display=1) # show every epochs/20 results
history = model.fit(x = train_x_std, y = train_y_for_model, epochs =numEpochs, verbose=0, callbacks=[out_epoch], batch_size =batchSize , validation_data =(dev_x_std, dev_y_for_model))
toc = time.time()
print ("\n" + "   . Training time = " + str((toc - tic)) + "s")

## ================= Part 4: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

print ("\n  .. Prediction: ...Done!\n");
predictions_train = model.predict(train_x_std)
similarity = np.zeros((1,m_train))
y_mean = np.mean(train_y_for_model, axis=0).reshape(1,train_y_for_model.shape[1])

# other similarity metrics
cos_sim = np.zeros((1,m_train))
sim_e1 = np.zeros((1,m_train))
sim_e2 = np.zeros((1,m_train))
sim_e3 = np.zeros((1,m_train))
sim_e4 = np.zeros((1,m_train))
r2 = np.zeros((1,m_train))

for i in range(predictions_train.shape[0]):
	p_vector = predictions_train[i,:]
	y_vector = train_y_for_model[i,:]
	p_vector = p_vector.reshape((1, p_vector.shape[0]))
	y_vector = y_vector.reshape((1, y_vector.shape[0]))

	cos_sim[0,i] = np.dot(p_vector, y_vector.T)/(np.linalg.norm(p_vector)*np.linalg.norm(y_vector)) # cos_sim
	sim_e1[0,i] = (np.sum(y_vector, axis=1) - euclidean(p_vector, y_vector, np.sum(y_vector, axis=1))) / np.sum(y_vector, axis=1) # for euclidean_sim1
	sim_e2[0,i] = 1 - np.square(euclidean(p_vector, y_vector, np.sum(y_vector, axis=1))) / np.sum(np.square(y_vector), axis=1) # for euclidean_sim2
	sim_e3[0,i] = 1 - euclidean(p_vector, y_vector, np.sum(y_vector, axis=1)) / np.sqrt(np.sum(np.square(y_vector), axis=1)) # for euclidean_sim3
	r2[0,i] = 1 - np.sum(np.square(y_vector - p_vector), axis=1) / np.sum(np.square(y_vector - y_mean), axis=1) # for R2
	
	sim_e4[0,i] = sim_e1[0,i] * cos_sim[0,i] # for euclidean_sim4 = euclidean_sim1 * cos_sim

'''
scores = model.evaluate(train_x_std, train_y_for_model, verbose=0)
'''

print("   . Cosine Similarity: "  + str((np.sum(sim_e3)/m_train*100)))
print("   . Similarity E1:     "  + str((np.sum(sim_e1)/m_train*100)))
print("   . Similarity E2:     "  + str((np.sum(sim_e2)/m_train*100)))
print("   . Similarity E3:     "  + str((np.sum(sim_e3)/m_train*100)))
print("   . R-Squared:         "  + str((np.sum(r2)/m_train*100)) + "\n")

print("   . Similarity E4:     "  + str((np.sum(sim_e4)/m_train*100)) + "\n")

## ================ Part 5: Save Data, Parameters and Metadata ================

# Plot train loss, train acc, val loss and val acc against epochs passed
plt.figure()
plt.plot(history.history['euclidean_sim4'])
plt.plot(history.history['val_euclidean_sim4'])
plt.title('Model Euclidean Similarity')
plt.ylabel('Similarity')
plt.xlabel('Epoch')
plt.grid(which='both')
plt.legend(['training set similarity', 'dev set similarity'], loc='lower right')
# Make sure there exists a folder called output in the current directory
# or replace 'output' with whatever direcory you want to put in the plots
plt.savefig(outFileGraph)
plt.close()

# Write predictions and similarity to dataframe and .csv
'''
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
	predictions_train = predictions_train * train_y_element_sums_by_row[:, None]

df_predictions = pd.DataFrame(predictions_train, columns = prediction_columns)
similiarities_array = np.concatenate((cos_sim.T, sim_e1.T, sim_e2.T, sim_e3.T, r2.T, sim_e4.T), axis=1)
df_similarities = pd.DataFrame(similiarities_array, columns = ['Cos_Sim', 'ES1', 'ES2', 'ES3', 'R2', 'ES4'])
df_all_data = pd.concat([df_train_data, df_predictions, df_similarities], axis = 1)

df_all_data.to_csv(outFile, index=False)
'''

# serialize weights to HDF5
# Write to files
stdFile = h5py.File(stdFileName,"w")
hyperParFile = open(hyperParFileName,"w+")

# Write hyperparameters to file
hyperParFile.write("learnRate=%f\n" % (learnRate))
hyperParFile.close
hyperParFile = open(hyperParFileName,"a+")
hyperParFile.write("numEpochs=%d\n" % (numEpochs))
hyperParFile.write("batchSize=%d\n" % (batchSize))
hyperParFile.write("regCoef=%f\n" % (regCoef))
hyperParFile.write("RNGseed=%d\n" % (RNGseed))
hyperParFile.write("layerArray=" + str(layerArray))
hyperParFile.close

accFile = open(accFileName,"w+")
accFile.write("algorithm=MVR\n")
accFile.write("DNNtask=" + DNNtask + "\n")
accFile.write("featSet=" + featSet + "\n")
accFile.write("cos_sim=%2.2f\n" % (np.sum(cos_sim)/m_train*100))
accFile.write("sim_E1=%2.2f\n" % (np.sum(sim_e1)/m_train*100))
accFile.write("sim_E2=%2.2f\n" % (np.sum(sim_e2)/m_train*100))
accFile.write("sim_E3=%2.2f\n" % (np.sum(sim_e3)/m_train*100))
accFile.write("sim_R2=%2.2f\n" % (np.sum(r2)/m_train*100))
accFile.write("sim_E4=%2.2f\n" % (np.sum(sim_e4)/m_train*100))
accFile.close

# serialize model to JSON and weights to H5
model_json = model.to_json()
with open(jsonFile, "w") as json_file:
    json_file.write(model_json)

model.save_weights(paramFile)

stdFile.create_dataset('mean', data=mean)
stdFile.create_dataset('scale', data=scale)
stdFile.close()

print ("  .. Parameters and Data: ...Saved!")
