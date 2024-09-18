import numpy as np
#
import h5py
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import glorot_uniform
# import scikit-optimize
import skopt
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
# from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from dnn_app_utils import *
# Set some random seeds
np.random.seed(1)
tf.random.set_seed(2)


def func_layerArray(num_dense_layers, 
                    num_dense_nodes_0, num_dense_nodes_1, num_dense_nodes_2, num_dense_nodes_3, num_dense_nodes_4, 
                    num_dense_nodes_5, num_dense_nodes_6, num_dense_nodes_7, num_dense_nodes_8, num_dense_nodes_9):
	layerArray = np.zeros(num_dense_layers)
	layerArray[0] = num_dense_nodes_0
	if num_dense_layers > 1: layerArray[1] = num_dense_nodes_1
	if num_dense_layers > 2: layerArray[2] = num_dense_nodes_2
	if num_dense_layers > 3: layerArray[3] = num_dense_nodes_3
	if num_dense_layers > 4: layerArray[4] = num_dense_nodes_4
	if num_dense_layers > 5: layerArray[5] = num_dense_nodes_5
	if num_dense_layers > 6: layerArray[6] = num_dense_nodes_6
	if num_dense_layers > 7: layerArray[7] = num_dense_nodes_7
	if num_dense_layers > 8: layerArray[8] = num_dense_nodes_8
	if num_dense_layers > 9: layerArray[9] = num_dense_nodes_9
	return layerArray

# This model should be the same as Scenxxx
def create_hypopt_model(input_shape, output_shape, learnRate, layerArray, batchSize, regCoef):
	# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
	X_input = keras.layers.Input(input_shape)
	# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
	X = keras.layers.Flatten()(X_input)
	
	#create a loop making a new dense layer for the amount passed to this model.
	#naming the layers helps avoid tensorflow error deep in the stack trace.
	for i in range(0,layerArray.shape[0]):
		name = 'layer_dense_{0}'.format(i+1)
		X = keras.layers.Dense(layerArray[i], activation=tf.nn.relu, kernel_initializer = glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(regCoef), name=name) (X)
	#add our classification layer.
	X = keras.layers.Dense(output_shape) (X)
	
	# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	hypopt_model = keras.models.Model(inputs = X_input, outputs = X, name='create_hypopt_model')
	
	#setup our optimizer and compile
	adam = Adam(learning_rate=learnRate)
	hypopt_model.compile(optimizer = adam, loss = "mean_absolute_error", metrics = [euclidean_sim4])
	return hypopt_model

def hyperOptMVR(input_shape, output_shape, 
             learnRate, layerArray, batchSize, regCoef, 
             RNGseed, 
             train_x_std, train_y_orig, dev_x_std, dev_y_orig,
             numEpochsOpt, n_callsOpt):

	# Set the space of the hyperparameters (dimensions, prior, etc.)
	dim_learnRate = Real(low=5e-5, high=5e-3, prior='log-uniform', name='learnRate')
	dim_num_dense_layers = Integer(low=3, high=10, name='num_dense_layers')
	dim_num_dense_nodes_0 = Integer(low=100, high=1000, name='num_dense_nodes_0')
	dim_num_dense_nodes_1 = Integer(low=100, high=1000, name='num_dense_nodes_1')
	dim_num_dense_nodes_2 = Integer(low=100, high=1000, name='num_dense_nodes_2')
	dim_num_dense_nodes_3 = Integer(low=100, high=1000, name='num_dense_nodes_3')
	dim_num_dense_nodes_4 = Integer(low=100, high=1000, name='num_dense_nodes_4')
	dim_num_dense_nodes_5 = Integer(low=100, high=1000, name='num_dense_nodes_5')
	dim_num_dense_nodes_6 = Integer(low=100, high=1000, name='num_dense_nodes_6')
	dim_num_dense_nodes_7 = Integer(low=100, high=1000, name='num_dense_nodes_7')
	dim_num_dense_nodes_8 = Integer(low=100, high=1000, name='num_dense_nodes_8')
	dim_num_dense_nodes_9 = Integer(low=100, high=1000, name='num_dense_nodes_9')
	dim_batchSize = Integer(low=128, high=4096, name='batchSize')
	dim_regCoef = Real(low=1e-7, high=1e-5, prior='log-uniform', name='regCoef')

	dimensions = [dim_learnRate,
				  dim_num_dense_layers,
				  dim_num_dense_nodes_0,
				  dim_num_dense_nodes_1,
				  dim_num_dense_nodes_2,
				  dim_num_dense_nodes_3,
				  dim_num_dense_nodes_4,
				  dim_num_dense_nodes_5,
				  dim_num_dense_nodes_6,
				  dim_num_dense_nodes_7,
				  dim_num_dense_nodes_8,
				  dim_num_dense_nodes_9,
				  dim_batchSize,
				  dim_regCoef]
	@use_named_args(dimensions=dimensions)
	# The fitness function for hyperparameter optimization
	def fitness(learnRate, num_dense_layers,
            num_dense_nodes_0, num_dense_nodes_1, num_dense_nodes_2, num_dense_nodes_3, num_dense_nodes_4, 
            num_dense_nodes_5, num_dense_nodes_6, num_dense_nodes_7, num_dense_nodes_8, num_dense_nodes_9, 
            batchSize, regCoef):

		layerArray = func_layerArray (num_dense_layers, 
                                  num_dense_nodes_0, num_dense_nodes_1, num_dense_nodes_2, num_dense_nodes_3, num_dense_nodes_4, 
                                  num_dense_nodes_5, num_dense_nodes_6, num_dense_nodes_7, num_dense_nodes_8, num_dense_nodes_9)
		
		print()
		print ("  .. Learning rate              = " + str(learnRate))
		print ("  .. Mini-batches size          = " + str(batchSize))
		print ("  .. Regularization coefficient = " + str(regCoef))
		print ("  .. layer Array                = " + str(layerArray[:]))
		print()

		hypopt_model = create_hypopt_model(input_shape, output_shape, learnRate=learnRate,
                                layerArray=layerArray,
                                batchSize=batchSize,
                                regCoef=regCoef
                                )
		

		#named blackbox becuase it represents the structure
		blackbox = hypopt_model.fit(x = train_x_std, y = train_y_orig, epochs = numEpochsOpt, verbose=1, batch_size =batchSize , validation_data =(dev_x_std, dev_y_orig))
		#return the validation similarity for the last epoch.
		similarity = blackbox.history['val_euclidean_sim4'][-1]

		# Print the classification similarity.
		print()
		print ("  .. Learning rate              = " + str(learnRate))
		print ("  .. Mini-batches size          = " + str(batchSize))
		print ("  .. Regularization coefficient = " + str(regCoef))
		print ("  .. layer Array                = " + str(layerArray[:]))
		print()
		print (".... Similarity: {0:.2%}".format(similarity))
		print()


		# Delete the Keras model with these hyper-parameters from memory.
		del hypopt_model
		
		# Clear the Keras session, otherwise it will keep adding new
		# models to the same TensorFlow graph each time we create
		# a model with a different set of hyper-parameters.
		K.clear_session()
		ops.reset_default_graph()
		
		# the optimizer aims for the lowest score, so we return our negative similarity
		return -similarity
	# Copy our default value for the hidden nodes to the usual 10 hidden layer arrays, the missing values would be 0's
	layerArrayType = np.zeros(10)
	for n in range(0,layerArray.shape[0]):
		layerArrayType[n] = layerArray[n]

	default_parameters = [learnRate, layerArray.shape[0], 
                          max(layerArrayType[0],100), max(layerArrayType[1],100), max(layerArrayType[2],100), max(layerArrayType[3],100), max(layerArrayType[4],100), 
                          max(layerArrayType[5],100), max(layerArrayType[6],100), max(layerArrayType[7],100), max(layerArrayType[8],100), max(layerArrayType[9],100), 
                          batchSize, regCoef]

	# The actual hyperparameters optimization
	print ("  .. Optimizing Hyperparameters ...             ", end = '')
	
	gp_result = gp_minimize(func=fitness,
							  dimensions=dimensions,
							  n_calls=n_callsOpt,
							  n_jobs=-1,
							  verbose=True,
							  x0=default_parameters)
	print (" ...Done!\n");
	print ("  .. Learning rate              = " + str(gp_result.x[0]))
	print ("  .. Mini-batches size          = " + str(gp_result.x[12]))
	print ("  .. Regularization coefficient = " + str(gp_result.x[13]))
	layerArray = func_layerArray(gp_result.x[1], 
								 gp_result.x[2], gp_result.x[3], gp_result.x[4], gp_result.x[5], gp_result.x[6], 
								 gp_result.x[7], gp_result.x[8], gp_result.x[9], gp_result.x[10], gp_result.x[11])
	print ("  .. layer Array                = " + str(layerArray[:]))
	
	return learnRate, layerArray, batchSize, regCoef