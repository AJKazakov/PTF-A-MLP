import numpy as np
import h5py
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist
import tensorflow.keras.backend as K

def mean_normalize(X):
	scaler = preprocessing.StandardScaler()
	std_scale = scaler.fit(X)
	X_std = std_scale.transform(X)
	mean = scaler.mean_
	scale = scaler.scale_
	return X_std, mean, scale

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ

def softmax(Z):
	expZ = np.exp(Z)
	A = expZ / expZ.sum(axis=0, keepdims=True)
	assert(A.shape == Z.shape)
	cache = Z
	return A, cache

def softmax_backward(dA, cache):
	Z = cache
	dZ = dA
	assert(dZ.shape == Z.shape)
	return dZ

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0], parameters["b" + str(l+1)].shape[1]))
        s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0], parameters["W" + str(l+1)].shape[1]))
        s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0], parameters["b" + str(l+1)].shape[1]))
    ### END CODE HERE ###

    return v, s

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	A -- the output of the activation function, also called the post-activation value
	cache -- a python dictionary containing "linear_cache" and "activation_cache";
			stored for computing the backward pass efficiently
	"""

	if activation == "sigmoid":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	elif activation == "softmax":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = softmax(Z)

	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
	"""
	Implement the cost function defined by equation (7).

	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""

	m = Y.shape[1]

	# Compute loss from aL and y.
	cost = -(1./m) * np.sum(Y * np.log(AL))
	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())

	return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.

	Arguments:
	dA -- post-activation gradient for current layer l
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	linear_cache, activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	elif activation == "softmax":
		dZ = softmax_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = AL - Y

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax")

    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-np.power(beta1,t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-np.power(beta1,t))
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*grads["dW" + str(l+1)]*grads["dW" + str(l+1)]
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*grads["db" + str(l+1)]*grads["db" + str(l+1)]
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-np.power(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-np.power(beta2,t))
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
        ### END CODE HERE ###

    return parameters, v, s

def predict(X, Y, parameters):
	"""
	This function is used to predict the results of a  L-layer neural network.

	Arguments:
	X -- data set of examples you would like to label
	Y -- true "label" vector (NOT the one-hot representation)
	parameters -- parameters of the trained model

	Returns:
	p -- predictions for the given dataset X
	"""

	m = X.shape[1]
	n = len(parameters) // 2 # number of layers in the neural network
	p = np.zeros((1,m))

	# Forward propagation
	probas, caches = L_model_forward(X, parameters)

	# convert probas to argmax predictions
	p = np.argmax(probas, axis=0)

    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
	#accuracy
	accuracy = np.sum(p == Y)/m

	print("   . Accuracy: "  + str((accuracy*100)) + "\n")

	return p, accuracy

def accuracy_check(probas, Y):
	"""
	This function is used to predict the results of a  L-layer neural network.

	Arguments:
	Y -- true "label" vector (NOT the one-hot representation)
	parameters -- parameters of the trained model

	Returns:
	accuracy -- accuracy evaluation
	"""

	m = Y.shape[0]
	p = np.zeros((1,m))
	p = np.argmax(probas, axis=1)
	accuracy = np.sum(p == Y)/m

	print("   . Accuracy: "  + str((accuracy*100)) + "\n")

	return p, accuracy

def manhattan(y_vector, p_vector):
	distance = cdist(p_vector, y_vector, metric='cityblock')
	if distance > 1:
		distance = 1
	return distance

def euclidean(y_vector, p_vector, y_sum):
	distance = np.linalg.norm(p_vector-y_vector)
	if distance > y_sum:
		distance = y_sum
	return distance

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def euclidean_sim(y_true, y_pred):
	"""
	Euclidean distance loss
	https://en.wikipedia.org/wiki/Euclidean_distance
	:param y_true: TensorFlow/Theano tensor
	:param y_pred: TensorFlow/Theano tensor of the same shape as y_true
	:return: float
	"""
	euclidean_distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
	sum_outputs = K.sum(y_true, axis=1)
	difference = sum_outputs - euclidean_distance
	result = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1])([difference, sum_outputs])
	return result

def euclidean_sim2(y_true, y_pred):
	"""
	Euclidean distance loss
	https://en.wikipedia.org/wiki/Euclidean_distance
	:param y_true: TensorFlow/Theano tensor
	:param y_pred: TensorFlow/Theano tensor of the same shape as y_true
	:return: float
	"""
	euclidean_distance = K.sum(K.square(y_pred - y_true), axis=-1)
	sum_outputs = K.sum(K.square(y_true), axis=1)
	difference = sum_outputs - euclidean_distance
	result = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1])([difference, sum_outputs])
	return result

def euclidean_sim3(y_true, y_pred):
	"""
	Euclidean distance loss
	https://en.wikipedia.org/wiki/Euclidean_distance
	:param y_true: TensorFlow/Theano tensor
	:param y_pred: TensorFlow/Theano tensor of the same shape as y_true
	:return: float
	"""
	euclidean_distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
	sum_outputs = K.sqrt(K.sum(K.square(y_true), axis=1))
	difference = sum_outputs - euclidean_distance
	result = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1])([difference, sum_outputs])
	return result

def euclidean_sim4(y_true, y_pred):
    """
    Euclidean distance loss number 4 = euclidean similarity * cosine similarity
    Note: See documentation that cosine similarity is from -1 to 1,
    but -1 is just for convenience to be used as a loss function => let's negate it to get it as percentage
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    euclidean_distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    sum_outputs = K.sum(y_true, axis=1)
    difference = sum_outputs - euclidean_distance
    cos_sim = keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
    result = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1] * inputs[2])([difference, sum_outputs, -cos_sim])
    return result

def similarity_check(p, Y, classRatioMatrix):
	"""
	This function is used to give the similarity of the NN's predictions to the true labels.

	Arguments:
	Y -- true "label" vector (NOT the one-hot representation)
	parameters -- parameters of the trained model

	Returns:
	accuracy -- accuracy evaluation
	"""

	m = Y.shape[0]
	similarity = np.zeros((1,m))
	for i in range(Y.shape[0]):
		p_vector = classRatioMatrix[int(p[i])]
		y_vector = classRatioMatrix[int(Y[i])]
		p_vector = p_vector.reshape((1, p_vector.shape[0]))
		y_vector = y_vector.reshape((1, y_vector.shape[0]))
		similarity[0,i] = 1 - euclidean(p_vector, y_vector)

	print("   . Similarity: "  + str((np.sum(similarity)/m*100)) + "\n")

	return similarity

def class_error_analysis(p, Y, n_y):
	"""
	This function is used to make a basic class error analysis on the predictions of the neural network.

	Arguments:
	p -- predictions for the given dataset X
	Y -- true "label" vector (NOT the one-hot representation)
	n_y -- size of the output layer

	Returns:
	class_sums -- class error evaluation matrix (where each row represents how often one class is classified as the column number,
				 therefore the diagonal of the matrix is the positive prediction for the classes)
	"""

	class_sums = np.zeros((n_y,n_y), dtype=int)
	class_err = np.zeros((n_y,4))

	# Class predictions collection
	for i in range(0, n_y):
		for j in range(0, n_y):
			class_sums[i, j] = np.sum(np.logical_and(Y == i, p == j))

	# Class error evaluation
	np.seterr(divide='ignore', invalid='ignore')
	mistaken_to_other = (np.sum(class_sums, axis = 1).T - np.diag(class_sums))
	other_is_mistaken = (np.sum(class_sums, axis = 0) - np.diag(class_sums))
	class_err[:, 0] = np.divide(mistaken_to_other, np.sum(class_sums, axis = 1).T, where=np.sum(class_sums, axis = 1).T!=0) *100
	class_err[:, 1] = np.divide(mistaken_to_other, np.sum(class_sums), where=np.sum(class_sums)!=0) *100
	class_err[:, 2] = np.divide(other_is_mistaken, np.sum(class_sums, axis = 0), where=np.sum(class_sums, axis = 0)!=0) *100
	class_err[:, 3] = np.divide(other_is_mistaken, np.sum(class_sums), where=np.sum(class_sums)!=0) *100

	print("   . Classification collection array: \n"  + str((class_sums)) + "\n")
	np.set_printoptions(precision=2)
	print("   . Classification error array: \n"  + str((class_err)) + "\n")

	return class_sums, class_err


def class_error_analysis_keras(p, Y, n_y):
	"""
	This function is used to make a basic class error analysis on the predictions of the neural network.

	Arguments:
	p -- predictions for the given dataset X
	Y -- true "label" vector (NOT the one-hot representation)
	n_y -- size of the output layer

	Returns:
	class_sums -- class error evaluation matrix (where each row represents how often one class is classified as the column number,
				 therefore the diagonal of the matrix is the positive prediction for the classes)
	"""

	class_sums = np.zeros((n_y,n_y), dtype=int)
	class_err = np.zeros((n_y,4))

	# Class predictions collection
	for i in range(0, n_y):
		for j in range(0, n_y):
			class_sums[i, j] = np.sum(np.logical_and(Y == i, p == j))

	# Class error evaluation
	np.seterr(divide='ignore', invalid='ignore')
	mistaken_to_other = (np.sum(class_sums, axis = 1).T - np.diag(class_sums))
	other_is_mistaken = (np.sum(class_sums, axis = 0) - np.diag(class_sums))
	class_err[:, 0] = np.divide(mistaken_to_other, np.sum(class_sums, axis = 1).T, where=np.sum(class_sums, axis = 1).T!=0) *100
	class_err[:, 1] = np.divide(mistaken_to_other, np.sum(class_sums), where=np.sum(class_sums)!=0) *100
	class_err[:, 2] = np.divide(other_is_mistaken, np.sum(class_sums, axis = 0), where=np.sum(class_sums, axis = 0)!=0) *100
	class_err[:, 3] = np.divide(other_is_mistaken, np.sum(class_sums), where=np.sum(class_sums)!=0) *100

	return class_sums, class_err

class NPeriodicLogger(keras.callbacks.Callback):
	"""
	A Logger that log average performance per `display` steps.
	ref: https://github.com/keras-team/keras/issues/2850
	"""
	def __init__(self, display):
		self.step = 0
		self.display = display
		self.metric_cache = {}

	def on_epoch_end(self, epoch, logs={}):
		self.step += 1
		for k in logs.keys():
			self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
		if self.step % self.display == 0 or self.step ==1:
			metrics_log = ''
			for (k, v) in self.metric_cache.items():
				if self.step ==1:
					val = v
				else:
					val = v / self.display
				if abs(val) > 1e-3:
					metrics_log += ' - %s: %.4f' % (k, val)
				else:
					metrics_log += ' - %s: %.4e' % (k, val)
			print('Epoch: {}/{} ... {}'.format(self.step,
											self.params['epochs'],
											metrics_log))
			self.metric_cache.clear()

