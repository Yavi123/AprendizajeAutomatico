import numpy as np


def cost(thetas, X, y):
	"""
	Compute the cost for a neural network.

	Parameters
	----------
	theta_list : list of array_like
		List of weight matrices for each layer in the neural network.
		Each matrix has shape (number of units in current layer x number of units in previous layer + 1).
	X : array_like
		Input data with shape (number of examples x number of features).
	y : array_like
		1-hot encoding of labels for the input, having shape 
		(number of examples x number of classes).

	Returns
	-------
	J : float
		The computed value for the cost function.
	"""
	
	m = len(y)

	# Calculamos los valores producidos por el forward propagation a traves de los pesos especificados
	layers, zs = forwardprop(thetas, X)

	# Usar la ultima capa para calcular el coste
	exitLayer = layers[-1]
	
	# Aplicamos la formula
	J = (-1 / m) * np.sum(y * np.log(exitLayer) + (1 - y) * np.log(1 - exitLayer))
	
	return J


def costL2(thetas, X, y, lambda_):
	
	m = len(y)

	#Calcular coste
	cost_J = cost(thetas, X, y)

	# Calcular el coste regularizado
	thetasSum = np.zeros(len(thetas[0]))
	# Calcular regularizacion
	for i in range(len(thetas)):
		thetasSum[i] = np.sum(thetas[i][:, 1:]**2)
	reg = (lambda_ / (2 * m)) * np.sum(thetasSum)

	# Aplicar regularizacion
	cost_J = cost_J + reg

	return cost_J

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_gradient(z):
	return sigmoid(z) * (1 - sigmoid(z))


def forwardprop(weights, input_data):
	# Numero de casos de entrenamiento
	num_examples = input_data.shape[0]
	# Agregar sesgo
	input_data = np.hstack([np.ones((num_examples, 1)), input_data])

	neuronOutput = [input_data]
	# Lista para almacenar las entradas
	weighted_inputs = list()

	# Recorrer los pesos
	for weight in weights:
		 # Calcular la entrada ponderada
		weighted_input = np.dot(neuronOutput[-1], weight.T)

		# Calcular sigmoide
		activation_output = sigmoid(weighted_input)

		# Comprobar que no es la ultima capa
		if weight is not weights[-1]:
			# Añadir sesgo
			activation_output = np.hstack([np.ones((num_examples, 1)), activation_output])

		# Almacenar datos en las listas
		weighted_inputs.append(weighted_input)
		neuronOutput.append(activation_output)

	return neuronOutput, weighted_inputs


def backprop(thetas, X, y, lambda_):
	"""
	Compute cost and gradient for 2-layer neural network. 

	Parameters
	----------
	theta_list : list of array_like
		List of weight matrices for each layer in the neural network.
		Each matrix has shape (number of units in current layer x number of units in previous layer + 1).

	X : array_like
		The inputs having shape (number of examples x number of dimensions).

	y : array_like
		1-hot encoding of labels for the input, having shape 
		(number of examples x number of labels).

	lambda_ : float
		The regularization parameter. 

	Returns
	-------
	J : float
		The computed value for the cost function. 

	grad1 : array_like
		Gradient of the cost function with respect to weights
		for the first layer in the neural network, theta1.
		It has shape (2nd hidden layer size x input size + 1)

	grad2 : array_like
		Gradient of the cost function with respect to weights
		for the second layer in the neural network, theta2.
		It has shape (output layer size x 2nd hidden layer size + 1)

	"""

	m = X.shape[0]

	# Crear predicciones
	allLayers, zs = forwardprop(thetas, X)
	lastLayer = allLayers[-1]



	# Calcular los deltas para las capas ocultas
	lastLayerDelta = lastLayer - y
	allDeltas = [lastLayerDelta]
	for i in range(len(thetas) - 1, 0, -1):
		previous_delta = allDeltas[0]
		thetaWithoutBias = thetas[i][:, 1:]
		zPrevLayer = zs[i - 1]

		weighted_delta = np.dot(previous_delta, thetaWithoutBias)
		sigmoid_gradient_z = sigmoid_gradient(zPrevLayer)

		thisDelta = weighted_delta * sigmoid_gradient_z
		allDeltas.insert(0, thisDelta)


	# Referencia a las capas ocultas
	hiddenLayers = allLayers[:-1]

	# Calcular los gradientes
	g = []
	for this_delta, a in zip(allDeltas, hiddenLayers):
		gradient = np.dot(this_delta.T, a) / m
		g.append(gradient)


	# Modificar los gradientes
	for i in range(len(g)):
		regularizationTerm = (lambda_ / m) * thetas[i][:, 1:]
		g[i][:, 1:] += regularizationTerm


	# Calcular coste
	totalCost = costL2(thetas, X, y, lambda_)

	return totalCost, g


def iterateThetas(weights, X, Y, iterations, myLambda, myAlpha):
	for iteration in range(iterations):
		cost_J, grads = backprop(weights, X, Y, myLambda)
		for i in range(len(weights)):
			weights[i] -= grads[i] * myAlpha
	
	return weights