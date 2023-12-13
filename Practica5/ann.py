import numpy as np

def cost(theta1, theta2, X, y, lambda_):
	"""
	Compute cost for 2-layer neural network. 

	Parameters
	----------
	theta1 : array_like
		Weights for the first layer in the neural network.
		It has shape (2nd hidden layer size x input size + 1)

	theta2: array_like
		Weights for the second layer in the neural network. 
		It has shape (output layer size x 2nd hidden layer size + 1)

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

	"""
	  
	m = len(y)
	
	activations, zs = forward([theta1, theta2], X)
	predictions = activations[-1]
	
	J = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


	return J

def forward(theta_list, input):
	m = input.shape[0]
	input = np.hstack([np.ones((m, 1)), input])
	activationValues = [input]
	zs = []

	for i in range(len(theta_list)):
		z = np.dot(activationValues[-1], theta_list[i].T)

		layerValues = 1 / (1 + np.exp(-zs[i]))

		if (i < len(theta_list) - 1):
			layerValues = np.hstack([np.ones((m, 1)), layerValues])

		zs.append(z)
		activationValues.append(layerValues)


	return activationValues, zs

def iterateThetas(theta1, theta2, X, Y, iterations, myLambda, myAlpha):

	# Utilizar los gradientes para ir modificando los thetas en cada iteracion
	for iteration in range(iterations):
		# Calcular los gradientes 
		cost, g1, g2 = backprop(theta1, theta2, X, Y, myLambda)
		theta1 -= myAlpha * g1
		theta2 -= myAlpha * g2
	
	return theta1, theta2

def costL2(theta_list, X, y, lambda_):
	
	"""
	Compute cost for 2-layer neural network. 

	Parameters
	----------
	theta1 : array_like
		Weights for the first layer in the neural network.
		It has shape (2nd hidden layer size x input size + 1)

	theta2: array_like
		Weights for the second layer in the neural network. 
		It has shape (output layer size x 2nd hidden layer size + 1)

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

	"""

	m = len(y)
	activations, zs = forward(theta_list, X)
	predictions = activations[-1]

	J = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

	reg_term = 0
	for theta in theta_list:
		reg_term += np.sum(theta[:, 1:]**2)
	reg_term *= (lambda_ / (2 * m))
	J += reg_term

	return J

def sig(z):
	return 1/(1+np.exp(-z))

def backprop(theta1, theta2, X, y, lambda_):
	"""
	Compute cost and gradient for 2-layer neural network.
	
	Parameters
	----------
	theta1 : array_like
		Weights for the first layer in the neural network.
		It has shape (2nd hidden layer size x input size + 1)
	
	theta2 : array_like
		Weights for the second layer in the neural network.
		It has shape (output layer size x 2nd hidden layer size + 1)
	
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
	theta_list = [theta1, theta2]
	m = X.shape[0]

	activations, zs = forward(theta_list, X)
	a_last = activations[-1]

	J = costL2(theta_list, X, y, lambda_)

	deltas = [a_last - y]
	for i in range(len(theta_list) - 1, 0, -1):
		# Calcular el gradiente
		sigGrad = sig(zs[i - 1]) * (1 - sig(zs[i - 1]))
		delta = np.dot(deltas[0], theta_list[i][:, 1:]) * sigGrad
		deltas.insert(0, delta)

	grad1 = np.dot(deltas[0].T, activations[0]) / m
	grad2 = np.dot(deltas[1].T, activations[1]) / m

	grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
	grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]

	return J, grad1, grad2