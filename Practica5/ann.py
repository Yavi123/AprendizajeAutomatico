import numpy as np

# def cost(theta1, theta2, X, y, lambda_):
	  
# 	m = len(y)

# 	# Calcular los valores de las capas
# 	layerValues, weighted_inputs = forwardprop([theta1, theta2], X)
	
# 	# Calcular costes
# 	layerSum = (1 - y) * np.log(1 - layerValues[-1])
# 	sum = np.sum(y * np.log(layerValues[-1]) + layerSum)
# 	cost_J = (-1 / m) * sum

# 	return cost_J

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
    activations, zs = forwardprop(thetas, X)
    predictions = activations[-1]
    
    J = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
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


# def backprop(theta1, theta2, X, y, lambda_):

# 	m = X.shape[0]

# 	# Crear array de los pesos
# 	weights = [theta1, theta2]

# 	# Calcular los valores de las capas
# 	layerValues, weighted_inputs = forwardprop(weights, X)

# 	# Calcular los Delta
# 	deltaA3 = layerValues[-1] - y
# 	deltaA2 = np.dot(deltaA3, theta2[:, 1:]) * (sigmoid(weighted_inputs[-2]) * (1 - sigmoid(weighted_inputs[-2])))

# 	# Calcular los gradientes dependiendo de los delta calculados anteriormente
# 	grad2 = (1/m) * np.dot(deltaA3.T, layerValues[-2])
# 	grad1 = (1/m) * np.dot(deltaA2.T, layerValues[-3])

# 	# # Aplicar/Sumar regularizacion
# 	grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
# 	grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]

# 	# Calcular coste
# 	cost_J = costL2(weights, X, y, lambda_)

# 	return cost_J, grad1, grad2

# # Modificar los parametros de las Thetas en cada iteracion
# def iterateThetas(theta1, theta2, X, Y, iterations, myLambda, myAlpha):

# 	# Utilizar los gradientes para ir modificando los thetas en cada iteracion
# 	for iteration in range(iterations):
# 		# Calcular los gradientes 
# 		cost_J, g1, g2 = backprop(theta1, theta2, X, Y, myLambda)
# 		theta1 -= g1 * myAlpha
# 		theta2 -= g2 * myAlpha
	
# 	return theta1, theta2



# def backprop(weights, X, y, lambda_):
#     m = X.shape[0]
#     layerValues, weighted_inputs = forwardprop(weights, X)
    
#     deltas = [layerValues[-1] - y]
#     for i in range(len(weights) - 1, 0, -1):
#         delta = np.dot(deltas[0], weights[i][:, 1:]) * (sigmoid(weighted_inputs[i - 1]) * (1 - sigmoid(weighted_inputs[i - 1])))
#         deltas.insert(0, delta)
    
#     grads = []
#     for i in range(len(weights) - 1, -1, -1):
#         grad = (1/m) * np.dot(deltas[i].T, layerValues[i])
#         grad[:, 1:] += (lambda_ / m) * weights[i][:, 1:]
#         grads.append(grad)

#     cost_J = costL2(weights, X, y, lambda_)
    
#     return cost_J, grads

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

    allLayerValues, zs = forwardprop(thetas, X)
    lastLayer = allLayerValues[-1]

    totalCost = costL2(thetas, X, y, lambda_)

    deltas = [lastLayer - y]
    for i in range(len(thetas) - 1, 0, -1):
        delta = np.dot(deltas[0], thetas[i][:, 1:]) * sigmoid_gradient(zs[i - 1])
        deltas.insert(0, delta)

    grads = [np.dot(d.T, a) / m for d, a in zip(deltas, allLayerValues[:-1])]

    for i in range(len(grads)):
        grads[i][:, 1:] += (lambda_ / m) * thetas[i][:, 1:]

    return totalCost, grads

def iterateThetas(weights, X, Y, iterations, myLambda, myAlpha):
    for iteration in range(iterations):
        cost_J, grads = backprop(weights, X, Y, myLambda)
        for i in range(len(weights)):
            weights[i] -= grads[i] * myAlpha
    
    return weights