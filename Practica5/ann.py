import numpy as np

def cost(theta1, theta2, X, y, lambda_):
	  
	m = len(y)

	# Calcular los valores de las capas
	layerValues, weighted_inputs = forwardprop([theta1, theta2], X)
	
	# Calcular costes
	layerSum = (1 - y) * np.log(1 - layerValues[-1])
	sum = np.sum(y * np.log(layerValues[-1]) + layerSum)
	cost_J = (-1 / m) * sum

	return cost_J


def costL2(weights_list, X, y, lambda_):
	
	m = len(y)

	#Calcular coste
	cost_J = cost(weights_list[0], weights_list[1], X, y, lambda_)

	# Calcular el coste regularizado
	thetasSum = np.zeros(len(weights_list[0]))
	# Calcular regularizacion
	for i in range(2):
		thetasSum[i] = np.sum(weights_list[i][:, 1:]**2)
	reg = (lambda_ / (2 * m)) * np.sum(thetasSum)

	# Aplicar regularizacion
	cost_J = cost_J + reg

	return cost_J

def sigmoid(z):
	return 1/(1+np.exp(-z))


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


def backprop(theta1, theta2, X, y, lambda_):

	m = X.shape[0]

	# Crear array de los pesos
	weights = [theta1, theta2]

	# Calcular los valores de las capas
	layerValues, weighted_inputs = forwardprop(weights, X)

	# Diferencia entre la capa de salida y los valores reales de salida
	deltas = [layerValues[-1] - y]
	for i in reversed(range(1, len(weights))):
		# Calcular el gradiente
		sigGrad = sigmoid(weighted_inputs[i - 1]) * (1 - sigmoid(weighted_inputs[i - 1]))
		delta = np.dot(deltas[0], weights[i][:, 1:]) * sigGrad
		deltas.insert(0, delta)

	# Calcular gradientes sin regularizacion
	grad1 = np.dot(deltas[0].T, layerValues[0]) / m
	grad2 = np.dot(deltas[1].T, layerValues[1]) / m

	# Aplicar/Sumar regularizacion
	grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
	grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]

	# Calcular coste
	cost_J = costL2(weights, X, y, lambda_)

	return cost_J, grad1, grad2

# Modificar los parametros de las Thetas en cada iteracion
def iterateThetas(theta1, theta2, X, Y, iterations, myLambda, myAlpha):

	# Utilizar los gradientes para ir modificando los thetas en cada iteracion
	for iteration in range(iterations):
		# Calcular los gradientes 
		cost_J, g1, g2 = backprop(theta1, theta2, X, Y, myLambda)
		theta1 -= myAlpha * g1
		theta2 -= myAlpha * g2
	
	return theta1, theta2