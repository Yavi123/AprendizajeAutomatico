import numpy as np

#########################################################################
# NN
#
def predict(theta1, theta2, X):
	"""
	Predict the label of an input given a trained neural network.

	Parameters
	----------
	theta1 : array_like
		Weights for the first layer in the neural network.
		It has shape (2nd hidden layer size x input size)

	theta2 : array_like
		Weights for the second layer in the neural network.
		It has shape (output layer size x 2nd hidden layer size)

	X : array_like
		The image inputs having shape (number of examples x image dimensions).

	Return 
	------
	p : array_like
		Predictions vector containing the predicted label for each example.
		It has a length equal to the number of examples.
	"""

	# Agrega un sesgo a los datos de entrada
	X = np.column_stack((np.ones((X.shape[0], 1)), X))  # Añadir una columna de unos

	# Calcula la capa oculta
	z2 = np.dot(X, theta1.T)
	# Ajusta los valores de la capa oculta
	a2 = sigmoid(z2)

	# Sumar un sesgo a la capa oculta de neuronas
	a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

	# Calcular los valores de la capa de salida
	z3 = np.dot(a2, theta2.T)
	# Ajustar los valores de salida
	a3 = sigmoid(z3)

	#Construir un array con los los valores mas probable de cada columna
	p = np.argmax(a3, axis=1)

	return p

def sigmoid(z):
	return 1 / (1 + np.exp(-z))