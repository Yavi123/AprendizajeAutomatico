import numpy as np
import copy
import math


# Normaliza los valores
def zscore_normalize_features(x):

    X_norm = np.zeros((np.size(x[:,:1]), np.size(x[1])), dtype='float32')
    mu =    np.zeros(np.size(x[1]), dtype='float32')
    sigma = np.zeros(np.size(x[1]), dtype='float32')

    for i in range(0, np.size(x[1])):
        mu[i] = np.mean(x[:,i])
        sigma[i] = np.std(x[:,i])
        X_norm[:,i] = (x[:,i] - mu[i]) / sigma[i]

    return (X_norm, mu,sigma)


def compute_cost(x, y, w, b):

    # n es el numero de variables en X
    y_pred = np.dot(x, w) + b  # Predicciones del modelo, de tamaño (m, 1), un array de tamaño m

    # m es el numero de ejemplos
    m = len(y)
    # aplicar formula
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

    return cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    """

    # m es el numero de ejemplos
    m = len(y)

    # predictions devuelve uun array de tamaño m
    predictions = np.dot(x, w) + b

    # dj_dw devuelve uun array de tamaño n
    # x.T tamaño (n, m)  &  predictions tamaño (m)
    # Al hacer una multiplicacion vectorial, se queda en un array de tamaño (n,1 = n)
    dj_dw = (1 / m) * np.dot(x.T, (predictions - y))

    # dj_db devuelve un numero escalar
    dj_db = (1 / m) * np.sum(predictions - y)
    return dj_db, dj_dw



def gradient_descent(x, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):

    """
	Performs batch gradient descent to learn theta. Updates theta by taking 
	num_iters gradient steps with learning rate alpha

	Args:
	  X : (array_like Shape (m,n)    matrix of examples 
	  y : (array_like Shape (m,))    target value of each example
	  w_in : (array_like Shape (n,)) Initial values of parameters of the model
	  b_in : (scalar)                Initial value of parameter of the model
	  cost_function: function to compute cost
	  gradient_function: function to compute the gradient
	  alpha : (float) Learning rate
	  num_iters : (int) number of iterations to run gradient descent
	Returns
	  w : (array_like Shape (n,)) Updated values of parameters of the model
	      after running gradient descent
	  b : (scalar)                Updated value of parameter of the model 
	      after running gradient descent
	  J_history : (ndarray): Shape (num_iters,) J at each iteration,
    	  primarily for graphing later
    """

    # w es de tamaño n, numero de variables en X
    w = w_in
    # b es un numero escalar
    b = b_in

    # Inicializar el array que almacenara los costes que se vayan calculando
    # Todos sus elementos empiezan siendo 0
    J_history = np.zeros(num_iters)

    # Modificar w, b en cada iteracion utilizando la gradiente
    for i in range(num_iters):

        # Guardar el coste de esta iteracion
        J_history[i] = cost_function(x, y, w, b)

        b_gradient,w_gradient  = gradient_function(x,y,w,b)

        w -= alpha * w_gradient
        b -= alpha * b_gradient
	
    return w, b, J_history