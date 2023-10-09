import numpy as np
import copy
import math


#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):    
    m = np.size(x)

    res = 0

    for i in range(0, m):
        res += (f_wb(w, b, x[i]) - y[i]) ** 2

    total_cost = res / (2 * m)

    return total_cost

def f_wb(w, b, x):
    return w * x + b


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
    m = len(x)
    predictions = np.dot(x, w) + b
    error = predictions - y

    dj_dw = (1 / m) * np.dot(error, x)

    dj_db = (1 / m) * np.sum(error)

    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    
    m = len(y)  # Número de ejemplos de entrenamiento
    w = w_in  # Inicialización de w
    b = b_in  # Inicialización de b
    J_history = np.zeros(num_iters)  # Inicialización del historial de costos
    
    for i in range(num_iters):
        # Calcular el costo actual
        # Almacenar el costo actual en el historial de costos
        J_history[i] = cost_function(x, y, w, b)
        
        # Calcular el gradiente actual
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        # Actualizar los parámetros w y b usando el descenso de gradiente
        w -= alpha * dj_dw
        b -= alpha * dj_db
    
    return w, b, J_history