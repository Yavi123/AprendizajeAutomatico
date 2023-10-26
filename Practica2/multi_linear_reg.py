import numpy as np
import copy
import math


def zscore_normalize_features(x):
    # """
    # computes  X, zcore normalized by column

    # Args:
    #   X (ndarray (m,n))     : input data, m examples, n features

    # Returns:
    #   X_norm (ndarray (m,n)): input normalized by column
    #   mu (ndarray (n,))     : mean of each feature
    #   sigma (ndarray (n,))  : standard deviation of each feature
    # """

    X_norm = np.zeros((np.size(x[:,:1]), np.size(x[1])), dtype='float32')
    mu =    np.zeros(np.size(x[1]), dtype='float32')
    sigma = np.zeros(np.size(x[1]), dtype='float32')

    for i in range(0, np.size(x[1])):
        mu[i] = np.mean(x[:,i:i+1])
        sigma[i] = np.std(x[:,i:i+1])
        X_norm[:,i:i+1] = (x[:,i:i+1] - mu[i]) / sigma[i]

    return (X_norm, mu,sigma)

def compute_cost(x, y, w, b):

    m = len(y)  # Número de ejemplos
    y_pred = np.dot(x, w) + b  # Predicciones del modelo

    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

    return cost


def compute_gradient(X, y, w, b):
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

    m = np.size(X[:, 0])
    predictions = np.dot(X, w) + b  # Predicciones del modelo

    # Compute the gradients
    dj_db = (1/m) * np.sum(predictions - y)
    dj_dw = (1/m) * np.dot(X.T, predictions - y)

    return dj_db, dj_dw



def gradient_descent(x, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
    # Inicialización
    w = w_in
    b = b_in

    # Inicializar el array que almacenará los costes que se vayan calculando
    # Todos sus elementos empiezan siendo 0
    J_history = np.zeros(num_iters)

    # Número de puntos
    m = len(y)

    for i in range(num_iters):

        # Guardar el coste de esta iteración
        J_history[i] = cost_function(x, y, w, b)

        # Calcular los gradientes
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Asegúrate de que dj_dw y dj_db sean arrays numpy
        dj_dw = np.array(dj_dw)
        dj_db = np.array(dj_db)

        # Cambiar w y b, para que se acerquen más a la recta real con cada iteración
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b, J_history



# def gradient_descent(x, y, w_in, b_in, cost_function,
#                      gradient_function, alpha, num_iters):
#     #Inicializacion
#     w = w_in
#     b = b_in

#     # Inicializar el array que almacenara los costes que se vayan calculando
#     # Todos sus elementos empiezan siendo 0
#     J_history = np.zeros(num_iters)

#     # Numero de puntos
#     m = len(y)
    
#     for i in range(num_iters):

#         # Guardar el coste de esta iteracion
#         J_history[i] = cost_function(x, y, w, b)
        
#         # Calcular los gradientes
#         dj_dw, dj_db = gradient_function(x, y, w, b)
        
#         # Cambiar w y b, para que se acerquen mas a la recta real con cada iteracion
#         w -= alpha * dj_dw
#         b -= alpha * dj_db
    
#     return w, b, J_history
