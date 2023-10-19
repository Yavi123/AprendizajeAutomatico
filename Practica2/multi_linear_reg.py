import numpy as np
import copy
import math


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """

    return (X_norm, mu, sigma)


def compute_cost(x, y, w, b):

    m = len(y)  # Número de ejemplos
    y_pred = np.dot(x, w) + b  # Predicciones del modelo

    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

    return cost


# def compute_gradient(x, y, w, b):
#     # Numero de puntos
#     m = len(x)
#     # linearRegressionValues almacena que valor en y deberian tener los numeros aleatorios si estuvieran en la recta
#     linearRegressionValues = (x * w) + b
#     # La diferencia de distancia entre los puntos aleatorios (x, y) y los calculados dentro de la recta
#     dif = linearRegressionValues - y

#     # aplicamos la derivada de la funcion de coste anterior
#     dj_dw = (1 / m) * np.sum(dif * x)
#     dj_db = (1 / m) * np.sum(dif)

#     return dj_dw, dj_db

def compute_gradient(X, y, w, b):
    m, n = X.shape  # Número de ejemplos (m) y número de características (n)
    y_pred = np.matmul(X, w) + b  # Predicciones del modelo

    # Gradiente con respecto a los parámetros w
    dj_dw = (1 / m) * np.matmul(X.T, (y_pred - y))

    # Gradiente con respecto al parámetro b
    dj_db = (1 / m) * np.sum(y_pred - y)

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
    #Inicializacion
    w = w_in
    b = b_in

    # Inicializar el array que almacenara los costes que se vayan calculando
    # Todos sus elementos empiezan siendo 0
    J_history = np.zeros(num_iters)

    # Numero de puntos
    m = len(y)
    
    for i in range(num_iters):

        # Guardar el coste de esta iteracion
        J_history[i] = cost_function(x, y, w, b)
        
        # Calcular los gradientes
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        # Cambiar w y b, para que se acerquen mas a la recta real con cada iteracion
        w -= alpha * dj_dw
        b -= alpha * dj_db
    
    return w, b, J_history
