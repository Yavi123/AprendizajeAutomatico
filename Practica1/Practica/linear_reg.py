import numpy as np
import copy
import math




#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
    m = np.size(x)

    # Calcular las predicciones en Y
    # Los valores que tendria Y si estuviese dentro de la recta de regresion
    prediction = np.multiply(x, w) + b

    # Resultado del sumatorio de la diferencia de costes entre la prediccion y los puntos de referencia
    res = np.sum((prediction - y)**2)

    # Calcular la media de los costes, 2 para simplificar mas
    total_cost = res / (2 * m)

    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
    # Numero de puntos
    m = len(x)
    # linearRegressionValues almacena que valor en y deberian tener los numeros aleatorios si estuvieran en la recta
    linearRegressionValues = (x * w) + b
    # La diferencia de distancia entre los puntos aleatorios (x, y) y los calculados dentro de la recta
    dif = linearRegressionValues - y

    # aplicamos la derivada de la funcion de coste anterior
    dj_dw = (1 / m) * np.sum(dif * x)
    dj_db = (1 / m) * np.sum(dif)

    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    
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