import numpy as np
import copy
import math


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_= 0):
    
    m = len(y)  # Number of examples

    # Calculate the logistic function
    z = np.dot(X, w) + b
    f_w_b = sigmoid(z)

    # Compute the cost
    term1 = -y * np.log(f_w_b)
    term2 = -(1 - y) * np.log(1 - f_w_b)

    total_cost = (1 / m) * np.sum(term1 + term2)
    
    return total_cost


def compute_gradient(X, y, w, b, lambda_= 0):
    """
    Computes the gradient for logistic regression

    Args:
    X (ndarray): Feature matrix of shape (m, n), where m is the number of examples and n is the number of features.
    y (ndarray): Target vector of shape (m,).
    w (ndarray): Weight vector of shape (n,).
    b (float): Bias term.
    lambda_ (float, optional): Regularization parameter (default is None).

    Returns:
    dj_db (float): The gradient of the cost w.r.t. the parameter b.
    dj_dw (ndarray): The gradient of the cost w.r.t. the parameters w.
    """
    
    m = len(y)  # Number of examples

    # Calculate the logistic function
    z = np.dot(X, w) + b

    f_w_b = sigmoid(z)

    # Compute the gradients
    dj_db = (1 / m) * np.sum(f_w_b - y)
    dj_dw = (1 / m) * np.dot(X.T, (f_w_b - y))

    return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """

    cost = compute_cost(X, y, w, b)

    m = X.shape[0]
    # Cacular el penalty con Regularization L2 (Ridge)
    penaltyL2 = ( lambda_ / ( 2*m ) ) * np.sum(w**2)

    # Aplicar el penalty
    cost += penaltyL2

    return cost


def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """

    # Calcular valores sin penalty, para aplicarselo despues
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    # Sumar penalty
    m = len(y)
    dj_dw += ((lambda_ / m) * w)

    return dj_db, dj_dw 


#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha
    
    Args:
    X : (array_like Shape (m, n)) Feature matrix.
    y : (array_like Shape (m,)) Target vector.
    w_in : (array_like Shape (n,)) Initial values of parameters of the model.
    b_in : (scalar) Initial value of parameter of the model.
    cost_function: function to compute cost.
    gradient_function: function to compute gradient.
    alpha : (float) Learning rate.
    num_iters : (int) number of iterations to run gradient descent.
    lambda_ (scalar, float) unused placeholder.

    Returns:
    w : (array_like Shape (n,)) Updated values of parameters of the model after running gradient descent.
    b : (scalar) Updated value of parameter of the model after running gradient descent.
    J_history : (ndarray): Shape (num_iters,) J at each iteration, primarily for graphing later.
    """

    # Valores iniciales
    w = w_in
    b = b_in
    J_history = np.zeros(num_iters)

    # Calculo de gradientes
    for i in range(num_iters):
        
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history[i] = cost_function(X, y, w, b, lambda_)

    return w, b, J_history


#########################################################################
# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """


    # Predicciones en Y de todas las X que existen (en este caso examenes hechos)
    # Multiplicando vectores de tamaño(m, n) y (n, 1) sale (m, 1)
    # Predictions por lo tanto sera de tamaño m
    predictions = np.dot(X, w) + b

    # Ajustar a valores entre 0 y 1
    s = sigmoid(predictions)

    # Si esta por encima de la mitad, aprobado
    p = (s >= 0.5)

    return p


def predict_test(target):
    np.random.seed(5)
    b = 0.5    
    w = np.random.randn(3)
    X = np.random.randn(8, 3)
    
    result = target(X, w, b)
    wrong_1 = [1., 1., 0., 0., 1., 0., 0., 1.]
    expected_1 = [1., 1., 1., 0., 1., 0., 0., 1.]
    if np.allclose(result, wrong_1):
        raise ValueError("Did you apply the sigmoid before applying the threshold?")
    assert result.shape == (len(X),), f"Wrong length. Expected : {(len(X),)} got: {result.shape}"
    assert np.allclose(result, expected_1), f"Wrong output: Expected : {expected_1} got: {result}"
    
    b = -1.7    
    w = np.random.randn(4) + 0.6
    X = np.random.randn(6, 4)
    
    result = target(X, w, b)
    expected_2 = [0., 0., 0., 1., 1., 0.]
    assert result.shape == (len(X),), f"Wrong length. Expected : {(len(X),)} got: {result.shape}"
    assert np.allclose(result,expected_2), f"Wrong output: Expected : {expected_2} got: {result}"

    print('\033[92mAll tests passed!')