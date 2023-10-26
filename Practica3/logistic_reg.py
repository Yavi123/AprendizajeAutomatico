import numpy as np
import copy
import math


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
    
    m = len(y)  # Number of examples

    # Calculate the logistic function
    z = np.dot(X, w) + b
    f_w_b = sigmoid(z)

    # Compute the cost
    term1 = -y * np.log(f_w_b)
    term2 = -(1 - y) * np.log(1 - f_w_b)
    
    # Regularization term (if lambda_ is provided)
    if lambda_ is not None:
        regularization_term = (lambda_ / (2 * m)) * np.sum(w**2)
    else:
        regularization_term = 0

    total_cost = (1 / m) * np.sum(term1 + term2) + regularization_term
    
    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
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

    print("X = ")
    print(X)

    print("w = ")
    print(w)

    print("b = ")
    print(b)



    # Calculate the logistic function
    z = np.dot(X, w) + b

    print("z = ")
    print(z)

    f_w_b = sigmoid(z)

    print("f_w_b = ")
    print(f_w_b)

    print("y = ")
    print(y)

    # Compute the gradients
    dj_db = (1 / m) * np.sum(f_w_b - y)
    dj_dw = (1 / m) * np.dot(X.T, (f_w_b - y))

    # Regularization term (if lambda_ is provided)
    if lambda_ is not None:
        dj_dw += (lambda_ / m) * w

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

    return total_cost


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
    m, n = X.shape  # Number of examples and features
    w = w_in
    b = b_in
    J_history = np.zeros(num_iters)

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

    return p
