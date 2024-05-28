import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss



# Dataset1 - a small one
X = np.array([[1, 2],
              [2, 1],
              [3, 4],
              [4, 3],
              [5, 6],
              [6, 5]])
y = np.array([0, 0, 1, 1, 1, 1])

# Dataset2 - Random generated
# np.random.seed(0)
# X1 = np.random.randn(100, 2) + np.array([2, 2])
# X2 = np.random.randn(100, 2) + np.array([-2, -2])
# X = np.vstack((X1, X2))
# y = np.hstack((np.zeros(100), np.ones(100)))

# Visualization
plt.scatter(X[:, 0], y, color='blue', label='X[:, 0]')
plt.scatter(X[:, 1], y, color='red', label='X[:, 1]')
plt.xlabel('Feature values')
plt.ylabel('Target values')


# Dataset3 - Breast Cancer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
# X = data.data
# y = data.target

# select 2 features for visualization
# feature1_index = 0
# feature2_index = 1
# plt.scatter(X[:, feature1_index], X[:, feature2_index], c=y, cmap='bwr')
# plt.xlabel(data.feature_names[feature1_index])
# plt.ylabel(data.feature_names[feature2_index])
# plt.title("Breast Cancer Dataset")
# plt.colorbar(ticks=[0, 1], label='Target')

# Plot the data
plt.legend()
plt.show()



"""
scikit-learn
"""

# the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# intercept and coefficients
intercept = model.intercept_
coefficient = model.coef_

# predicted probabilities
probabilities = model.predict_proba(X)
loss = log_loss(y, probabilities)

print('The results by Scikit-learn package are:')
print(f'Intercept: {intercept}')
print(f'Coefficients: {coefficient}')
print(f'Log Loss: {loss}\n')

"""
log loss for binary classification
"""

def my_log_loss(y, p):
    epsilon = 1e-15  # Small value to avoid numerical instability
    p = np.clip(p, epsilon, 1 - epsilon)  # Clip probabilities to prevent log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


"""
The sigmoid function is commonly used as an activation function in neural networks, particularly in the output layer for binary classification problems.

The sigmoid function is also used in logistic regression, where it models the probability of an event occurring given input features. 
It maps the linear combination of input features and their corresponding coefficients to a probability value between 0 and 1.
"""
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


"""
Newton-Raphson
"""

def newton_raphson(X, y, learning_rate=0.01, convergence_threshold=0.00001, num_iterations=1000):
    # Add bias term
    X = np.c_[np.ones(X.shape[0]), X]

    # Initialize weights
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)

    for iteration in range(num_iterations):
        scores = np.dot(X, weights)
        predictions = sigmoid(scores)
        error = y - predictions

        # Calculate the Hessian matrix
        hessian = -np.dot(X.T, np.dot(np.diag(predictions * (1 - predictions)), X))

        # Calculate the gradient
        gradient = np.dot(X.T, error)

        # Update the weights using Newton-Raphson update rule
        weights -= learning_rate * np.linalg.inv(hessian).dot(gradient)

        probabilities = expit(scores)
        loss = my_log_loss(y, probabilities)

        # Check convergence based on the change in weights
        weight_change = learning_rate * np.linalg.norm(gradient)
        if weight_change < convergence_threshold:
            print("Converged after", iteration+1, "iterations.")
            break

    intercept = weights[0]
    coefficients = weights[1:]

    return intercept, coefficients, loss


start_time = time.time()
intercept, coefficient, loss = newton_raphson(X, y)
end_time = time.time()

print('The results by Newton-Raphson are:')
print(f'Intercept: {intercept}')
print(f'Coefficients:{coefficient}')
print(f'My log loss: {loss}')
print(f'runtime: {end_time-start_time}\n')


"""
Gradient Descent
"""


def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000, threshold=0.0001):
    # Add bias term
    X = np.c_[np.ones(X.shape[0]), X]

    # Initialize weights
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)

    for iteration in range(num_iterations):
        scores = np.dot(X, weights)
        predictions = sigmoid(scores)
        error = y - predictions

        gradient = np.dot(X.T, error)
        weights += learning_rate * gradient
        # Calculate the change in weights
        weight_change = learning_rate * np.linalg.norm(gradient)

        probabilities = expit(scores)
        loss = my_log_loss(y, probabilities)

        # Check convergence based on the change in weights
        if weight_change < threshold:
            print("Converged after", iteration+1, "iterations.")
            break

    intercept = weights[0]
    coefficients = weights[1:]

    return intercept, coefficients, loss


start_time = time.time()
intercept, coefficient, loss = gradient_descent(X, y)
end_time = time.time()

print('The results by Gradient Descent are:')
print(f'Intercept: {intercept}')
print(f'Coefficients:{coefficient}')
print(f'My log loss: {loss}')
print(f'runtime: {end_time-start_time}\n')





