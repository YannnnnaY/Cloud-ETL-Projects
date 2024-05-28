import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# sample dataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

"""
# Dataset 1
X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5],
              [5, 6]])
y = np.array([3, 4, 5, 6, 7])

# Plot the data
plt.scatter(X[:, 0], y, color='blue', label='X[:, 0]')
plt.scatter(X[:, 1], y, color='red', label='X[:, 1]')
plt.xlabel('Feature values')
plt.ylabel('Target values')
plt.legend()
plt.show()
"""

# Dataset 2 - California housing
california_housing = fetch_california_housing()
X_raw = california_housing.data  # Feature matrix
y = california_housing.target  # Target vector

imputer = SimpleImputer(strategy='mean')  # Use mean imputation
X_raw = imputer.fit_transform(X_raw)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
feature_names = california_housing.feature_names
# for i in range(X.shape[1]):
#     plt.scatter(X[:, i], y)
#     plt.xlabel(feature_names[i])
#     plt.ylabel('Median House Value')
#     plt.title(f'Scatter plot: {feature_names[i]} vs. Median House Value')
#     plt.show()
    


"""
scikit-learn
"""

# Linear Regression model
regression_model = LinearRegression()
regression_model.fit(X, y)

# intercept and coefficient
intercept = regression_model.intercept_
coefficients = regression_model.coef_

# Make predictions
predicted = regression_model.predict(X)

# Calculate MSE
mse = mean_squared_error(y, predicted)

print(f'The results by Scikit-learn package are: ')
print(f'Intercept: {intercept}')
print(f'Coefficients: {coefficients}')
print(f'MSE loss: {mse}\n')


"""
Newton-Raphson
"""



def newton_raphson(X, y, num_iterations=1000, learning_rate=0.1, threshold=1e-6):
    # Initialize parameters
    number_sample, number_feature = X.shape
    intercept = np.zeros(1)
    coefficients = np.zeros(number_feature)  # Adjust the shape for the intercept term

    # Add a column of ones to X for the intercept term
    X = np.concatenate((X, np.ones((number_sample, 1))), axis=1)    # Dataset - CA housing

    for iteration in range(num_iterations):
        # predicted values
        predicted = np.dot(X, np.concatenate((coefficients, intercept))) # Dataset - CA housing
        # predicted = np.dot(X, coefficients) + intercept  # Dataset 1

        # loss gradient
        residuals = predicted - y
        loss_gradient = 2 * np.dot(X.T, residuals) / number_sample
        mse_loss = np.mean((residuals) ** 2)

        # Hessian matrix
        loss_hessian = 2 * np.dot(X.T, X) / number_sample

        # update parameters
        update = np.linalg.solve(loss_hessian, loss_gradient)
        coefficients = coefficients - learning_rate * update[:-1]  # exclude the intercept term
        intercept = intercept - learning_rate * update[-1]  # update the intercept term

        # convergence
        if np.linalg.norm(update) < threshold:
            print("Converged after", iteration+1, "iterations.")
            break

    return intercept, coefficients, mse_loss


start_time = time.time()
intercept, coefficients, loss = newton_raphson(X, y)
end_time = time.time()

print('The results by Newton-Raphson are:')
print(f'Intercept: {intercept}')
print(f'Coefficients:{coefficients}')
print(f'My MSE loss: {loss}')
print(f'runtime: {end_time-start_time}\n')



"""
Gradient Descent 
"""


def gradient_descent(X, y, num_iterations=5000, learning_rate=0.01, threshold=1e-6):
    # Initialize parameters
    number_sample, number_feature = X.shape
    intercept = np.zeros(1)
    coefficient = np.zeros(number_feature)

    for iteration in range(num_iterations):
        # Compute predicted values
        predicted = np.dot(X, coefficient) + intercept

        # Compute loss gradient
        residual = predicted - y
        loss_gradient = 2 * np.dot(X.T, residual) / number_sample
        mse_loss = np.mean(residual ** 2)

        # Update parameters using gradient descent
        # california housing dataset
        coefficient = coefficient - learning_rate * loss_gradient
        intercept = intercept - learning_rate * np.mean(residual)

        # Dataset 1
        # coefficient = coefficient - learning_rate * loss_gradient[:-1]
        # intercept = intercept - learning_rate * loss_gradient[-1]

        # Check convergence
        if np.linalg.norm(loss_gradient) < threshold:
            print("Converged after", iteration+1, "iterations.")
            break

    return intercept, coefficient , mse_loss


start_time = time.time()
intercept, coefficients, loss = gradient_descent(X, y)
end_time = time.time()

print('The results by Gradient Descent are:')
print(f'Intercept: {intercept}')
print(f'Coefficients:{coefficients}')
print(f'My MSE loss: {loss}')
print(f'runtime: {end_time-start_time}\n')


