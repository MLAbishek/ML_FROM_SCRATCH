# Linear Regression from Scratch

## Overview
This repository contains an implementation of Linear Regression using gradient descent optimization. The implementation is built from first principles, with the mathematical derivations carefully worked out.

## Mathematical Foundation

### The Prediction Model
For a linear regression model:
- $\hat{y}_i = w_i x_i + b$ (for a single feature)
- $\hat{y}_i = w^T x_i + b$ (for multiple features)

Where:
- $\hat{y}_i$ is the predicted value
- $w_i$ is the weight for feature $i$
- $x_i$ is the input feature value
- $b$ is the bias term

### Loss Function
For a single sample, the loss function is defined as:
$$L = \frac{1}{2}(y_i - \hat{y}_i)^2 = \frac{1}{2}(y_i - (w^T x_i + b))^2$$

Where:
- $y_i$ is the actual target value
- $\hat{y}_i$ is the predicted value

### Gradient Descent
To optimize our model, we need to compute the gradients of the loss function with respect to the weights and bias:

#### Gradient with respect to weights:
$$\frac{\partial L}{\partial w_i} = -\frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial w_i} = -(y_i - \hat{y}_i) \cdot x_i$$

Since:
$$\frac{\partial \hat{y}_i}{\partial w_i} = x_i$$

And:
$$\frac{\partial L}{\partial \hat{y}_i} = -(y_i - \hat{y}_i)$$

#### Gradient with respect to bias:
$$\frac{\partial L}{\partial b} = -\frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial b} = -(y_i - \hat{y}_i) \cdot 1 = -(y_i - \hat{y}_i)$$

Since:
$$\frac{\partial \hat{y}_i}{\partial b} = 1$$

### Update Rules
For each epoch in gradient descent:
- $w = w - \alpha \cdot \frac{\partial L}{\partial w}$
- $b = b - \alpha \cdot \frac{\partial L}{\partial b}$

Where $\alpha$ is the learning rate.

## Implementation

The `LinearRegressionScratch` class provides a simple implementation of linear regression using gradient descent.

```python
import numpy as np
import pandas as pd


class LinearRegressionScratch:
    """
    Author @Abishek J
    A simple linear regression model that uses gradient descent for optimization.
    weights -> The weights of the model.
    bias    -> The bias of the model.
    sample  -> The number of rows in the dataset.
    feature ->The number of columns in the dataset.
    dw      -> The gradient of the loss function with respect to the weights.
    db      -> The gradient of the loss function with respect to the bias.
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, x, y, learning_rate=0.01, epochs=1500):
        sample, feature = x.shape
        self.weights = np.zeros(feature)
        self.bias = 0

        for n in range(epochs):
            y_pred = np.dot(x, self.weights) + self.bias

            dw = (1 / sample) * np.dot(x.T, (y_pred - y))
            db = (1 / sample) * np.sum(y_pred - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred
```

### Key Methods:

1. **`__init__`**: Initializes the model weights and bias as None.
2. **`fit`**: Trains the model on the provided data using gradient descent.
   - Initializes weights to zero vector and bias to 0
   - For each epoch:
     - Computes predictions: $\hat{y} = Xw + b$
     - Calculates gradients for weights and bias
     - Updates weights and bias using the gradients and learning rate
3. **`predict`**: Makes predictions on new data using the trained model.

## Usage Example

```python
# Import necessary libraries
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import our custom linear regression model
from linear_regression import LinearRegressionScratch

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegressionScratch()
model.fit(X_train, y_train, learning_rate=0.01, epochs=1000)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Learned weights: {model.weights}")
print(f"Learned bias: {model.bias:.2f}")
```

## Connection to Mathematical Derivation

The code implementation directly follows the mathematical derivation shown in the handwritten notes:

1. The prediction formula: $\hat{y}_i = w_i x_i + b$ is implemented as `y_pred = np.dot(x, self.weights) + self.bias`

2. The gradient calculations:
   - $\frac{\partial L}{\partial w} = -(y_i - \hat{y}_i) \cdot x_i$ is implemented as `dw = (1 / sample) * np.dot(x.T, (y_pred - y))`
   - $\frac{\partial L}{\partial b} = -(y_i - \hat{y}_i)$ is implemented as `db = (1 / sample) * np.sum(y_pred - y)`

3. The weight updates:
   - $w = w - \alpha \cdot \frac{\partial L}{\partial w}$ is implemented as `self.weights -= learning_rate * dw`
   - $b = b - \alpha \cdot \frac{\partial L}{\partial b}$ is implemented as `self.bias -= learning_rate * db`

The implementation averages the gradients across all samples (by dividing by `sample`), which gives us the batch gradient descent approach.

## Author
@Abishek J
