
# Feedforward Neural Network Classifier and Regressor

This repository provides implementations of feedforward neural network models for classification and regression using PyTorch with a scikit-learn compatible interface.

## Installation

To install the required dependencies, run:

```bash
pip install torch numpy scikit-learn
```

## Usage

### Classification Example

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from ffnn_classifier import FeedforwardNNClassifier

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
clf = FeedforwardNNClassifier(hidden_layer_sizes=(50, 50), max_epochs=100)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

### Regression Example

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from ffnn_regressor import FeedforwardNNRegressor

# Generate synthetic regression data
X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
reg = FeedforwardNNRegressor(hidden_layer_sizes=(50, 50), max_epochs=100)
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)

print("Predictions:", predictions)
```

## Parameters

Both `FeedforwardNNClassifier` and `FeedforwardNNRegressor` share the following parameters:

- `hidden_layer_sizes`: Tuple defining the number of neurons in each hidden layer (default: `(100,)`).
- `activation`: Activation function to use ('relu' or 'tanh', default: `'relu'`).
- `learning_rate`: Learning rate for the optimizer (default: `0.001`).
- `max_epochs`: Maximum number of training epochs (default: `200`).
- `batch_size`: Batch size for training (default: `32`).
- `random_state`: Random seed for reproducibility (default: `None`).

## License

This project is licensed under the MIT License.

## Author

Masahiro Kato
