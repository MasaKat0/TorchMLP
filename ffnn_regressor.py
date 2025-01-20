import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class FeedforwardNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', 
                 learning_rate=0.001, max_epochs=200, batch_size=32, 
                 random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _build_model(self, input_dim):
        layers = []
        in_dim = input_dim

        for out_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        input_dim = X.shape[1]
        self._build_model(input_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()
        return outputs.flatten()
