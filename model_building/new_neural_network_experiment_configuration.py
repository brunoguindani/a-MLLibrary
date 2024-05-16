"""
Copyright 2019 Marco Lattuada
Copyright 2021 Bruno Guindani

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import sklearn.linear_model as lr
import torch
import torch.nn as nn


import model_building.experiment_configuration as ec

class NeuralNetwork(nn.Module):
    """
    Definizione della classe della rete neurale.
    """

    def __init__(self, layer_sizes, dropout_prob):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                # Activate Function
                layers.append(nn.ReLU())
                # Dropout
                layers.append(nn.Dropout(dropout_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class NewNeuralNetworkExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for linear regression

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the linear model

    print_model()
        Print the representation of the generated model

    initialize_regressor()
        Initialize the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values
    """
    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dict of str: object
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved

        prefix: list of str
            The prefix to be added to the signature of this experiment configuration
        """
        assert prefix
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.NEWNEURAL

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration

        Parameters
        ----------
        prefix: list of str
            The signature of this experiment configuration without considering hyperparameters

        Returns
        -------
            The signature of the experiment
        """
        assert isinstance(prefix, list)
        signature = prefix.copy()
        signature.append("layer_sizes" + str(self._hyperparameters['layer_sizes']))
        signature.append("dropout_prob" + str(self._hyperparameters['dropout_prob']))
        signature.append("epochs" + str(self._hyperparameters['epochs'])) 
        return signature

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        x_train_tensor = torch.tensor(xdata, dtype=torch.float32)
        y_train_tensor = torch.tensor(ydata, dtype=torch.float32)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self._regressor.parameters(), lr=0.01)
        # Train the neural network
        for epoch in range(self._hyperparameters['epochs']): #TO DO epoch
            # Forward pass: compute predicted y by passing x to the model
            y_pred = self._regressor.forward(x_train_tensor)

            # Compute loss
            loss = loss_fn(y_pred, y_train_tensor)

            # Zero gradients, backward pass, and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._logger.debug("Model built")
        for idx, col_name in enumerate(self.get_x_columns()):
            self._logger.debug("The coefficient for %s is %f", col_name, self._regressor.coef_[idx])

    def compute_estimations(self, rows):#to do
        """
        Compute the predictions for data points indicated in rows estimated by the regressor

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        return self._regressor.predict(xdata)

    def print_model(self):
        """
        Print the representation of the generated model
        """
        initial_string = "New Neural Network:\n"
        ret_string = initial_string
        coefficients = self._regressor.coef_
        columns = self.get_x_columns()

        assert len(columns) == len(coefficients)

        # Show coefficients in order of decresing absolute value
        idxs = np.argsort(np.abs(coefficients))[::-1]
        signif_digits = 4
        for i in idxs:
            column = columns[i]
            coefficient = coefficients[i]
            ret_string += " + " if ret_string != initial_string else "   "
            coeff = str(round(coefficient, signif_digits))
            ret_string = ret_string + "(" + str(coeff) + " * " + column + ")\n"
        coeff = str(round(self._regressor.intercept_, signif_digits))
        ret_string = ret_string + " + (" + coeff + ")"
        return ret_string

    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments
        """
        if not getattr(self, '_hyperparameters', None):
            self._regressor = NeuralNetwork()
        else:
            self._regressor = NeuralNetwork(
                layer_sizes = self._hyperparameters['layer_sizes'],
                dropout_prob = self._hyperparameters['dropout_prob'])            

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return {'layer_sizes': [64, 32], 'dropout_prob': 0.5, 'epochs': 100}  # Set default value for epochs
