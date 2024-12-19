import math
from typing import List

import numpy as np

import mnist_loader


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

raw_training_data, _, _ = mnist_loader.load_data()
print(f"Min pixel value: {raw_training_data[0][0].min()}")
print(f"Max pixel value: {raw_training_data[0][0].max()}")


print(f"Training data size: {len(training_data)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")

print(f"Example training input shape: {training_data[0][0].shape}")
print(f"Example training output shape: {training_data[0][1].shape}")


class FNN:
    """
    Fully Connected Neural Network (Feedforward Neural Network).

    This class implements a neural network with a customizable number of
    hidden layers, ReLU activation, and softmax output. It supports
    backpropagation for training using gradient descent.

    Attributes:
        input_size (int): The number of input features.
        hidden_sizes (List[int]): List containing the number of neurons in each hidden layer.
        output_size (int): The number of output classes.
        learning_rate (float): The learning rate for gradient descent.
        weights (List[np.ndarray]): List of weight matrices for each layer.
        biases (List[np.ndarray]): List of bias vectors for each layer.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, learning_rate: float = 0.5) -> None:
        """
        Initialize the neural network with random weights and biases.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (List[int]): List of neurons in each hidden layer.
            output_size (int): The number of output classes.
            learning_rate (float): Learning rate for gradient descent (default 0.5).
        """
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros((hidden_sizes[0], 1)))

        for layer_index in range(1, len(hidden_sizes)):
            self.weights.append(
                np.random.randn(hidden_sizes[layer_index - 1], hidden_sizes[layer_index])
                * math.sqrt(2 / hidden_sizes[layer_index - 1])
            )
            self.biases.append(np.zeros((hidden_sizes[layer_index], 1)))

        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * math.sqrt(1 / hidden_sizes[-1]))
        self.biases.append(np.zeros((output_size, 1)))

    def relu(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the ReLU activation function.

        Args:
            z (np.ndarray): The input array.

        Returns:
            np.ndarray: Element-wise ReLU applied to the input.
        """
        return np.maximum(0, z)

    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the ReLU activation function.

        Args:
            z (np.ndarray): The input array.

        Returns:
            np.ndarray: Derivative of ReLU applied element-wise.
        """
        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax activation function.

        Args:
            z (np.ndarray): The input array.

        Returns:
            np.ndarray: Softmax output probabilities.
        """
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.

        Args:
            y_true (np.ndarray): One-hot encoded true labels.
            y_pred (np.ndarray): Predicted probabilities from softmax.

        Returns:
            float: The average cross-entropy loss.
        """
        m = y_true.shape[1]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / m

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Args:
            x (np.ndarray): Input data (features).

        Returns:
            np.ndarray: Output probabilities from the softmax layer.
        """
        self.activation: List[np.ndarray] = [x]
        self.z_values: List[np.ndarray] = []

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w.T, self.activation[-1]) + b
            self.z_values.append(z)
            self.activation.append(self.relu(z))

        z_output = np.dot(self.weights[-1].T, self.activation[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        self.output = self.softmax(z_output)
        return self.output

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Perform backpropagation to compute gradients and update weights.

        Args:
            x (np.ndarray): Input data (features).
            y (np.ndarray): True labels (one-hot encoded).
        """
        m = x.shape[1]
        dz = self.output - y

        grads_w: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []

        dw = np.dot(self.activation[-1], dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        grads_w.append(dw)
        grads_b.append(db)

        for layer_index in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(self.weights[layer_index + 1], dz) * self.relu_derivative(self.z_values[layer_index])
            dw = np.dot(self.activation[layer_index], dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            grads_w.append(dw)
            grads_b.append(db)

        grads_w.reverse()
        grads_b.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int) -> None:
        """
        Train the neural network using gradient descent.

        Args:
            x_train (np.ndarray): Training data (features).
            y_train (np.ndarray): Training labels (one-hot encoded).
            epochs (int): Number of epochs for training.
        """
        for epoch in range(1, epochs + 1):
            if epoch == 400:
                self.learning_rate = 0.1
            indices = np.random.permutation(x_train.shape[1])
            x_train = x_train[:, indices]
            y_train = y_train[:, indices]

            self.forward(x_train)
            self.backward(x_train, y_train)

            loss = self.cross_entropy_loss(y_train, self.output)
            if (epoch) % 50 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the model on the test set.

        Args:
            x_test (np.ndarray): Test data (features).
            y_test (np.ndarray): True labels (one-hot encoded).

        Returns:
            float: Accuracy of the model on the test set.
        """
        y_pred = self.forward(x_test)
        predictions = np.argmax(y_pred, axis=0)
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy


# Data normalization
x_train = np.hstack([x for x, _ in training_data])
y_train = np.hstack([y for _, y in training_data])

x_val = np.hstack([x for x, _ in validation_data])
y_val = np.array([y for _, y in validation_data])

x_test = np.hstack([x for x, _ in test_data])
y_test = np.array([y for _, y in test_data])


print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

input_size = x_train.shape[0]
output_size = y_train.shape[0]

layer_configs = [[128], [128, 64], [256, 128, 64]]

learning_rates = [0.1, 0.5]

results = []
for hidden_sizes in layer_configs:
    for learning_rate in learning_rates:
        print(f"\nRunning experiment with layers: {hidden_sizes}, learning rate: {learning_rate}")

        model = FNN(input_size, hidden_sizes, output_size, learning_rate)
        model.train(x_train, y_train, 500)

        print("Final evaluation on test set:")
        accuracy = model.evaluate(x_test, y_test)

        results.append({"layers": hidden_sizes, "learning_rate": learning_rate, "test_accuracy": accuracy})
