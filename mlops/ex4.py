from typing import Tuple

from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt


def activation_function(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def activation_function_deriv(x: float) -> float:
    return activation_function(x) * (1 - activation_function(x))


class Neuron:
    def __init__(self, input_size: int, act_func: Callable, act_func_deriv: Callable):
        self._init_weights_and_bias(input_size)
        self._activation_function = act_func
        self._activation_function_deriv = act_func_deriv

    def _init_weights_and_bias(self, input_size: int):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def __call__(self, x: np.array) -> float:
        return self._forward_propagation(x)

    def _forward_propagation(self, x: np.array) -> float:
        z = np.dot(self.weights, x) + self.bias
        return self._activation_function(z)

    def gradient_descent(self, x: np.array, y_target: float, alpha: float, iterations: int) -> None:
        for _ in range(iterations):
            dW, dB = self._backward_propagation(x, y_target)
            self.weights -= alpha * dW
            self.bias -= alpha * dB

    def _backward_propagation(self, x: np.array, y: float) -> tuple[np.array, float]:
        output = self._forward_propagation(x)
        error = output - y
        dZ = error * self._activation_function_deriv(np.dot(self.weights, x) + self.bias)
        dW = dZ * x
        dB = dZ
        return dW, dB


def train_neuron(neuron: Neuron, dataset_x: np.array, dataset_y: np.array, alpha: float, epochs: int) -> None:
    for _ in range(epochs):
        for x, y in zip(dataset_x, dataset_y):
            x = np.array(x)
            neuron.gradient_descent(x, y, alpha, 1)


def visualize_results(neuron: Neuron, dataset_x: np.array, dataset_y: np.array, title: str) -> None:
    fig, ax = plt.subplots()
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([neuron(point) for point in grid_points]).reshape(xx.shape)

    # Plot decision boundary (regression line)
    contour = ax.contourf(xx, yy, zz, levels=50, cmap="coolwarm", alpha=0.6)

    # Plot the dataset points with labels
    for point, y in zip(dataset_x, dataset_y):
        color = "red" if y == 1 else "blue"
        label = "Output: 1" if y == 1 else "Output: 0"
        ax.scatter(point[0], point[1], color=color, label=label)

    # Prevent duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    plt.colorbar(contour)
    plt.show()


class NeuralNetwork:
    def __init__(self, input_size: int, act_func: Callable, act_func_deriv: Callable):
        self._neuron_1 = Neuron(input_size, act_func, act_func_deriv)
        self._neuron_2 = Neuron(input_size, act_func, act_func_deriv)
        self._neuron_3 = Neuron(2, act_func, act_func_deriv)

    def __call__(self, x: np.array) -> float:
        return self._network_forward_propagation(x)

    def _network_forward_propagation(self, x: np.array) -> float:
        input_3_1 = self._neuron_1(x)
        input_3_2 = self._neuron_2(x)
        input_3 = np.array([input_3_1, input_3_2])
        return self._neuron_3(input_3)

    def _network_backwards_propagation(self, x: np.array, y: float) -> tuple[np.array, float, float]:
        # Forward pass
        output = self._network_forward_propagation(x)

        # Compute the error
        error = output - y

        # Backpropagation
        dZ3 = error * self._neuron_3._activation_function_deriv(
            np.dot(self._neuron_3.weights, np.array([self._neuron_1(x), self._neuron_2(x)])) + self._neuron_3.bias
        )
        dW3 = dZ3 * np.array([self._neuron_1(x), self._neuron_2(x)])
        dB3 = dZ3

        # Compute gradients for neuron 2
        dZ2 = (
            dZ3
            * self._neuron_3.weights[1]
            * self._neuron_2._activation_function_deriv(np.dot(self._neuron_2.weights, x) + self._neuron_2.bias)
        )
        dW2 = dZ2 * x
        dB2 = dZ2

        # Compute gradients for neuron 1
        dZ1 = (
            dZ3
            * self._neuron_3.weights[0]
            * self._neuron_1._activation_function_deriv(np.dot(self._neuron_1.weights, x) + self._neuron_1.bias)
        )
        dW1 = dZ1 * x
        dB1 = dZ1

        return dW1, dB1, dW2, dB2, dW3, dB3

    def gradient_descent(self, x: np.array, y: float, alpha: float) -> None:
        dW1, dB1, dW2, dB2, dW3, dB3 = self._network_backwards_propagation(x, y)

        # Update weights and biases for each neuron
        self._neuron_1.weights -= alpha * dW1
        self._neuron_1.bias -= alpha * dB1

        self._neuron_2.weights -= alpha * dW2
        self._neuron_2.bias -= alpha * dB2

        self._neuron_3.weights -= alpha * dW3
        self._neuron_3.bias -= alpha * dB3


def train_network(network: NeuralNetwork, dataset_x: np.array, dataset_y: np.array, alpha: float, epochs: int) -> None:
    for _ in range(epochs):
        for x, y in zip(dataset_x, dataset_y):
            network.gradient_descent(np.array(x), y, alpha)


def visualize_results_net(neuralnet: NeuralNetwork, dataset_x: np.array, dataset_y: np.array, title: str) -> None:
    fig, ax = plt.subplots()
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([neuralnet(point) for point in grid_points]).reshape(xx.shape)

    # Plot decision boundary (regression line)
    contour = ax.contourf(xx, yy, zz, levels=50, cmap="coolwarm", alpha=0.6)

    # Plot the dataset points with labels
    for point, y in zip(dataset_x, dataset_y):
        color = "red" if y == 1 else "blue"
        label = "Output: 1" if y == 1 else "Output: 0"
        ax.scatter(point[0], point[1], color=color, label=label)

    # Prevent duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    # Customize plot
    ax.set_title(title)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    plt.colorbar(contour)
    plt.show()
