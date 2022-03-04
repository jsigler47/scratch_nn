import numpy as np
import matplotlib.pyplot as plt

# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropogate(self):
        d_weights2 = np.dot(
            self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(
            self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

        return (np.square(self.output - self.y)).mean(axis=0)


if __name__ == "__main__":

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)
    iterations = 1000
    losses = []
    for i in range(iterations):
        nn.feedforward()
        losses.append(nn.backpropogate())
    print(nn.output)
    print(nn.output.round())
    fig, ax = plt.subplots()

    ax.plot(losses)
    ax.set(xlabel='iteration', ylabel='loss', title='Loss over iterations')
    ax.grid()
    plt.show()
