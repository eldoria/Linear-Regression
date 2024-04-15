import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, y):
        self.a = np.random.uniform(-10, 10)
        self.b = np.random.uniform(-10, 10)
        self.X = X
        self.y = y

    def sum_of_a_derivative(self):
        return sum([self.a_derivative(self.X[i], self.y[i]) for i in range(len(self.X))])

    def a_derivative(self, X, y):
        '''
        :param X: observed input
        :param y: observed output
        :return: derivative of the SSR according to a
        '''
        return -2 * X * (y - (self.a * X + self.b))

    def sum_of_b_derivative(self):
        return sum([self.b_derivative(X[i], y[i]) for i in range(len(self.X))])

    def b_derivative(self, X, y):
        '''
        :param X: observed input
        :param y: observed output
        :return: derivative of the SSR according to b
        '''
        return -2 * (y - (self.a * X + self.b))

    def plot(self, iteration=0):
        plt.scatter(self.X, self.y)

        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = self.a * x_vals + self.b

        plt.plot(x_vals, y_vals, color='red')
        plt.savefig(f'images/image_{iteration}.png')
        plt.show()

    def fit(self, learning_rate=1e-4):
        '''
        Find the optimal weights for the models using gradient descent
        :param learning_rate: Attenuate the gradient by this factor
        :param steps: number of times we iterate using gradient descent
        '''
        i = 0
        self.plot()
        while True:
            a_gradient = self.sum_of_a_derivative() * learning_rate
            b_gradient = self.sum_of_b_derivative() * learning_rate

            self.a -= a_gradient
            self.b -= b_gradient
            i += 1
            self.plot(iteration=i)
            print(a_gradient)
            print(b_gradient)
            if abs(a_gradient) + abs(b_gradient) < 1e-4:
                break


if __name__ == "__main__":
    X, y = make_regression(n_samples=300, n_features=1, noise=20, random_state=42)

    LinearRegression(X.flatten(), y).fit()
