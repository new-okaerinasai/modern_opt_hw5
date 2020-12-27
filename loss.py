import tensorflow as tf


class StrangeFunc:
    def __init__(self, A):
        self.A = A

    def __call__(self, x):
        return tf.reduce_max(tf.tensordot(self.A, x, 1))


class SmoothStrangeFunc:
    def __init__(self, A, mu):
        self.A = A
        self.mu = mu

    def __call__(self, x):
        return self.mu * tf.math.reduce_logsumexp(tf.tensordot(self.A, x, 1) / self.mu)


def solve_dichotomy(func, x_max):
    x_min = 0.0
    x_star = x_max
    eps = tf.Variable(1e-6)
    increasing = False
    if func(x_max) > func(x_min):
        increasing = True
    while tf.abs(x_min - x_max) > eps:
        x_star = (x_min + x_max) / 2
        y = func(x_star)
        if not increasing:
            if y > 0:
                x_min = x_star
            else:
                x_max = x_star
        else:
            if y < 0:
                x_min = x_star
            else:
                x_max = x_star
    return x_star


def project(x):
    x = tf.nn.relu(x)
    x_lambda = lambda l: (x + tf.sqrt(x ** 2 + 4 * l)) / 2
    # print(x_lambda(1e-4), x_lambda(100))
    func = lambda l: -tf.reduce_sum(tf.math.log(x_lambda(l)))
    x_max = 100
    lambda_star = solve_dichotomy(func, x_max)
    x_star = x_lambda(lambda_star)
    return x_star


if __name__ == "__main__":
    pass
