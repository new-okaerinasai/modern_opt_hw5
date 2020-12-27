import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from loss import SmoothStrangeFunc, StrangeFunc, project
from run_stm import run_stm
from run_subgradient import run_subgrad

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_seed(121)
mu = 0.01
n = 50
m = 10
A = tf.abs(tf.random.normal((n, m), 2, 3))
lr = 1e-1
x0 = tf.abs(tf.random.normal(shape=(m,)) / np.sqrt(m))


def plot(loss_history, title, fname):
    loss_history = np.array(loss_history)
    loss_history = loss_history - loss_history.min()
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("loss - loss_min, log-log-scale")
    plt.yscale("log")
    plt.plot(loss_history)
    plt.savefig(fname)


def exp_stm():
    L_m = 1 / mu * tf.reduce_max(tf.linalg.norm(A, axis=1))
    strage_func = SmoothStrangeFunc(A, mu)
    xmin, loss_history = run_stm(strage_func, project, x0, max_iter=100, L_m=L_m)
    plot(loss_history, "stm loss/iteration", "./stm_n={}_m={}_mu={:.4f}.png".format(n, m, mu))
    return xmin


def exp_subgrad():
    strage_func = StrangeFunc(A)
    xmin, loss_history = run_subgrad(strage_func, x0, project, max_iter=2000, lr=lr)
    plot(loss_history, "subgradient method loss/iteration", "./subgrad_n={}_m={}_mu={:.4f}_lr={}.png".format(n, m, mu, lr))
    return xmin


if __name__ == "__main__":
    xmin2 = exp_subgrad()
