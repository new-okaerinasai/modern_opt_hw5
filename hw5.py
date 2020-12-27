import matplotlib.pyplot as plt
import tensorflow as tf

from loss import SmoothStrangeFunc, StrangeFunc, project
from run_stm import run_stm
from run_subgradient import run_subgrad

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


tf.random.set_seed(121)
mu = 0.1
A = tf.nn.relu(tf.random.normal((5, 10)))


def exp_stm():
    x0 = tf.random.normal(shape=(10,))
    L_m = 1 / mu * tf.reduce_max(tf.linalg.norm(A, axis=1))
    strage_func = SmoothStrangeFunc(A, mu)
    tf.tensordot(strage_func.A, x0, 1) / strage_func.mu
    xmin, loss_history = run_stm(strage_func, project, x0, max_iter=200, L_m=L_m)
    plt.figure(figsize=(10, 8))
    plt.title("subgradient method loss/iteration")
    plt.xlabel("iteration")
    plt.ylabel("loss, log-scale")
    plt.yscale("log")
    plt.plot(loss_history)
    plt.savefig("./stm.png")
    return xmin


def exp_subgrad():
    x0 = tf.random.normal(shape=(10,))
    strage_func = StrangeFunc(A)
    xmin, loss_history = run_subgrad(strage_func, x0, project, max_iter=2000, lr=1e-2)
    plt.figure(figsize=(10, 8))
    plt.title("subgradient method loss/iteration")
    plt.xlabel("iteration")
    plt.ylabel("loss, log-scale")
    plt.yscale("log")
    plt.plot(loss_history)
    plt.savefig("./subgrad.png")
    return xmin


if __name__ == "__main__":
    xmin2 = exp_stm()
