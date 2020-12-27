import numpy as np
import tensorflow as tf
import tqdm


def run_subgrad(func, x_0, proj=None, lr=1e-2, max_iter=1000):
    """
        Run Adam optimizer to find minimum of a function `func`
         with the initial point `x_0`.
        proj: callable. Function of projection on some set.
    """
    loss_prev = 1e+12
    progress_bar = tqdm.tqdm(range(max_iter))
    loss_history = []
    try:
        for i in progress_bar:
            with tf.GradientTape() as tape:
                tape.watch(x_0)
                loss = func(x_0)
            gradients = tape.gradient(loss, x_0)
            x_0 = x_0 - lr * gradients
            if proj is not None:
                x_0 = tf.Variable(proj(x_0))
            if tf.abs(loss - loss_prev) < 1e-8 * loss:
                break
            loss_prev = loss
            progress_bar.set_description("Loss_value = {:.4f}".format(loss.numpy()))
            loss_history.append(func(x_0).numpy())
    except KeyboardInterrupt:
        print("Cancelled by user")
    return x_0, loss_history
