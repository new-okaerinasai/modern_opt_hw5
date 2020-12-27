import tensorflow as tf
import tqdm

def solve_xlambda(func):
    pass


def minimize(func, x_0, proj=None):
    """
        Run Adam optimizer to find minimum of a function `func`
         with the initial point `x_0`.
        proj: callable. Function of projection on some set.
    """
    optimizer = tf.optimizers.Adam(1)
    loss_prev = 1e+12
    for i in range(100):
        with tf.GradientTape() as tape:
            tape.watch(x_0)
            loss = func(x_0)
        gradients = tape.gradient(loss, x_0)
        optimizer.apply_gradients(zip([gradients], [x_0]))
        if proj is not None:
            x_0 = tf.Variable(proj(x_0))
        if tf.abs(loss - loss_prev) < 1e-5 * loss:
            break
        loss_prev = loss
    return x_0


def run_stm(func, proj_func, x0, max_iter=10000, L_m=1.):
    v_k = tf.Variable(tf.zeros(shape=x0.shape))
    x_k = tf.Variable(tf.zeros(shape=x0.shape))
    grad_history = []
    loss_history = []
    progress_bar = tqdm.tqdm(range(max_iter))
    try:
        for k in progress_bar:
            with tf.GradientTape() as tape:
                y_k = k / (k + 2) * x_k + 2 / (k + 2) * v_k
                tape.watch(y_k)
                res = func(y_k)
            grad_k = tape.gradient(res, y_k)
            grad_history.append(grad_k)
            minimizee = lambda v: tf.tensordot(sum((i + 1) / 2 * grad_history[i] for i in range(len(grad_history))), v - x0,
                                               1) + L_m * tf.linalg.norm(v - x0)
            v_0 = tf.Variable(tf.nn.relu(v_k + tf.random.normal(shape=v_k.shape)))
            v_k = minimize(minimizee, v_0, proj_func)
            x_k = (k / (k + 2)) * x_k + (2 / (k + 2)) * v_k
            progress_bar.set_description("Loss_value = {:.4f}".format(func(x_k).numpy()))
            # print(tf.reduce_sum(tf.math.log(x_k)))
            loss_history.append(func(x_k).numpy())
    except KeyboardInterrupt:
        print("Cancelled by user")
    return x_k, loss_history


if __name__ == "__main__":
    pass
