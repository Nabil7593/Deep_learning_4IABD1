import keras
import numpy as np
import tensorflow as tf


def my_softmax(x: keras.KerasTensor):
    x_max = keras.ops.max(x, axis=1, keepdims=True)
    negative_x = x - x_max
    return keras.ops.exp(negative_x) / keras.ops.sum(keras.ops.exp(negative_x), axis=1, keepdims=True)


def my_categorical_cross_entropy(y_true: keras.KerasTensor, y_pred: keras.KerasTensor):
    loss = -keras.ops.mean(
        keras.ops.sum(y_true * keras.ops.log(y_pred + tf.constant(1e-7, dtype=tf.float32)), axis=1, keepdims=True))
    return loss


def manual_exp_softmax(optimizer='SGD'):
    for seed in [42, 51, 89, 66, 16]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        keras.utils.set_random_seed(seed)

        model = keras.models.Sequential([
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=my_softmax),
        ]
        )

        x_test = (x_test - np.mean(x_train)) / np.std(x_train)
        x_train = (x_train - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        batch_size = 32
        alpha = 0.01 if optimizer == "SGD" or optimizer == "SGD with Classic Momentum" else 0.001
        gamma1 = 0.8 if optimizer == "SGD with Classic Momentum" else 0.9
        gamma2 = 0.999
        exp_name = f"manual_softmax_and_cat_xentropy_seed_{seed}_{optimizer}"

        tb_writer = tf.summary.create_file_writer(f'./logs/{exp_name}/train')
        tb_writer_validation = tf.summary.create_file_writer(f'./logs/{exp_name}/val')

        train_dataset_indices = np.arange(0, len(x_train))

        velocities = [np.zeros_like(w) for w in model.trainable_variables]
        squared_ema_gradients = [np.zeros_like(w) for w in model.trainable_variables]

        optimizer_step_count = 0
        for epoch in range(100):
            mean_loss = 0.0
            mean_accuracy = 0.0
            np.random.shuffle(train_dataset_indices)
            x_train = x_train[train_dataset_indices]
            y_train = y_train[train_dataset_indices]
            steps_count = len(x_train) // 32
            for step in range(steps_count):
                batch_x_train = x_train[step * batch_size: (step + 1) * batch_size]
                batch_y_train = y_train[step * batch_size: (step + 1) * batch_size]
                mean_accuracy, mean_loss, velocities, squared_ema_gradients = train_step(
                    optimizer,
                    alpha,
                    mean_accuracy,
                    mean_loss, model,
                    tf.constant(batch_x_train, dtype=tf.float32),
                    tf.constant(batch_y_train, dtype=tf.float32),
                    tf.constant(gamma1, dtype=tf.float32),
                    tf.constant(gamma2, dtype=tf.float32),
                    velocities,
                    squared_ema_gradients,
                    tf.constant(optimizer_step_count, dtype=tf.float32),
                )
                optimizer_step_count += 1
            mean_loss /= steps_count
            mean_accuracy /= steps_count

            y_pred_test = model.predict(x_test)

            mean_val_accuracy = keras.ops.mean(keras.metrics.categorical_accuracy(y_test, y_pred_test))

            tb_writer_validation.set_as_default()
            tf.summary.scalar("epoch_categorical_accuracy", mean_val_accuracy, epoch)
            tf.summary.flush(tb_writer_validation)

            tb_writer.set_as_default()
            tf.summary.scalar("epoch_loss", mean_loss, epoch)
            tf.summary.scalar("epoch_categorical_accuracy", mean_accuracy, epoch)
            tf.summary.flush(tb_writer)
            print(f'Epoch : {epoch}, Loss : {mean_loss}, Accuracy : {mean_accuracy}')


@tf.function
def train_step(optimizer, alpha, mean_accuracy, mean_loss, model, batch_x_train, batch_y_train,
               gamma1,
               gamma2,
               velocities,
               squared_ema_gradients,
               step):
    with tf.GradientTape() as tape:
        batch_y_pred = model(batch_x_train)
        loss = my_categorical_cross_entropy(batch_y_train, batch_y_pred)
    mean_loss += keras.ops.mean(loss)
    mean_accuracy += keras.ops.mean(keras.metrics.categorical_accuracy(batch_y_train, batch_y_pred))
    grads = tape.gradient(loss, model.trainable_variables)
    if optimizer == 'SGD with Classic Momentum':
        velocities = [gamma1 * v - alpha * grad for (v, grad) in zip(velocities, grads)]
    if optimizer == 'SGD with EMA Momentum' or optimizer == 'Adam':
        velocities = [gamma1 * v + (1 - gamma1) * grad for (v, grad) in zip(velocities, grads)]
    if optimizer == "RMS Prop" or optimizer == 'Adam':
        squared_ema_gradients = [gamma2 * s + (1 - gamma2) * grad * grad for (s, grad) in
                                 zip(squared_ema_gradients, grads)]
    for (w, grad, v, s) in zip(model.trainable_variables, grads, velocities, squared_ema_gradients):
        if optimizer == 'SGD':
            w.assign(w - alpha * grad)
        elif optimizer == 'SGD with Classic Momentum':
            w.assign(w + v)
        elif optimizer == 'SGD with EMA Momentum':
            w.assign(w - alpha * v)
        elif optimizer == 'RMS Prop':
            w.assign(w - alpha * grad / keras.ops.sqrt(s + tf.constant(1e-7, dtype=tf.float32)))
        elif optimizer == 'Adam':
            v_hat = v / (1 - gamma1 ** (step + tf.constant(1, dtype=tf.float32)))
            s_hat = s / (1 - gamma2 ** (step + tf.constant(1, dtype=tf.float32)))
            w.assign(w - alpha * v_hat / keras.ops.sqrt(s_hat + tf.constant(1e-7, dtype=tf.float32)))
        else:
            raise "OMG BBQ !"
    return mean_accuracy, mean_loss, velocities, squared_ema_gradients


def exp_tanh():
    for seed in [42, 51, 89, 66, 16]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        keras.utils.set_random_seed(seed)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=keras.activations.tanh),
        ]
        )

        x_test = (x_test - np.mean(x_train)) / np.std(x_train)
        x_train = (x_train - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10) * 2.0 - 1.0
        y_test = keras.utils.to_categorical(y_test, 10) * 2.0 - 1.0

        exp_name = f"tanh_and_mse_seed_{seed}"

        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(x_train, y_train, 32, 100,
                  callbacks=[keras.callbacks.TensorBoard(log_dir=f'./logs/{exp_name}')])


def exp_softmax():
    for seed in [42, 51, 89, 66, 16]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        keras.utils.set_random_seed(seed)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=keras.activations.softmax),
        ]
        )

        x_test = (x_test - np.mean(x_train)) / np.std(x_train)
        x_train = (x_train - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        exp_name = f"softmax_and_cat_xentropy_seed_{seed}"

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(x_train, y_train, 32, 100,
                  callbacks=[keras.callbacks.TensorBoard(log_dir=f'./logs/{exp_name}')])


def exp_softmax_with_mse():
    for seed in [42, 51, 89, 66, 16]:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        keras.utils.set_random_seed(seed)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=keras.activations.softmax),
        ]
        )

        x_test = (x_test - np.mean(x_train)) / np.std(x_train)
        x_train = (x_train - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        exp_name = f"softmax_and_mse_seed_{seed}"

        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(x_train, y_train, 32, 100,
                  callbacks=[keras.callbacks.TensorBoard(log_dir=f'./logs/{exp_name}')])


def run():
    exp_tanh()
    exp_softmax()
    exp_softmax_with_mse()
    manual_exp_softmax('SGD')
    manual_exp_softmax('SGD with Classic Momentum')
    manual_exp_softmax('SGD with EMA Momentum')
    manual_exp_softmax('RMS Prop')
    manual_exp_softmax('Adam')


if __name__ == "__main__":
    run()
