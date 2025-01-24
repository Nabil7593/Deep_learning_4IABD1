import keras
import numpy as np

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

        x_train = (x_train - np.mean(x_train)) / np.std(x_train)
        x_test = (x_test - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10) * 2.0 - 1.0
        y_test = keras.utils.to_categorical(y_train, 10) * 2.0 - 1.0

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

        x_train = (x_train - np.mean(x_train)) / np.std(x_train)
        x_test = (x_test - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_train, 10)

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

        x_train = (x_train - np.mean(x_train)) / np.std(x_train)
        x_test = (x_test - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_train, 10)

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

if __name__ == "__main__":
    run()
