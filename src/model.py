import os
import tensorflow as tf
import numpy as np


class BoilModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.input_shape = (2,)
        self.load()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mse'])
        self.model = model

    def load(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.create_model()

    def save(self):
        if self.model is None:
            raise FileNotFoundError("Model isn't defined!")
        tf.keras.models.save_model(self.model, self.model_path)

    def train(self, X, y):
        if self.model is None:
            raise FileNotFoundError("Model isn't defined!")
        self.model.fit(X, y, epochs=1000)
        self.save()

    def predict(self, X):
        if self.model is None:
            raise FileNotFoundError("Model isn't defined!")
        return self.model.predict(X)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device_id = 0
    Xy = np.loadtxt("data/out1.csv", delimiter=",")
    X = Xy[:, 0].reshape(-1, 1)
    y = Xy[:, 1].reshape(-1, 1)
    model_path = os.path.join("data", str(device_id))
    boil_model = BoilModel(model_path)
    boil_model.train(X, y)

    x_pred = np.linspace(0, 600, 100)
    y_pred = boil_model.predict(x_pred)
    plt.plot(x_pred, y_pred)
    plt.scatter(X, y)
    plt.show()
    
