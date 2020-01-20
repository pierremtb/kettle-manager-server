import os
import tensorflow as tf

class BoilModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.input_shape = (1,)
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
        self.model.fit(X, y)
        self.save()

    def predict(self, X):
        if self.model is None:
            raise FileNotFoundError("Model isn't defined!")
        return self.model.predict(X)