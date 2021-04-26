import tensorflow as tf
import numpy as np


class AstroModel:

    def __init__(self):
        self.model = tf.keras.Sequential()
        self.x = np.load('data/first_train_data_x.npy', allow_pickle=True)
        self.y = np.load('data/first_train_data_y.npy', allow_pickle=True)
        self.Build()
        pass

    def Build(self):
        self.model.add()
        pass

    def Fit(x, y):
        pass

    def Predict(x):
        pass

    def LoadData(pathx, pathy):
        pass
