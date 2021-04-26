import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow.keras.initializers as tki
import tensorflow.keras.optimizers as tko
import tensorflow.keras.losses as tkls
import tensorflow.keras.callbacks as tkc
from tensorflow.keras import backend as K
import numpy as np


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class AstroModel:

    def __init__(self, input_shape=(2600, 1, 1)):
        self.xt = np.load('data/xt.npy',
                          allow_pickle=True)
        self.yt = np.load('data/yt.npy',
                          allow_pickle=True)
        self.xv = np.load('data/xv.npy',
                          allow_pickle=True)
        self.yv = np.load('data/yv.npy',
                          allow_pickle=True)
        self.Build(ishape=input_shape)
        pass

    def Build(self, ishape):
        iL = tkl.Input(shape=ishape, dtype='float32', name='InputLayer')
        X3 = tkl.Conv2D(32, kernel_size=(3, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X3 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X3)
        X3 = tkl.Conv2D(32, kernel_size=(3, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X3)
        X3 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X3)
        X3 = tkl.Conv2D(32, kernel_size=(3, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X3)
        X3 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X3)
        X3 = tkl.Conv2D(32, kernel_size=(3, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X3)
        X3 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X3)

        X5 = tkl.Conv2D(32, kernel_size=(5, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X5 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X5)
        X5 = tkl.Conv2D(32, kernel_size=(5, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X5)
        X5 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X5)
        X5 = tkl.Conv2D(32, kernel_size=(5, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X5)
        X5 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X5)
        X5 = tkl.Conv2D(32, kernel_size=(5, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X5)
        X5 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X5)

        X7 = tkl.Conv2D(32, kernel_size=(7, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X7 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X7)
        X7 = tkl.Conv2D(32, kernel_size=(7, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X7)
        X7 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X7)
        X7 = tkl.Conv2D(32, kernel_size=(7, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X7)
        X7 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X7)
        X7 = tkl.Conv2D(32, kernel_size=(7, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X7)
        X7 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X7)

        X9 = tkl.Conv2D(32, kernel_size=(9, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X9 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X9)
        X9 = tkl.Conv2D(32, kernel_size=(9, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X9)
        X9 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X9)
        X9 = tkl.Conv2D(32, kernel_size=(9, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X9)
        X9 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X9)
        X9 = tkl.Conv2D(32, kernel_size=(9, 1), strides=(1, 1),
                        padding='same', activation='relu',
                        kernel_initializer=tki.glorot_uniform(seed=921212))(X9)
        X9 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X9)

        X11 = tkl.Conv2D(32, kernel_size=(11, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X11 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X11)
        X11 = tkl.Conv2D(32, kernel_size=(11, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X11)
        X11 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X11)
        X11 = tkl.Conv2D(32, kernel_size=(11, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X11)
        X11 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X11)
        X11 = tkl.Conv2D(32, kernel_size=(11, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X11)
        X11 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X11)

        X13 = tkl.Conv2D(32, kernel_size=(13, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X13 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X13)
        X13 = tkl.Conv2D(32, kernel_size=(13, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X13)
        X13 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X13)
        X13 = tkl.Conv2D(32, kernel_size=(13, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X13)
        X13 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X13)
        X13 = tkl.Conv2D(32, kernel_size=(13, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X13)
        X13 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X13)

        X15 = tkl.Conv2D(32, kernel_size=(15, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X15 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X15)
        X15 = tkl.Conv2D(32, kernel_size=(15, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X15)
        X15 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X15)
        X15 = tkl.Conv2D(32, kernel_size=(15, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X15)
        X15 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X15)
        X15 = tkl.Conv2D(32, kernel_size=(15, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X15)
        X15 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X15)

        X17 = tkl.Conv2D(32, kernel_size=(17, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(iL)
        X17 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X17)
        X17 = tkl.Conv2D(32, kernel_size=(17, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X17)
        X17 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X17)
        X17 = tkl.Conv2D(32, kernel_size=(17, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X17)
        X17 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X17)
        X17 = tkl.Conv2D(32, kernel_size=(17, 1), strides=(1, 1),
                         padding='same', activation='relu',
                         kernel_initializer=tki.glorot_uniform(seed=921212))(X17)
        X17 = tkl.AveragePooling2D(pool_size=(3, 1), strides=(3, 1))(X17)

        X = tkl.concatenate([X3, X5, X7, X9, X11, X13, X15, X17], axis=3)
        X = tkl.Flatten()(X)
        X = tkl.Dense(1024, activation='relu',
                      kernel_initializer=tki.glorot_uniform(seed=(921212)))(X)
        X = tkl.Dense(512, activation='relu',
                      kernel_initializer=tki.glorot_uniform(seed=(921212)))(X)
        X = tkl.Dense(256, activation='tanh',
                      kernel_initializer=tki.glorot_uniform(seed=(921212)))(X)
        X = tkl.Dense(4, activation='softmax',
                      kernel_initializer=tki.glorot_uniform(seed=(921212)))(X)

        self.model = tf.keras.Model(inputs=iL, outputs=X, name='AstroNet')
        self.model.compile(optimizer=tko.SGD(learning_rate=0.001),
                           metrics=[f1],
                           loss=tkls.categorical_crossentropy)
        pass

    def Fit(self):
        filepath = "model_{epoch:d}.h5"
        checkpoint = tkc.ModelCheckpoint(
            filepath=filepath,
            monitor='f1',
            save_weights_only=True,
            save_freq=5)
        self.model.fit(x=self.xt, y=self.yt, batch_size=256, epochs=1000,
                       validation_data=(self.xv, self.yv), shuffle=True,
                       use_multiprocessing=True,
                       callbacks=[checkpoint])
        pass

    def Predict(self, x):
        return self.model.predict(x)

    def LoadAndValidate(self):
        xv = np.load('data/xv.npy',
                     allow_pickle=True)
        yv = np.load('data/yv.npy',
                     allow_pickle=True)
        self.model.load_weights('models/model_1000.h5')
        res = self.Predict(xv)
        print('f1 score: {0}'.format(self.f1(yv, res)))
        pass

    def ViewModel(self):
        self.model.summary()

    def f1(self, y_true, y_pred):

        def recall():
            true_positives = 0
            possible_positives = 0
            for i in range(0, 10000):
                true_positives += K.argmax(
                    K.round(y_true[i] * y_pred[i])).numpy() > 0.7
                possible_positives += K.argmax(
                    K.round(y_true[i])).numpy() > 0.7
            rec = true_positives*1.0 / (possible_positives + K.epsilon())
            return rec

        def precision():
            true_positives = 0
            predicted_positives = 0
            for i in range(0, 10000):
                true_positives += K.argmax(
                    K.round(y_true[i] * y_pred[i])).numpy() > 0.7
                predicted_positives += K.argmax(
                    K.round(y_pred[i])).numpy() > 0.7
            precision = true_positives * 1.0 / \
                (predicted_positives + K.epsilon())
            return precision
        precision = precision()
        rec = recall()
        return 2*((precision*rec)/(precision+rec+K.epsilon()))
