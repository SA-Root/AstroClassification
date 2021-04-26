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
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class AstroModel:

    def __init__(self, input_shape=(2600, 1, 1)):
        self.x = np.load('data/first_train_data_x.npy',
                         allow_pickle=True).reshape([10000, 2600, 1, 1]).astype(np.float32)
        self.y = np.load('data/first_train_data_y.npy',
                         allow_pickle=True)
        tmp = []
        for i in self.y:
            if i == 'star':
                t = [1, 0, 0, 0]
            elif i == 'unknown':
                t = [0, 1, 0, 0]
            elif i == 'galaxy':
                t = [0, 0, 1, 0]
            else:
                t = [0, 0, 0, 1]
            tmp.append(t)
        self.y = np.array(tmp)
        # self.x_train, self.x_val, self.y_train, self.y_val = model_selection.train_test_split(
        #     self.x, self.y, train_size=7000)
        self.Build(ishape=input_shape)
        pass

    def Build(self, ishape):
        iL = tkl.Input(shape=ishape, dtype=self.x.dtype, name='InputLayer')
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
            verbose=1,
            save_weights_only=True,
            period=3)
        self.model.fit(x=self.x, y=self.y, batch_size=64, epochs=1000,
                       validation_split=0.3, shuffle=True,
                       workers=4, use_multiprocessing=True,
                       callbacks=[checkpoint])
        pass

    def Predict(self, x):
        pass

    def LoadData(self, pathx, pathy):
        pass

    def ViewModel(self):
        self.model.summary()
