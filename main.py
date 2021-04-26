import AstroModel as AM
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

if __name__ == '__main__':

    # a = np.array([0.0012, 0.00012312, 0.98712, 0.000124])
    # b = np.array([0.00041, 0.900014, 0.000324, 0.00124])
    # c = K.argmax(K.round(a)).numpy()
    # print(c > 0.3)

    model = AM.AstroModel()
    # model.ViewModel()
    model.Fit()
    # model.LoadAndTest()
