import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import os, random

def reset_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    dataset = tf.keras.datasets.boston_housing
    train, test = dataset.load_data()
    # 学習用データセット
    x_train = np.array(train[0], np.float32)
    t_train = np.array(train[1], np.int32)

    # テスト用データセット
    x_test = np.array(test[0], np.float32)
    t_test = np.array(test[1], np.uint32)
    
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(13, )))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])
    model.summary()
    history = model.fit(x_train, t_train, 
                    epochs=30,
                    batch_size=32,
                    validation_data=(x_test, t_test))
    score = model.evaluate(x_test, t_test)

    # 結果の可視化
    result = pd.DataFrame(history.history)

    # 目的関数の可視化
    result[['loss', 'val_loss']].plot()
    # 評価指標の可視化
    result[['mae', 'val_mae']].plot()
    plt.show()

if __name__ == '__main__':
    main()