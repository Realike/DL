import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    # x_train取前number数据并向量化
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    x_train = x_train / 255
    x_test = x_test / 255
    x_test = np.random.normal(x_test) # 加入随机噪声 noise

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()

# use activation='relu', x_train accuracy 99.48% -> 100.00%(overfitting), x_test accuracy 94.% -> 96.51%
model.add(Dense(input_dim=28*28, units=500, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=500,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=500,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=10,activation='softmax'))

# use categorical_crossentropy and adam, x_train accuracy 11.% -> 99.48%, x_test accuracy 11.% -> 94.%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=20)

score_train = model.evaluate(x_train, y_train)
print('\nTrain Acc:', score_train[1])

score_test = model.evaluate(x_test, y_test)
print('\nTest Acc:', score_test[1])
