import numpy as np
import os
import matplotlib.pyplot as plt


## load data
# pickle data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# split data and labels
def load_CFIR10_btch(file_name):
    #### load per batch
    pic_file = unpickle(file_name)
    X = pic_file['data']
    Y = pic_file['labels']
    return X, Y


def load_CFIR10(f):
    Xs = []
    Ys = []
    for i in range(1, 6):
        ef = os.path.join(f, 'data_batch_' + str(i))
        X, Y = load_CFIR10_btch(ef)
        Xs.append(X)
        Ys.append(Y)
    X_train = np.concatenate(Xs)
    Y_train = np.concatenate(Ys)
    test = os.path.join(f, 'test_batch')
    X_test, Y_test = load_CFIR10_btch(test)
    m = X_train.mean()
    v = X_train.std()
    X_train = (X_train - m) / v
    X_test = (X_test - m) / v
    return X_train, Y_train, X_test, Y_test


# #reshape data to fit model
X_train, Y_train, X_test, Y_test = load_CFIR10('cifar-10-batches-py')

X_train = X_train.reshape(50000, 32, 32, 3)
X_test = X_test.reshape(10000, 32, 32, 3)

from keras.utils import to_categorical

# one-hot encode target column
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

from keras import layers
from keras import backend as K

class MinPooling2D(layers.MaxPooling2D):


  def __init__(self, pool_size=(2, 2), strides=None,
               padding='valid', data_format=None, **kwargs):
    super(MaxPooling2D, self).__init__(pool_size, strides, padding,
                                       data_format, **kwargs)

  def pooling_function(inputs, pool_size, strides, padding, data_format):
    return -K.pool2d(-inputs, pool_size, strides, padding, data_format,pool_mode='max')


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, AveragePooling2D,Dropout

# create model
model = Sequential()
# add model layers
model.add(Conv2D(20, kernel_size=3, strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from keras.models import Model

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[1].reshape(1, 32, 32, 3))


def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index])
            activation_index += 1


# train the model
history = model.fit(X_train, Y_train, validation_split=0.33, batch_size=100, epochs=3)
display_activation(activations, 3, 3, 1)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()