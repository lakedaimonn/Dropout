from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Activation, Dropout

(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784).astype('float64') / 255
X_validation = X_validation.reshape(X_validation.shape[0], 784).astype('float64') / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_validation = np_utils.to_categorical(Y_validation, 10)

model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, input_dim=784, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400, input_dim=784, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) # 10은 0-9까지 숫자

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

#model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),           epochs=100, batch_size=200, verbose=0,          callbacks=[cb_checkpoint, cb_early_stopping])


history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=10, batch_size=500)

print('\nAccuracy: {:.4f}'.format(model.evaluate(X_validation, Y_validation)[1]))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
