import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Activation, Dropout

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)


word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value,key) for (key, value) in word_index.items()]
)
decode_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]]
)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    print(results.shape) ## -> input_shape에 그대로 들어가야
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 0, 1, 스칼라 값으로 저장되어 있음
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = Sequential()
model.add(Dense(16,activation='relu', input_shape=(10000,)))
model.add(Dropout(0.2))
model.add(Dense(40,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) ## 1?

#model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  #samples 개수
#model.add(Dropout(0.2))


model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data=(x_val, y_val))


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
