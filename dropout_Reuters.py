import numpy as np
import pandas as pd
from keras.datasets import reuters #데이터셋  수치화 , 토픽 46가지, softmax
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Activation, Dropout


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) # 빈도수 많은 데이터  10000개


word_index = reuters.get_word_index()
reverse_word_index = dict(
                        [(value,key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#print(word_index.items()) #딕션너리 형태


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    print(results.shape)    # check!!
    return results


x_train = vectorize_sequences(train_data)
print(len(x_train))
x_test = vectorize_sequences(test_data)

#print(x_train[0])
#print(x_test[0])


# ont-hot-encoding
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#print(one_hot_train_labels[0])


from keras import models
from keras import layers

model = models.Sequential()
model = Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  #samples 개수
model.add(Dropout(0.2))
model.add(layers.Dense(1000, activation='relu'))  #samples 개수
model.add(Dropout(0.2))
model.add(layers.Dense(400, activation='relu'))  #samples 개수
model.add(Dropout(0.2))
model.add(Dense(46, activation='softmax'))
#model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  #samples 개수
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
y_val = one_hot_train_labels[:1000]

X_train = x_train[1000:]
Y_train = one_hot_train_labels[1000:]


history = model.fit(X_train,
                    Y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))


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
