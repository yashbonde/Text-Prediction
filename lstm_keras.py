#import tensorflow as tf
#from tensorflow.contrib.rnn import LSTMCell

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
import numpy as np

import data_ret

'''
My implementation would be as following:
All the words would depend on the 4 previous words.
1) This would require me preparing a custom data set, that would support this. 
2) Also fetch large number of word pairs and pad them '0' so that they would perform better
3) This ia implemenatable in Keras itself.
'''
# some params:
input_length = n_grams = 64
valid_part = 0.2

# Making data for training and validation
data, id2char, char2id = data_ret.getDataChars(n_grams = n_grams)
data = np.array(data)
np.random.shuffle(data)

'''
# This is the custum validation to be used for TF implementations
# creating training data
training_data = data[int(valid_part*len(data)):]
training_labels = training_data[:,[-1]]
training_data = training_data[:, :n_grams]
# creating validation data
valid_data = data[:int(valid_part*len(data))]
valid_labels = valid_data[:,[-1]]
valid_data = valid_data[:, :n_grams]
'''

data = data[:, :n_grams]

# making labels
labels = data[:, -1]
lables_final = np.zeros([len(labels), max(lables)+1])
for i in range(len(lables_final)):
	lables_final[i][int(labels[i])] = 1.0

'''
ML Part
'''

model = Sequential()
model.add(Embedding(len(id2char), 128, input_length = input_length))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dense(len(id2char), activation = 'softmax'))
print(model.summary())

model.compile('RMSProp', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(data, lables_final, epochs = 1, batch_size = 5000, validation_split = 0.2)

input_string = u'No part of this book may be reproduced or transmitted in any form or by any means'
input_string = input_string[:50]
print(input_string, end = '')

# Encoding
for i in range(len(input_string)):
	input_string[i] = char2id[input_string[i]]

for i in range(1000):
	y = model.predict(input_string)
	y = np.argmax(y)
	print(id2char[int(y)], end = '')

	# making the new input_string
	input_string = np.roll(input_string, -1)
	input_string[-1] = y
