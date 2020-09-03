import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras import layers, models, Model, Sequential
import matplotlib.pyplot as plt
import io
import json

sarcasm_json = open(r'C:\Datasets\SarcasmDataset\Sarcasm_Headlines_Dataset_v2.json', 'r')
sarcasm_json_obj = sarcasm_json.readlines()

headline_list = []
label = []

for x in sarcasm_json_obj:
    headline_list.append((json.loads(x))['headline'])
    label.append((json.loads(x))['is_sarcastic'])

training_size = 20000

train_sentences_list = headline_list[:training_size]
train_label_list = label[:training_size]

test_sentence_list = headline_list[training_size:]
test_label_list = label[training_size:]

vocab = 10000
embeding_dim = 16
maxlen = 120
truncate = 'post'
epoch = 10
padded = 'post'

train_label_arr = np.array(train_label_list)
test_label_arr = np.array(test_label_list)

tokenizer = Tokenizer(num_words=vocab, oov_token='<oov>')
tokenizer.fit_on_texts(train_sentences_list)
word_idx = tokenizer.index_word
train_seq = tokenizer.texts_to_sequences(train_sentences_list)
train_padded = pad_sequences(train_seq, maxlen=maxlen, truncating=truncate, padding=padded)
train_padded = np.array(train_padded)


test_seq = tokenizer.texts_to_sequences(test_sentence_list)
test_padded = pad_sequences(test_seq, maxlen=maxlen)
test_padded = np.array(test_padded)


def graph(history):
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']

    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    n_epoch = range(len(val_acc))

    plt.plot(n_epoch, train_acc, 'r', label='train acc')
    plt.plot(n_epoch, val_acc, 'b', label='val acc')
    plt.title("train vs val acc")
    plt.legend()

    plt.figure()

    plt.plot(n_epoch, train_loss, 'r', label='train loss')
    plt.plot(n_epoch, val_loss, 'b', label='val loss')
    plt.title("train vs val loss")
    plt.legend()

    plt.show()


def makeSimpleModel():
    l1 = layers.Embedding(vocab, embeding_dim, input_length=maxlen)
    l2 = layers.Flatten()
    l3 = layers.Dense(6, activation='relu')
    l4 = layers.Dense(1, activation='sigmoid')

    model = Sequential([l1, l2, l3, l4])
    return model


def makeDoubleLSTMModel():
    l1 = layers.Embedding(vocab, embeding_dim, input_length=maxlen)
    l2 = layers.Bidirectional(layers.LSTM(64, return_sequences=truncate))
    l3 = layers.Bidirectional(layers.LSTM(32))
    l4 = layers.Dense(64, activation='relu')
    l5 = layers.Dense(1, activation='sigmoid')

    model = Sequential([l1, l2, l3, l4, l5])
    return model


def makeCNNModel():
    l1 = layers.Embedding(vocab, embeding_dim, input_length=maxlen)
    l2 = layers.Conv1D(128, 5, activation='relu')
    l3 = layers.GlobalAveragePooling1D()
    l4 = layers.Dense(24, activation='relu')
    l5 = layers.Dense(1, activation='sigmoid')

    model = Sequential([l1, l2, l3, l4, l5])
    return model


def makeprepandtrain():
    model = makeCNNModel()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_padded,
                        train_label_arr,
                        epochs=epoch,
                        validation_data=(test_padded, test_label_arr))

    #model.save('t1-npl-model')

    # graph(history)


makeprepandtrain()
