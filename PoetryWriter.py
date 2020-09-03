from tensorflow.keras import models, layers, utils
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


checkpoint_dir = r'C:\Users\rsaroj\PycharmProjects\NLPTensorflow\rnn_chk_point'
checkpoint_dir2 = r'C:\Users\rsaroj\PycharmProjects\NLPTensorflow\rnn_chk_point2'
checkpoint_dir3 = r'C:\Users\rsaroj\PycharmProjects\NLPTensorflow\rnn_chk_point3'

path_to_file = utils.get_file('shakespeare.txt',
                              'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# print(path_to_file)

poetryData = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# print(poetryData[:250])

# Create Vocabulary
vocab = sorted(set(poetryData))

# Create mapping dict from char to idx and idx to char mappings
char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = np.array(vocab)

# use the mapping to convert the whole training poetry dataset into numbers based on mappings (because AI only understands numbers)
text2idx = np.array([char2idx.get(char) for char in poetryData])


sequnce_size = 100
vocab_size = len(vocab)

# Note we have vocab_size=65, Training input sequnce_size=100, batch_size=64,
# Every training batch shape will be of shape (batch_size, sequnce_size) i.e (64, 100)
batch_size = 64
buffer_size = 10000
steps_per_epoch = len(text2idx) / sequnce_size + 1

embedding_dims = 256
rnn_units = 1024


# Convert the numpy based array to tensor slices
data = tf.data.Dataset.from_tensor_slices(text2idx)

# print(" ".join([idx2char[x.numpy()] for x in data.take(13)]))

# batch the tensors
sequences = data.batch(batch_size=sequnce_size + 1, drop_remainder=True)


def create_train_and_test(chunk):
    train_data = chunk[:-1]
    target_data = chunk[1:]
    return train_data, target_data


dataset = sequences.map(create_train_and_test)

# Batch and Shuffle the dataset (it contains train and test data) return type will be batchedDataset every batch block of shape (batch_size, sequnce_size, vocab_size) i.e (64, 100)
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

def display_inputs_n_expected_outputs():
    # Display input character and expected output character
    for train_data, target_data in dataset.take(1):
        train_data = train_data.numpy()
        target_data = target_data.numpy()

    for i, (trainIDX, targetIDx) in enumerate( zip(train_data[:5], target_data[:5])):
        print("Step {:4d}".format(i))
        print("train input: {} ({:s}) ".format(trainIDX, repr(idx2char[trainIDX])))
        print("target expected: {} ({:s}) ".format(targetIDx, repr(idx2char[targetIDx])))

#display_inputs_n_expected_outputs()


def makeModel(batch_size):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dims,batch_input_shape=[batch_size, None]),
        layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    return model

#makeModel(batch_size).summary()

def TestModel_outputshapes_and_outputs():
    model = makeModel()
    for train_data, target_data in dataset.take(1):

        # input training batch shape (batch_size, sequnce_size) i.e (64, 100)
        # train_batch_pred predicted output shape will be (batch_size, sequnce_size, vocab_size) i.e (64, 100, 65)
        train_batch_pred = model(train_data)

        # interpret model output
        samples = tf.random.categorical(logits=train_batch_pred[0], num_samples=1)

        samples = tf.squeeze(samples, axis=-1).numpy()


        train_char = "".join(idx2char[train_data[0].numpy()])
        print(train_char)

        predicted_char = "".join(idx2char[samples])
        print(predicted_char)

    return train_batch_pred





def loss(labels, logits):
    return tf.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True)


def compile_model():
    model = makeModel(batch_size)
    model.compile(optimizer='adam', loss=loss)
    return model


def configure_checkpoint_callback():

    check_folder = os.path.join(checkpoint_dir3, "rnn_ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_folder,
                                                             save_weights_only=True)
    return checkpoint_callback



def train_model():

    # Get the compiled model
    model = compile_model()

    # Fit the model to the data (Training)
    history = model.fit(dataset, epochs=10, callbacks=[configure_checkpoint_callback()])


#train_model()


def resume_from_checkpoint_training():
    model = makeModel(batch_size)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir2))
    initial_epoch = 15
    epoch = 40
    model.compile(optimizer='adam', loss=loss)
    model.fit(dataset, epochs=epoch, initial_epoch=initial_epoch, callbacks=[configure_checkpoint_callback()])

#resume_from_checkpoint_training()


def Load_Model_with_saved_weights():
    batch_sz = 1
    model = makeModel(batch_sz)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir3))
    model.build(tf.TensorShape([1, None]))
    return model


def AiShakespear():

    model = Load_Model_with_saved_weights()
    num_char = 1000
    start_text = u"ROMEO: "
    temperature = 1.0
    text_gen = []
    char2id = [ char2idx[x] for x in start_text]
    input_eval = tf.expand_dims(char2id, axis=0)

    model.reset_states()



    for x in range(num_char):

        pred = model.predict(input_eval)
        pred = tf.squeeze(pred, axis=0)
        pred = pred/temperature

        pred = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([pred], axis=0)

        text_gen.append(idx2char[pred])

    print(start_text+ "".join(text_gen))

AiShakespear()

