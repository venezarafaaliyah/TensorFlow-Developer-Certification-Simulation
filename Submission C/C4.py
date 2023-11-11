# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    # YOUR CODE HERE
    with open('sarcasm.json', 'r') as json_read:
        json_data = json.load(json_read)

    for item in json_data:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    training_sentences = sentences[:training_size]
    training_labels = np.array(labels[:training_size])
    test_sentences = sentences[training_size:]
    test_labels = np.array(labels[training_size:])

    # Fit your tokenizer with training data
    # YOUR CODE HERE
    tokenizer = Tokenizer(
        num_words = vocab_size,
        oov_token = oov_tok
    )
    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(
        training_sequences,
        maxlen = max_length,
        padding = padding_type,
        truncating = trunc_type
    )

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(
        test_sequences,
        maxlen = max_length,
        padding = padding_type,
        truncating = trunc_type
    )

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.75 and logs.get('val_accuracy') > 0.75):
                print("\nAccuracy is more than 75%, training stopped!")
                self.model.stop_training = True

    callback = myCallback()

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[max_length]),
        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    model.fit(training_padded,
              training_labels,
              epochs = 500,
              validation_data = (test_padded, test_labels),
              callbacks = [callback])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4S4.h5")
