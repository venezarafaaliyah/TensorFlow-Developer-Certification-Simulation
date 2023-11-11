# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf
import tensorflow_datasets as tfds


def solution_B2():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # DEFINE YOUR MODEL HERE
    model = tf.keras.Sequential([
        # Set the input shape to (28, 28, 1), kernel size=3, filters=16 and use ReLU activation,
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),

        tf.keras.layers.MaxPooling2D(),

        # Set the number of filters to 32, kernel size to 3 and use ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

        # Flatten the output layer to 1 dimension
        tf.keras.layers.Flatten(),

        # Add a fully connected layer with 64 hidden units and ReLU activation
        tf.keras.layers.Dense(units=64, activation='relu'),

        # End with 10 Neuron Dense, activated by softmax
        # Attach a final softmax classification head
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    # COMPILE MODEL HERE
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.90):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    # TRAIN YOUR MODEL HERE
    history = model.fit(x_train,
                        y_train,
                        batch_size=64,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks=[callbacks])

    import matplotlib.pyplot as plt

    # Plot the model results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")