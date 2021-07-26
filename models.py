import tensorflow as tf
import matplotlib.pyplot as plt
import os

class Model:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,(3,3),input_shape=[28,28,1],activation="relu"),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64,activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25,activation="softmax")
        ])
        model.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy", optimizer="adam")
        return model

    def fit_model(self, X, Y, cp_callback):
        history = self.model.fit(X, Y, epochs=10, validation_split=0.20, callbacks=[cp_callback])
        return history

    def score(self, X, Y):
        score = self.model.evaluate(X, Y)
        return score

    def describe(self):
        self.model.summary()

    def plots_acc(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def plots_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def load_model(self, checkp):
        self.model.load_weights(filepath=checkp)

    def store_weights(self):
        checkpoint_path = "training_ckpt/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        return checkpoint_path, cp_callback

    def prediction(self, X):
        return self.model.predict(X)

class ASL_Model(Model):

    def __init__(self):

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,(3,3),strides=1,input_shape=[64, 64, 1],activation="relu"),
            tf.keras.layers.Conv2D(64,(3,3),strides = 2, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(128,(3,3),strides = 1,activation="relu"),
            tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation="relu"),
            tf.keras.layers.Conv2D(256, (3, 3), strides=2, activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512,activation="relu"),
            tf.keras.layers.Dense(29,activation="softmax")
        ])
        model.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy", optimizer="adam")
        return model

    def store_weights(self):
        checkpoint_path = "training_ckpt_ASL_VW/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        return checkpoint_path, cp_callback








