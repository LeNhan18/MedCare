import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SymptomClassifier:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Embedding(input_dim=10000, output_dim=128, input_length=input_shape[0]),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, x):
        return self.model.predict(x)