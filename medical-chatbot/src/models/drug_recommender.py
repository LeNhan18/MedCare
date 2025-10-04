import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DrugRecommender:
    def __init__(self, symptom_data, drug_data):
        self.symptom_data = symptom_data
        self.drug_data = drug_data
        self.model = None
        self.label_encoder = LabelEncoder()
        self.prepare_data()

    def prepare_data(self):
        self.drug_data['label'] = self.label_encoder.fit_transform(self.drug_data['drug_name'])
        self.X = self.symptom_data['symptoms']
        self.y = self.drug_data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=1000, output_dim=64))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, epochs=10, batch_size=32):
        self.X_train = pad_sequences(self.X_train, padding='post')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def recommend(self, symptoms):
        symptoms = pad_sequences(symptoms, padding='post')
        predictions = self.model.predict(symptoms)
        recommended_drugs = self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        return recommended_drugs.tolist()