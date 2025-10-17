import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.models.symptom_classifier import SymptomClassifier
from src.models.drug_recommender import DrugRecommender

class Trainer:
    def __init__(self, model, data_loader, epochs=10, batch_size=32):
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        X, y = self.data_loader.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def suggest_medications(self, symptoms):
        classified_symptoms = self.model.predict(symptoms)
        drug_recommender = DrugRecommender()
        return drug_recommender.recommend(classified_symptoms)