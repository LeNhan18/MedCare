import numpy as np
import random

def augment_text(text):
    """Augment text data by applying random transformations."""
    transformations = [
        lambda x: x.replace("a", "e"),  # Example transformation
        lambda x: x.replace("e", "i"),
        lambda x: x.replace("i", "o"),
        lambda x: x.replace("o", "u"),
        lambda x: x + " " + random.choice(["is", "can be", "may be"]) + " related to symptoms.",
    ]
    
    augmented_texts = [transformation(text) for transformation in transformations]
    return augmented_texts

def augment_symptom_data(symptom_data):
    """Apply data augmentation techniques to symptom data."""
    augmented_data = []
    for symptom in symptom_data:
        augmented_data.extend(augment_text(symptom))
    return augmented_data

def random_sample(data, sample_size):
    """Randomly sample a subset of data."""
    return random.sample(data, min(sample_size, len(data)))