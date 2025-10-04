import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def normalize_text(text):
    """Normalize the input text by converting to lowercase and removing special characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def tokenize_text(text):
    """Tokenize the input text into words."""
    return word_tokenize(text)

def stem_words(words):
    """Stem the input words using Porter Stemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]