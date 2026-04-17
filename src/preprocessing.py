import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text):
    """
    Cleans raw text by removing URLs, punctuation, and stopwords (using sklearn).
    Returns a single string of cleaned words.
    """
    if text is None:
        return ""
    # Text to lowercase:
    text = text.lower()

    # Pattern to match urls
    pattern = r'https?://\S+'
    text = re.sub(pattern, '', text)

    # Removing punctuation and newline:
    text = re.sub(r'[^\w\s]',' ',text)
    text = text.strip()

    # getting only words that are not in the stop words
    words_list = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]

    return  " ".join(words_list)

from sklearn.feature_extraction.text import CountVectorizer

def vectorize_data(train_texts, test_texts):
    """
    Converts text to binary vectors (0 or 1) based on word presence.
    Uses CountVectorizer with binary=True as per assignment requirements.
    """
    vectorizer = CountVectorizer(max_features=5000,binary=True)

    #Fit and Transform on the train text
    X_train_transformed = vectorizer.fit_transform(train_texts)

    # Transform on the test text
    X_test_transformed = vectorizer.transform(test_texts)

    return  X_train_transformed, X_test_transformed, vectorizer

