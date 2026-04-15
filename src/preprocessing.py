import re
import string
import nltk
from nltk.corpus import stopwords

#Downloading the stopwords
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Cleans raw text by removing punctuation, gibberish, and stopwords.
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
    words_list = [word for word in text.split() if word not in STOP_WORDS]

    return  " ".join(words_list)

def encode_labels(df, label_column):
    """
    Encodes categorical labels into binary integers.
    Mapping: 'Spam' -> 1, 'Not Spam' -> 0
    """
    mapping = {'Spam': 1,'Not Spam': 0}
    df[label_column] = df[label_column].map(mapping)
    return df

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

if __name__ == "__main__":
    test_messages = [
        "WINNER!! Call 0800-123-456 today to claim your $1000 prize!",

        "Hey, check out this link: http://example.com/xyz \n\n It is awesome 100%",

        "hello-world...how are you doing today? call me at 9pm.",
    ]

    print("--- Testing clean_text function ---\n")
    for i, msg in enumerate(test_messages):
        print(f"Original {i + 1}: {msg}")
        print(f"Cleaned  {i + 1}: {clean_text(msg)}")
        print("-" * 50)
