import re
import string
import nltk
from nltk.corpus import stopwords

#Downloading the stopwords
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Cleans raw text by removing punctuation, digits/gibberish, and stopwords.
    Returns a single string of cleaned words.
    """
    if text is None:
        return ""
    # Text to lowercase:
    text = text.lower()

    # Removing punctuation:
    translator = str.maketrans('','',string.punctuation)
    text = text.translate(translator)

    #Gibberish Removal
    #Pattern to match words with at least one digit
    pattern = r'\b[a-z]*\d[a-z\d]*\b'

    #removing the words that has both letters and numbers
    text = re.sub(pattern,'',text)


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
    #Creating the CountVectorizer object
    vectorizer = CountVectorizer(max_features=5000,binary=True)

    #Fit and Transform on the train text
    X_train_transformed = vectorizer.fit_transform(train_texts)

    # Transform on the test text
    X_test_transformed = vectorizer.transform(test_texts)

    return  X_train_transformed, X_test_transformed, vectorizer

