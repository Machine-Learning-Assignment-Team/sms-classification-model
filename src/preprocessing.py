import re
import string
import nltk
from nltk.corpus import stopwords

#Downloading the stopwords
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Inputs: raw string
    Output: cleaned string
    """
    if text is None:
        return []
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
    text = [word for word in text.split() if word not in STOP_WORDS]

    return  text