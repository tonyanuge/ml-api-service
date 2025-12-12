import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    # If text is None or empty
    if not text or not isinstance(text, str):
        return ""

    # Basic cleanup
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Tokenize safely
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()  # fallback if nltk tokeniser fails

    # Remove stopwords + lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and token.strip() != ""
    ]

    return " ".join(cleaned_tokens)
