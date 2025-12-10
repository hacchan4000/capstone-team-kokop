import re
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

# ===========================================================
# BASIC TEXT CLEANING
# ===========================================================
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)

    # Lemmatization
    text = " ".join(lemm.lemmatize(w) for w in text.split())

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===========================================================
# REMOVE SPOKEN FILLERS (untuk analisis kualitas, bukan WER)
# ===========================================================
def normalize_spoken_form(text):
    """
    Menormalkan bentuk lisan yang umum di speech.
    Ini SAH secara akademik.
    """
    mapping = {
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "kinda": "kind of",
        "sorta": "sort of"
    }

    words = text.split()
    norm = [mapping.get(w, w) for w in words]
    return " ".join(norm)

# ===========================================================
# REMOVE DUPLICATED WORDS
# ===========================================================
def remove_repeated_words(text):
    words = text.split()
    cleaned = []
    prev = ""

    for w in words:
        if w != prev:
            cleaned.append(w)
        prev = w

    return " ".join(cleaned)

# ===========================================================
# SOFT SIMILARITY
# ===========================================================
def soft_similarity(a, b):
    a = clean_text(a).split()
    b = clean_text(b).split()

    if len(a) == 0 or len(b) == 0:
        return 0.0

    match = SequenceMatcher(None, a, b)
    return match.ratio() * 100