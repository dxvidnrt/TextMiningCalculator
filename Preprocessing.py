from typing import List, Set
import re
import string


def tokenize(text: str) -> str:
    """Returns a sequence of terms given an input text."""
    # Remove HTML markup using a regular expression.
    re_html = re.compile("<[^>]+>")
    text = re_html.sub(" ", text)
    # Replace punctuation marks (including hyphens) with spaces.
    for c in string.punctuation:
        text = text.replace(c, " ")
    # Lowercase and split on whitespaces.
    print(f"Tokenized Text: {text.lower().split()}")


def remove_stopwords(tokens: List[str], stopwords: Set[str]) -> str:
    """Removes stopwords from a sequence of tokens."""
    print(f"Text without stopwords: {[token for token in tokens if token not in stopwords]}")


def suffix_s_stemmer(terms: List[str]) -> List[str]:
    """Removes the s-suffix from all terms in a sequence."""
    stemmed_terms = []
    for term in terms:
        stemmed_term = term[:-1] if term[-1] == "s" else term
        stemmed_terms.append(stemmed_term)
    print(f"Text with suffix-s removal: {stemmed_terms}")