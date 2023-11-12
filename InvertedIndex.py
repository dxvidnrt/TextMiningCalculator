# This class creates an inverted index
from typing import List, Dict
from collections import defaultdict


def get_word_frequencies(doc: str) -> Dict[str, int]:
    """Extracts word frequencies from a document.

    Args:
        doc: Document content given as a string.

    Returns:
        Dictionary with words as keys and their frequencies as values.
    """
    freq_dict = {}
    words = doc.replace("\n", " ").replace(".", " ").replace(",", " ").replace("?", " ").replace("\t", " "). \
        replace(":", " ").replace(";", " ").replace("!", " ").lower().split(" ")
    for word in words:
        if word == "":
            pass
        elif word in freq_dict:
            freq_dict[word] = 1 + freq_dict[word]
        else:
            freq_dict[word] = 1
    return freq_dict


def inverted_index_representation(docs: List[str]):
    print("STOPWORD REMOVAL AND STEMMING MISSING!")
    dict_index = defaultdict(lambda: defaultdict(int))
    doc_id = 0
    for doc in docs:
        doc_id += 1
        doc_freq = get_word_frequencies(doc)
        for key, value in doc_freq.items():
            dict_index[key][doc_id] += value
    for word, payload in sorted(dict_index.items()):
        word_info = word+" -> "
        for doc_id, freq in payload.items():
            word_info += str(doc_id)+":"+str(freq)+", "
        word_info = word_info.rstrip(", ")
        print(word_info)



