import math
from typing import List, Tuple, Dict


def _get_tf_vector(doc_term_vector: List[int]) -> List[float]:
    """Computes the normalized term frequency vector from a raw term-frequency vector."""
    sum_freq = sum(doc_term_vector)
    if sum_freq == 0:  # This would mean that the document has no content.
        return None
    tf_vector = [freq / sum_freq for freq in doc_term_vector]
    return tf_vector


def get_tf_vector(doc_term_vector: List[int]) -> List[float]:
    print(get_tf_vector(doc_term_vector))


def _get_term_idf(doc_term_matrix: List[List[int]], term_index: int) -> float:
    """Computes the IDF value of a term, given by its index, based on a document-term matrix."""
    N = len(doc_term_matrix)
    n_t = sum([1 if doc_freqs[term_index] > 0 else 0 for doc_freqs in doc_term_matrix])
    return math.log10(N / n_t)


def _get_term_idf(doc_term_matrix: List[List[int]], term_index: int) -> float:
    print(_get_term_idf(doc_term_matrix))


def _get_tfidf_vector(doc_term_matrix: List[List[int]], doc_index: int) -> List[float]:
    """Computes the TFIDF vector from a raw term-frequency vector."""
    tf_vector = get_tf_vector(doc_term_matrix[doc_index])
    tfidf_vector = []
    for term_index, tf in enumerate(tf_vector):
        idf = _get_term_idf(doc_term_matrix, term_index)
        tfidf_vector.append(tf * idf)
    return tfidf_vector


def _get_tfidf_vector(doc_term_matrix: List[List[int]], doc_index: int) -> List[float]:
    print(_get_tfidf_vector(doc_term_matrix))


def bm_25(doc_term_freq: Dict[str, int], query: List[Tuple[str, int]], docs_total: int, doc_len_avg: int, k_1: float, b: float):
    """
    Calculating bm_25 with formula:
    $\text{BM25} = \sum_{t \in q} \frac{f_{t,d} \cdot (1 + k_1)}{(f_{t,d} + k_1 \cdot (1 - b + b \cdot
    \frac{|d|}{\text{avgdl}}))} \cdot \log\left(\frac{{N}}{{n_t}}\right)$
    !Using base 10 logarithm
    :param doc_term_freq: Each key is a term, each value is the occurence of the term in the document.
    :param query List of tokenized query. Each token is a tuple of (term, n_t).
    n_t amount of docs that contain the term t
    :return:
    """
    res = 0
    for token in query:
        term = token[0]
        n_t = token[1]
        if n_t:
            f_t_d = doc_term_freq[term]
            idf_t = math.log10(docs_total/n_t)
            doc_len = sum(doc_term_freq.values())
            res += ((f_t_d * (1 + k_1))/(f_t_d + k_1 * (1 - b + b * (doc_len/doc_len_avg))))*idf_t
    print("The retrieval score of BM25 is: "+str(res))


def count_ordered_bigrams(term_1: str, term_2: str, document: List[str]) -> str:
    """Counts ordered bigrams for the pair (term_1, term_2) in document.

    :returns
        Bigrams count"""

    count = 0
    for j, d_term in enumerate(document[:-1]):
        if term_1 == d_term and term_2 == document[j+1]:
            count += 1
    print(f"There are {count} ordered bigrams.")


def count_unordered_bigrams(term_1: str, term_2:str, document: List[str]):
    count = 0
    for i, d_term in enumerate(document[:-1]):
        for j, d_term2 in enumerate(document[i+1:]):
            if term_1 == d_term and term_2 == d_term2:
                count += 1
    print(f"There are {count} unordered bigrams.")


def precision_at_k(docs: List[int], total_relevant: int, k: int):
    """
    Calculate Precision@k
    :param docs: List of documents from highest rank to lowest. 1 means relevant, 0 non-relevant.
    :param total_relevant: number of total existing relevant documents
    :param k: first k documents that are looked at.
    """
    if k is None:
        k = len(docs)
    relevant = 0
    for i, value in enumerate(docs[:k]):
        relevant += value
    print(f"The precision@{k} score is: {float(relevant)/k}")


def _precision_at_k(docs: List[int], total_relevant: int, k: int):
    """
    Calculate Precision@k
    :param docs: List of documents from highest rank to lowest. 1 means relevant, 0 non-relevant.
    :param total_relevant: number of total existing relevant documents
    :param k: first k documents that are looked at.
    """
    if k is None:
        k = len(docs)
    relevant = 0
    for i, value in enumerate(docs[:k]):
        relevant += value
    return float(relevant)/k


def recall_at_k(docs: List[int], total_relevant: int, k: int):
    """
    Calculate Recall@k.
    If only Recall is asked, set k to length of document or set to None.
    :param docs: List of documents from highest rank to lowest. 1 means relevant, 0 non-relevant.
    :param total_relevant: number of total existing relevant documents
    :param k: first k documents that are looked at.
    """
    if k is None:
        k = len(docs)
    relevant = 0
    for i, value in enumerate(docs[:k]):
        relevant += value
    print(f"The precision@{k} score is: {float(relevant)/total_relevant}")


def average_precision(docs: List[int], total_relevant: int):
    precision = 0
    for i, value in enumerate(docs):
        if value:
            precision += _precision_at_k(docs, total_relevant, i+1)
    print(f"The Average Precision is {float(precision)/total_relevant}")
