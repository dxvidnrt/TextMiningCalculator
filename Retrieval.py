import math
from typing import List, Tuple


def bm_25(doc_term_freq: dict[str, int], query: List[Tuple[str, int]], docs_total: int, doc_len_avg: int, k_1: float, b: float):
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




