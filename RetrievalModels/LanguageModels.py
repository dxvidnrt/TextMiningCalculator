from typing import List, Dict
import Converter
#Smoothing


def empirical_language_model(query: str, docs: str, doc_term_matrix: List[List[int]], doc_names: List[str],
                             term_names: List[str]):
    docs_tokens = docs.split(" ")
    query_tokens = query.split(" ")
    res = 1
    for doc in docs_tokens:
        for term in query_tokens:
            temp = _P_t_d(term, doc, Converter.doc_term_matrix_2_doc_term_dict(doc_term_matrix, term_names, doc_names))
            res *= temp
            print(f"The probability of term {term} for {doc} in the empirical language model is: {temp}")
        print(f"The overall probability of query {query} for {doc} in the empirical language model is: {res}")
        res = 1
    return res


def background_language_model(query: str, doc_term_matrix: List[List[int]], doc_names: List[str],
                             term_names: List[str]):
    query_token = query.split(" ")
    res = 1
    for term in query_token:
        temp = _P_t_C(term, Converter.doc_term_matrix_2_doc_term_dict(doc_term_matrix, term_names, doc_names))
        res *= temp
        print(f"The probability of {term} in the background language model is: {temp}")
    print(f"The overall probability of query {query} in the background language model is: {res}")
    return res


def jelinek_mercer_smoothing(smoothing: float, query: str, docs: str,
                             doc_term_matrix: List[List[int]], doc_names: List[str], term_names: List[str]):
    """
    Jelinek Mercer Smoothing
    :param smoothing: smoothing parameter lambda
    :param term name of term
    :param doc name of document
    :param doc_term_matrix: List of documents of list of terms.
    :param doc_names names of documents
    :param term_names names of terms
    :return: the smoothed probability that a term is relevant given a document language model.
    """
    query_tokens = query.split(" ")
    doc_tokens = docs.split(" ")
    res = 1
    doc_term_dict = Converter.doc_term_matrix_2_doc_term_dict(doc_term_matrix, term_names, doc_names)
    for doc in doc_tokens:
        for term in query_tokens:
            temp = ((1-smoothing) * _P_t_d(term, doc, doc_term_dict)) + (smoothing * _P_t_C(term, doc, doc_term_dict))
            res *= temp
            print(f"Jelinek Mercer Smoothing with parameter {smoothing} for term {term}: {temp}")
        print(f"The overall probability of query {query} for doc {doc} in the Jelenik Mercer Smoothing model is: {res}")
        res = 1
    return res


def dirichlet_smoothing(smoothing: float, query: str, docs: str,
                        doc_term_matrix: List[List[int]], doc_names: List[str], term_names: List[str]):
    """
    Dirichlet Smoothing
    :param smoothing: smoothing parameter lambda
    :param term name of term
    :param doc name of document
    :param doc_term_matrix: List of documents of list of terms.
    :param doc_names names of documents
    :param term_names names of terms
    :return: the smoothed probability that a term is relevant given a document language model.
    """
    query_token = query.split(" ")
    docs_tokens = docs.split(" ")
    res = 1
    doc_term_dict = Converter.doc_term_matrix_2_doc_term_dict(doc_term_matrix, term_names, doc_names)
    for doc in docs_tokens:
        for term in query_token:
            term_freq_doc = doc_term_dict[doc][term]
            document_length = sum(doc_term_dict[doc].values())
            temp = (term_freq_doc + (smoothing * _P_t_C(term, doc_term_dict)))/(document_length + smoothing)
            res *= temp
            print(f"Dirichlet Smoothing with parameter {smoothing} for term {term}: {temp}")
        print(f"The overall probability of query {query} for doc {doc} in the Dirichlet Smoothing model is: {res}")
        res = 1
    return res


def _P_t_d(term: str, document: str, doc_term_dict: Dict[str, Dict[str, int]]):
    """
    Calculates the probability of a term t belonging to a document d.
    :param term: name of term
    :param document: name of document
    :param doc_term_dict: Dictionary of documents. Each containing the term frequency for each term.
    :return: P(t|d)
    """
    doc_len = sum(doc_term_dict[document].values())
    term_freq_doc = doc_term_dict[document][term]
    return term_freq_doc / doc_len


def _P_t_C(term: str, doc_term_dict: Dict[str, Dict[str, int]]):
    """
    Calculates the probability of a term t belonging to a document d.
    :param term: name of term
    :param document: name of document
    :param doc_term_dict: Dictionary of documents. Each containing the term frequency for each term.
    :return: P(t|C)
    """
    term_occurences_collection = 0
    collection_length = 0
    for document, term_freq in doc_term_dict.items():
        term_occurences_collection += term_freq[term]
        collection_length += sum(term_freq.values())
    return term_occurences_collection / collection_length
