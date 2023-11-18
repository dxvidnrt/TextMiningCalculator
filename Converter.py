# This file converts data from one layout to another
from collections import defaultdict
from typing import List, Dict


def doc_term_matrix_2_doc_term_dict(doc_term_matrix: List[List[int]], term_names: List[str],
                                    document_names: List[str]) -> Dict[str, Dict[str, int]]:

    """
    Takes a doc term matrix and turns it into a doc term dictionary
    :param doc_term_matrix:
    :param term_names:
    :param document_names:
    :return:
    """
    doc_term_dict = defaultdict(lambda: defaultdict(str))
    for i in range(len(doc_term_matrix)):
        for j in range(len(doc_term_matrix[i])):
            doc_term_dict[document_names[i]][term_names[j]] = doc_term_matrix[i][j]
    return doc_term_dict
