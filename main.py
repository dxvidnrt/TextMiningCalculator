import ClassificationEvaluation
import InvertedIndex
import Retrieval


def main():
    # region Classification Evolution
    classification_evaluation_actual = [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    classification_evaluation_predicted = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
    # endregion
    ClassificationEvaluation.classification_evaluation(classification_evaluation_actual,
                                                       classification_evaluation_predicted)

    # region Creating an inverted index
    inverted_index_doc1 = "The old man and his two sons went fishing."
    inverted_index_doc2 = "Recreational fishing is an activity with important social implications."
    inverted_index_doc3 = "Introduction to social protection benefits for old age."
    inverted_index_doc4 = "Introduction to how lake trout fishing works."
    stopwords = "an, and, are, for, how, in, is, not, or, the, these, this, to, with"
    # endregion
    InvertedIndex.inverted_index_representation([inverted_index_doc1, inverted_index_doc2, inverted_index_doc3,
                                                inverted_index_doc4], stopwords=stopwords, suffix_s_removal=True)

    # region Classification Evaluation
    #endregion
    ClassificationEvaluation.one_vs_rest_classification([1, 1, 1, 1])
    ClassificationEvaluation.one_vs_one_classification([1, 1, 0, 1, 0, 1])

    # region BM25 Retrieval
    doc = {
        "T1": 3,
        "T2": 0,
        "T3": 2,
        "T4": 1,
        "T5": 10,
        "T6": 5
    }
    query = [("T2", 50), ("T2", 50), ("T5", 100)]
    # endregion
    Retrieval.bm_25(doc, query, 1000, 50, 1.25, 0.8)


if __name__ == "__main__":
    main()
