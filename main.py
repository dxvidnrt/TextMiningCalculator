import ClassificationEvaluation
import CrossValidator
import InvertedIndex
import Retrieval
import PageRanker
import Similarity
import NaiveBayes
import RocchioFeedback
import DiscountedCumulativeGain
import EvaluationMeasures
import StatisticalSignificanceTesting
import RetrievalModels.LanguageModels
import KmeansClustering
import numpy as np

def main():
    # region Classification Evolution
    classification_evaluation_actual = [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    classification_evaluation_predicted = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
    # endregion
    # ClassificationEvaluation.classification_evaluation(classification_evaluation_actual,
    # classification_evaluation_predicted)

    # region Creating an inverted index
    inverted_index_doc1 = "The old man and his two sons went fishing."
    inverted_index_doc2 = "Recreational fishing is an activity with important social implications."
    inverted_index_doc3 = "Introduction to social protection benefits for old age."
    inverted_index_doc4 = "Introduction to how lake trout fishing works."
    stopwords = "an, and, are, for, how, in, is, not, or, the, these, this, to, with"
    # endregion
    # InvertedIndex.inverted_index_representation([inverted_index_doc1, inverted_index_doc2, inverted_index_doc3,
    # inverted_index_doc4], stopwords=stopwords, suffix_s_removal=True)

    # region Classification Evaluation
    # endregion
    # ClassificationEvaluation.one_vs_rest_classification([1, 1, 1, 1])
    # ClassificationEvaluation.one_vs_one_classification([1, 1, 0, 1, 0, 1])

    # region Ranking Evaluation
    docs = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    # endregion
    # Retrieval.precision_at_k(docs, 6, 10)
    # Retrieval.recall_at_k(docs, 6, None)
    # Retrieval.average_precision(docs, 6)

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
    # Retrieval.bm_25(doc, query, 1000, 50, 1.25, 0.8)

    # region Bigrams
    term_1 = "A"
    term_2 = "B"
    doc = ["A", "B", "C", "A", "B"]
    # endregion
    # Retrieval.count_ordered_bigrams(term_1, term_2, doc)
    # Retrieval.count_unordered_bigrams(term_1, term_2, doc)

    # region Similarity
    term_vector_1 = [1, 1, 0, 2]
    term_vector_2 = [1, 1, 2, 0]
    # endregion
    # Similarity.jaccard(term_vector_1, term_vector_2)
    # Similarity.cosine(term_vector_1, term_vector_2)

    # region PageRanker
    graph_edges = [("A", "B"), ("B", "C"), ("C", "E"), ("E", "D"), ("D", "B"), ("E", "F")]
    # endregion
    # PageRanker.pagerank(graph_edges, q=0.4, iterations=3)

    # region CrossValidation k-fold
    instances = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    k = 3
    # endregion
    # CrossValidator.create_folds(instances, k)

    # region Naive Byes
    document_term_matrix = {}
    # document_term_matrix[' document name '] = [' class name as str ' , frequency of term 1, ..., frequency of term n]
    document_term_matrix['doc1'] = ['c1', 2, 0, 1, 2, 0, 2, 4]
    document_term_matrix['doc2'] = ['c3', 0, 0, 0, 0, 3, 2, 2]
    document_term_matrix['doc3'] = ['c2', 3, 4, 0, 2, 0, 0, 2]
    document_term_matrix['doc4'] = ['c3', 4, 0, 3, 1, 1, 1, 0]
    document_term_matrix['doc5'] = ['c2', 1, 0, 0, 3, 1, 2, 0]
    document_term_matrix['doc6'] = ['c1', 0, 1, 1, 0, 3, 4, 1]
    # endregion
    # P(c3) (document_term_matrix, 'class name')
    #NaiveBayes.prior_prob(document_term_matrix, 'c3')
    # P('t1, ..., tn') (document_term_matrix, [term1, term4, term5]
    #NaiveBayes.evidence(document_term_matrix, [1,4,5])
    # P('t1, ..., tn' | c3) (document_term_matrix, [term1, term4, term5], 'class name', smoothing parameter x, smoothing parameter y)
    #NaiveBayes.class_conditional_probability(document_term_matrix, [1,4,5], 'c3', 1, 3)
    # P(c3 | 't1, ..., tn') (document_term_matrix, [term1, term4, term5], 'class name', smoothing parameter x, smoothing parameter y)
    #NaiveBayes.prob_new_doc(document_term_matrix, [1, 4, 5], 'c3', 1, 3)

    # region Rocchio Feedback
    # Vocab
    VOCAB = ['news', 'about', 'presidental', 'campaign', 'food', 'text']

    # Query vector
    Q = [1, 1, 1, 1, 0, 0]

    # docuement term matrix (each line corresponds to one document)
    DT_MATRIX = [
        [1.5, 0.1, 0, 0, 0, 0],
        [1.5, 0.1, 0, 2, 2, 0],
        [1.5, 0, 3, 2, 0, 0],
        [1.5, 0, 4, 2, 0, 0],
        [1.5, 0, 0, 6, 2, 0]
    ]

    # set of relevant documents
    D_POS = [2,3]

    # set of non-relevant documents
    D_NEG = [0,1,4]

    # parameter alpha
    alpha = 0.6

    # parameter beta
    beta = 0.2

    # parameter gamma
    gamma = 0.2
    # endregion
    # RocchioFeedback.rocchio_feedback(VOCAB, DT_MATRIX, Q, D_POS, D_NEG, alpha, beta, gamma)

    # region Discounted Cumulative Gain
    # Relevances: List with the ground truth relevance levels corresponding to a ranked list of documents (rel_i)
    relevances = [0, 3, 0, 0, 2, 3, 1, 0, 0, 0]

    # Rank cut-off
    k = 5

    # System Ranking: key is the query id, value is the list of document ids (ranked from most to least relevant)
    system_rankings = {
        "q1": [2, 1, 3, 4, 5, 6, 10, 7, 9, 8],
        "q2": [1, 2, 9, 4, 5, 6, 7, 8, 3, 10],
        "q3": [1, 7, 4, 5, 3, 6, 9, 8, 10, 2]
    }

    # Ground Truth: key is the query id, values is a dictionary (doc id, relevance)
    # Relevance is measured on a 3-point scale: (0: non-relevant, 1: poor, 2: good, 3: excellent)
    # Documents not listed here are non-relevant
    ground_truth = {
        "q1": {4: 3, 1: 2, 2: 1},
        "q2": {3: 3, 4: 3, 1: 2, 2: 1, 8: 1},
        "q3": {1: 3, 4: 3, 7: 2, 5: 2, 6: 1, 8: 1}
    }

    # endregion
    # DiscountedCumulativeGain.dcg(relevances, 5)
    # DiscountedCumulativeGain.ndcg(system_rankings['q3'], ground_truth['q3'], k)

    # region Evaluation Measure
    ranking_A_q1 = [1,2,4,5,3,6,9,8,10,7]
    ranking_A_q2 = [1,2,4,5,3,9,8,6,10,7]
    ranking_A_q3 = [1,7,4,5,3,6,9,8,10,2]

    ground_truth_1 = [1,3]
    ground_truth_2 = [2,4,5,6]
    ground_truth_3 = [7]

    # endregion

    #EvaluationMeasures.precision(ranking_A_q3, ground_truth_3, 5)
    #EvaluationMeasures.precision(ranking_A_q3, ground_truth_3, 10)

    #EvaluationMeasures.average_precision(ranking_A_q3, ground_truth_3)

    #EvaluationMeasures.reciprocal_rank(ranking_A_q3, ground_truth_3)

    #EvaluationMeasures.mean_calculation([EvaluationMeasures._precision(ranking_A_q1, ground_truth_1, 5),
     #                                  EvaluationMeasures._precision(ranking_A_q2, ground_truth_2, 5),
     #                                 EvaluationMeasures._precision(ranking_A_q3, ground_truth_3, 5)])

    #EvaluationMeasures.mean_calculation([EvaluationMeasures._average_precision(ranking_A_q1, ground_truth_1),
     #                                    EvaluationMeasures._average_precision(ranking_A_q2, ground_truth_2),
     #                                    EvaluationMeasures._average_precision(ranking_A_q3, ground_truth_3)])

    #EvaluationMeasures.mean_calculation([EvaluationMeasures._reciprocal_rank(ranking_A_q1, ground_truth_1),
     #                                    EvaluationMeasures._reciprocal_rank(ranking_A_q2, ground_truth_2),
     #                                    EvaluationMeasures._reciprocal_rank(ranking_A_q3, ground_truth_3)])

    #region Statistical Significance Testing
    system_A = [0.2215, 0.3924, 0.654, 0.5611, 0.9186, 0.1104, 0.6086, 0.5062, 0.9688, 0.995]
    system_B = [0.0765, 0.0426, 0.5738, 0.1571, 0.9881, 0.7164, 0.7507, 0.435, 0.3959, 0.8709]
    n = len(system_A)
    # endregion
    # StatisticalSignificanceTesting.t_stat(system_A, system_B, n)
    # StatisticalSignificanceTesting.p_value(n, StatisticalSignificanceTesting._t_stat(system_A, system_B, n))

    # region Language Models
    document_term_matrix = [
        [1, 0, 2, 4, 1],
        [1, 2, 0, 0, 2],
        [2, 0, 1, 1, 1],
        [1, 1, 0, 2, 0]
    ]
    term_names = ["term1", "term2", "term3", "term4", "term5"]
    document_names = ["doc1", "doc2", "doc3", "doc4"]
    smoothing = 6
    # endregion
    #RetrievalModels.LanguageModels.dirichlet_smoothing(smoothing, "term1 term3 term5", "doc1 doc2 doc3 doc4", document_term_matrix, document_names, term_names)
    #RetrievalModels.LanguageModels.empirical_language_model("term1 term3 term5", "doc4", document_term_matrix, document_names, term_names)
    #RetrievalModels.LanguageModels.background_language_model("term4 term2", document_term_matrix, document_names, term_names)

    #region K-means Clustering

    # points
    p1 = np.array([3,1,0,4])
    p3 = np.array([1,3,4.5,5])
    p5 = np.array([6,3,2,0])

    # centroids
    c1 = np.array([3,5,4,4.5])
    c2 = np.array([1.5,2.5,1,2])
    c3 = np.array([2,3.5,1,0])

    # ACHTUNG, wenn minimale Distanz über 100
    KmeansClustering.k_means_clustering(3, p1,p3,p5, c1,c2,c3)
    #endregion
if __name__ == "__main__":
    main()