from typing import List

def get_confusion_matrix(
        actual: List[int], predicted: List[int]
) -> List[List[int]]:
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        List of two lists of length 2 each, representing the confusion matrix.
    """
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(actual)):
        if actual[i] is predicted[i]:
            if actual[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predicted[i] == 1:
                fp += 1
            else:
                fn += 1
    return [[tn, fp], [fn, tp]]


def accuracy(actual: List[int], predicted: List[int]) -> float:
    """Computes the accuracy from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Accuracy as a float.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    return ((conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] +
                                                       conf_matrix[1][1]))


def precision(actual: List[int], predicted: List[int]) -> float:
    """Computes the precision from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Precision as a float.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    return conf_matrix[1][1] / (conf_matrix[0][1] + conf_matrix[1][1])


def recall(actual: List[int], predicted: List[int]) -> float:
    """Computes the recall from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Recall as a float.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    return conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1])


def f1(actual: List[int], predicted: List[int]) -> float:
    """Computes the F1-score from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of harmonic mean of precision and recall.
    """
    return (2 * precision(actual, predicted) * recall(actual, predicted)) / (
                precision(actual, predicted) + recall(actual, predicted))


def false_positive_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false positive rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as positive divided
            by number of actually negative instances.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    return conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[0][0])


def false_negative_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false negative rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as negative divided
            by number of actually positive instances.
    """
    conf_matrix = get_confusion_matrix(actual, predicted)
    return conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[1][1])


def classification_evaluation(actual: List[int], predicted: List[int]):
    print("Confusion Matrix: ")
    confusion_matrix = get_confusion_matrix(actual, predicted)
    print("True positive: " + str(confusion_matrix[1][1]))
    print("False positive: " + str(confusion_matrix[0][1]))
    print("True negative: " + str(confusion_matrix[0][0]))
    print("False negative: " + str(confusion_matrix[1][0]))
    print("----------------")
    print("Precision: " + str(precision(actual, predicted)))
    print("Recall: " + str(recall(actual, predicted)))
    print("F1-measure: " + str(f1(actual, predicted)))