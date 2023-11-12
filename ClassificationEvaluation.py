import math
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


def one_vs_rest_classification(instance_predictions: List[int]) -> str:
    """Returns votes for each class given test instance using one-vs-rest voting scheme.

    Args:
        instance_predictions: List of predictions for given test instance.
        Each element in the list corresponds to a prediction of a binary
        classification. 1 is positive prediction for the classifier at that index and 0 is negative prediction.

    Returns:
        A list with the number of votes each class received."""
    amount_classes = len(instance_predictions)
    output = [0] * amount_classes

    for i in range(amount_classes):
        if instance_predictions[i]:
            output[i] += 1
        else:
            output = [value + 1 if index != i else value for index, value in enumerate(output)]
    print("One vs. Rest Classification: "+str(output))


def one_vs_one_classification(instance_predictions: List[int]) -> str:
    """Returns votes for each class given test instance using one-vs-rest voting scheme.

        Args:
            instance_predictions: List of predictions for given test instance.
            Each element in the list corresponds to a prediction of a binary
            classification. 1 is positive prediction for the classifier at that index and 0 is negative prediction.
            Results are made for one pair (x_1, x_2), (x_1, x_3), ...


        Returns:
            A list with the number of votes each class received."""

    # Function to iterate and group the values
    n = len(instance_predictions)

    amount_classes = 1/2*(math.sqrt(8*n+1)-1)+1
    if not amount_classes.is_integer():
        print("Error in list length")
        return
    amount_classes = int(amount_classes)
    output = [0] * amount_classes
    index = 0
    for i in range(amount_classes):
        for j in range(i+1, amount_classes):
            if instance_predictions[index]:
                output[i] += 1
            else:
                output[j] += 1
            index += 1
    print("One vs. One Classification: " + str(output))


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
