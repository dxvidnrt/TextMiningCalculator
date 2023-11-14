from typing import List


def jaccard(x: List[int], y: List[int]) -> str:
    """Computes the Jaccard similarity between two term vectors."""
    intersection = sum([x_i > 0 and y_i > 0 for x_i, y_i in zip(x, y)])
    union = sum([x_i > 0 or y_i > 0 for x_i, y_i in zip(x, y)])
    print(f"The Jaccard Similarity is: {float(intersection) / union}")
