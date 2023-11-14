from typing import List
import math


def jaccard(x: List[int], y: List[int]) -> str:
    """Computes the Jaccard similarity between two term vectors."""
    intersection = sum([x_i > 0 and y_i > 0 for x_i, y_i in zip(x, y)])
    union = sum([x_i > 0 or y_i > 0 for x_i, y_i in zip(x, y)])
    print(f"The Jaccard Similarity is: {float(intersection) / union}")

def cosine(x: List[float], y: List[float]) -> float:
    """Computes the Cosine similarity between two term vectors."""
    dot_product = sum([x_i * y_i for x_i, y_i in zip(x, y)])
    norm_x = math.sqrt(sum(x_i**2 for x_i in x))
    norm_y = math.sqrt(sum(y_i**2 for y_i in y))
    print(f"The cosine similarity is: {float (dot_product / (norm_x * norm_y))}")