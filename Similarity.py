from typing import List
import math


def jaccard(x: str, y: str) -> str:
    """Computes the Jaccard similarity between two terms.
    Testes on old exam."""
    set1, set2 = set(x.split()), set(y.split())
    res = len(set1.intersection(set2))/len(set1.union(set2))
    print(f"Jaccard Similarity of {x} and {y} is {res}")
    return res



def cosine(x: List[float], y: List[float]) -> float:
    """Computes the Cosine similarity between two term vectors."""
    dot_product = sum([x_i * y_i for x_i, y_i in zip(x, y)])
    norm_x = math.sqrt(sum(x_i**2 for x_i in x))
    norm_y = math.sqrt(sum(y_i**2 for y_i in y))
    print(f"The cosine similarity is: {float (dot_product / (norm_x * norm_y))}")