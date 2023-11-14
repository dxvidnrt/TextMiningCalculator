import math
from typing import List, Dict


def dcg(relevances: List[int], k: int) -> float:
  """Computes DCG@k, given the corresponding relevance levels for a ranked list of documents.

  For example, given a ranking [2, 3, 1] where the relevance levels according to the ground
  truth are {1:3, 2:4, 3:1}, the input list will be [4, 1, 3].

  Args:
      relevances: List with the ground truth relevance levels corresponding to a ranked list of documents.
      k: Rank cut-off.

  Prints:
      DCG@k (float).
  """

  print(_dcg(relevances, k))

def _dcg(relevances: List[int], k: int) -> float:
  """Computes DCG@k, given the corresponding relevance levels for a ranked list of documents.

  For example, given a ranking [2, 3, 1] where the relevance levels according to the ground
  truth are {1:3, 2:4, 3:1}, the input list will be [4, 1, 3].

  Args:
      relevances: List with the ground truth relevance levels corresponding to a ranked list of documents.
      k: Rank cut-off.

  Returns:
      DCG@k (float).
  """
  # Note: Rank position is indexed from 1.
  return relevances[0] + sum(rel / math.log(i + 2, 2) for i, rel in enumerate(relevances[1:k]))

def ndcg(system_ranking: List[int], ground_truth: List[int], k: int = 10) -> float:
  """Computes NDCG@k for a given system ranking.

  Args:
      system_ranking: Ranked list of document IDs (from most to least relevant).
      ground_truth: Dict with document ID: relevance level pairs. Document not present here are to be taken with relevance = 0.
      k: Rank cut-off.

  Prints:
      NDCG@k (float).
  """
  # Relevance levels for the ranked docs.
  relevances = [ground_truth.get(rank, 0) for rank in system_ranking]

  # Relevance levels of the idealized ranking.
  relevances_ideal = sorted(ground_truth.values(), reverse=True)

  print(_dcg(relevances, k) / _dcg(relevances_ideal, k))