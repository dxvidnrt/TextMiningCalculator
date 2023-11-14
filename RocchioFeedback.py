from typing import List, Tuple, Dict

def rocchio_feedback(
    vocab: List[str], dt_matrix,
    q: List[int], d_pos: List[int], d_neg: List[int],
    alpha: float, beta: float, gamma: float
) -> List[int]:
  """Computes an updated query model using Rocchio feedback.

  Args:
      vocab: Vocabulary
      dt_matrix: document term matrix
      q: Query vector.
      d_pos: List of positive feedback document IDs.
      d_neg: List of positive feedback document IDs.
      alpha: Feedback parameter alpha.
      beta: Feedback parameter beta.
      gamma: Feedback parameter gamma.

  Prints:
      Updated query vector.
  """
  q_m = [alpha * t for t in q]

  # Positive feedback docs
  for idx in d_pos:
    for t in range(len(vocab)):
      q_m[t] += beta / len(d_pos) * dt_matrix[idx][t]

  # Negative feedback docs
  for idx in d_neg:
    for t in range(len(vocab)):
      q_m[t] -= gamma / len(d_neg) * dt_matrix[idx][t]

  print(q_m)