from typing import List, Tuple, Dict

def prior_prob(document_term_matrix, y: str):
  """Computes the class prior probability, P(y).

    Args:
        y: class
        document_term_matrix: document term matrix

    Prints:
        The probability P(y).
    """

  print('prior prob: ', _prior_prob(document_term_matrix, y))

def evidence(document_term_matrix, terms: List[int]):
  """ Computes the evidence for a term (same for all classes)

  x documents out of y contain 't1, ..., tn' -> P('t1, ..., tn') = x/y
  Args:
    document_term_matrix: document term matrix
    terms: List of terms (in the list the term number is given (t1 -> 1)

  Prints:
    The evidence P('t1, ..., tn')
  """

  print('term evidence: ', _evidence(document_term_matrix, terms))

def class_conditional_probability(document_term_matrix, terms: List[int], y: str, smoothing_x: int, smoothing_y: int):
  """ Calculates the class conditional probability for the given list of terms and given class

  Args:
    document_term_matrix: document term matrix
    terms: List of terms to be considered
    y: class identifier
    smoothing_x: smoothing parameter for nominator
    smoothing_y: smoothing parameter for denominator

  Prints: P('t1, ...., tn' | y)

  """

  print('class conditional probability: ', _class_con_probability(document_term_matrix, terms, y, smoothing_x, smoothing_y))

def term_prob(document_term_matrix, term: int, y: str, smoothing_x: int, smoothing_y: int):
  """ Calculates the term probability for a given term and class
  Args:
    document_term_matrix: document term matrix
    term: term to calculate the probability for
    y: class name
    smoothing_x: smoothing parameter for the numerator
    smoothing_y: smoothing parameter for the denominator

  Returns: P(x_i | y)

  """
  all_same_length = len(set(map(len, document_term_matrix.values()))) == 1 if document_term_matrix else True
  print('' if all_same_length else "document term matrix is incorrect, not all term frequencies are defined")

  numerator = 0
  denominator = 0

  for element in document_term_matrix:

    if document_term_matrix[element][0] == y:
      # count number of times term appears in class y (do not consider term frequencies)
      if document_term_matrix[element][term] > 0:
        numerator += 1

      # count total number of terms in class y (do consider term frequencies)
      for i in range(1, len(document_term_matrix[element])):
        temp = document_term_matrix[element][i]
        denominator += document_term_matrix[element][i]

  # apply smoothing
  return ((numerator + smoothing_x)/(denominator + smoothing_y))

def prob_new_doc(document_term_matrix, terms: List[int], y: str, smoothing_x: int, smoothing_y: int):
  """ Calculation of P(y | 't1, ..., tn')
  Args:
    document_term_matrix: document term matrix
    terms: List of terms to be considered
    y: class name
    smoothing_x: smoothing parameter for the numerator
    smoothing_y: smoothing parameter for the denominator

  Prints: P(y | 't1, ..., tn')

  """
  all_same_length = len(set(map(len, document_term_matrix.values()))) == 1 if document_term_matrix else True
  print('' if all_same_length else "document term matrix is incorrect, not all term frequencies are defined")

  print('porbability of a new document: ', (_class_con_probability(document_term_matrix, terms, y, smoothing_x, smoothing_y) * _prior_prob(document_term_matrix, y))/_evidence(document_term_matrix, terms))

def _class_con_probability(document_term_matrix, terms: List[int], y: str, smoothing_x: int, smoothing_y: int):
  """ Calculates the class conditional probability for the given list of terms and given class

  Args:
    document_term_matrix: document term matrix
    terms: List of terms to be considered
    y: class identifier
    smoothing_x: smoothing parameter for nominator
    smoothing_y: smoothing parameter for denominator

  Returns: P('t1, ...., tn' | y)

  """
  result = None
  first = True

  for term in terms:
    if first:
      result = 1
      first = False

    result *= term_prob(document_term_matrix, term, y, smoothing_x, smoothing_y)

  return result

def _prior_prob(document_term_matrix, y: str):
  """ Private version: Computes the class prior probability, P(y).

    Args:
        y: class
        document_term_matrix: document term matrix

    Returns:
        The probability P(y).
    """
  temp_result = 0

  for element in document_term_matrix:
    if document_term_matrix[element][0] == y:
      temp_result += 1

  return temp_result / len(document_term_matrix)

def _evidence(document_term_matrix, terms: List[int]):
  """ Private Version: Computes the evidence for a term (same for all classes)

  x documents out of y contain 't1, ..., tn' -> P('t1, ..., tn') = x/y
  Args:
    document_term_matrix: document term matrix
    terms: List of terms (in the list the term number is given (t1 -> 1)

  Returns:
    The evidence P('t1, ..., tn')
  """

  temp_result = 0
  for element in document_term_matrix:
    if all(document_term_matrix[element][term] > 0 for term in terms):
      temp_result += 1

  return temp_result / len(document_term_matrix)
