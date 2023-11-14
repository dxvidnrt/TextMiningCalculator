from typing import List

def reciprocal_rank(ranking: List[int], ground_truth: List[int]):
  """
  Reciprocal of the rank at which the first relevant document is retrieved
  Args:
    ranking:
    ground_truth:

  Returns: RR

  """

  for i in range(len(ranking)):
    if ranking[i] in ground_truth:
      print('The reciprocal rank is: ', 1 / (i+1))
      return

  print('The reciprocal rank is: zero')
  print('The reciprocal rank is: ', _reciprocal_rank())

def _reciprocal_rank(ranking: List[int], ground_truth: List[int]):
  """
  Reciprocal of the rank at which the first relevant document is retrieved
  Args:
    ranking:
    ground_truth:

  Returns: RR

  """

  for i in range(len(ranking)):
    if ranking[i] in ground_truth:
      return  1 / (i+1)
      return

  return 0

def precision(ranking: List[int], ground_truth: List[int], k:int):
  print("The precision at ", k, " is: ", _precision(ranking, ground_truth, k))

def _precision(ranking: List[int], ground_truth: List[int], k:int):

  temp_result = 0

  for i in range(k):
    if ranking[i] in ground_truth:
      temp_result += 1

  return temp_result/k

def average_precision(ranking: List[int], ground_truth: List[int]):
  print('The average precision is: ', _average_precision(ranking, ground_truth))

def _average_precision(ranking: List[int], ground_truth: List[int]):
  temp_result = 0

  for i in range(len(ranking)):
    if ranking[i] in ground_truth:
      temp_result += _precision(ranking, ground_truth, i+1)

  return temp_result / len(ground_truth)

def mean_calculation(*precisions):
  temp_result = 0
  for precision in precisions:
    for element in precision:
      temp_result += element
  print('The mean average is: ', temp_result / len(precision))