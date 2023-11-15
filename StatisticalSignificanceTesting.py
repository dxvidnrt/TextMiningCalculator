import numpy as np
from typing import List
from scipy import stats

def t_stat(a: List[float], b: List[float], n: int) -> float:
  """Computes the t statistic between two systems.

    Args:
      a: System A recorded metric for each topic.
      b: System B recorded metric for each topic.
      n: Size of the sample.

    Prints:
      t statistic for t-test between two systems.
    """
  print('The t statistic is: ', _t_stat(a, b, n))

def _t_stat(a: List[float], b: List[float], n: int) -> float:
  """Computes the t statistic between two systems.

  Args:
    a: System A recorded metric for each topic.
    b: System B recorded metric for each topic.
    n: Size of the sample.

  Retuns:
    t statistic for t-test between two systems.
  """
  n = min(len(a), n)
  x = np.array(a[:n]) - np.array(b[:n])

  x_D = np.mean(x)
  s_D = np.sqrt(sum((x - x_D) ** 2) / (n - 1))

  return x_D / (s_D / np.sqrt(n))

def p_value(n: int, t_stat_variable: float) -> float:
  """Computes the p-value.

  Args:
    n: Size of the sample.
    t_stat: t statisitic.

  Prints:
    p-value for t statistic.
  """
  df = n - 1
  p = (1.0 - stats.t.cdf(abs(t_stat_variable), df)) * 2.0
  print('The p-value is: ', p)