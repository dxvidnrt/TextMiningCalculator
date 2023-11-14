from typing import List


def create_folds(instances: List[int], k: int = 5) -> str:
    """Given a set of instances, it returns k splits of train and test."""

    folds = []
    for fold_id in range(k):
        train, test = [], []
        for i in range(len(instances)):
            if i % k == fold_id:
                test.append(instances[i])
            else:
                train.append(instances[i])

        folds.append({
            'train': train,
            'test': test
        })
    print(f"For instances {instances} and k = {k}, the data is split into: \n train: {train} \n test: {test}")
