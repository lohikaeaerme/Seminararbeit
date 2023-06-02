def k_fold(folds):
    """
    method for creating lists of indices for test und train sets. 
    It returns a list of tupel with (list of train indices, list of test indices
    and the number of the fold).
    """
    fold_ids = []
    for i in range(len(folds)):
        val = folds[i]
        test = folds[(i+1)%len(folds)]
        train = []
        for j in range(len(folds)):
            if i != j & (j != i+1 %len(folds)):
                train.extend(folds[j])
        fold_ids.append((val,test, train, (i + 1)))        

    return fold_ids