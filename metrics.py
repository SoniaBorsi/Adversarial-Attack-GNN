import torch

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)
