from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import f1_score as sk_f1_score

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)

def compute_f1_score(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    return sk_f1_score(labels, preds, average='macro')

def compute_precision(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    return precision_score(labels, preds, average='macro', zero_division=0)

def compute_recall(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    return recall_score(labels, preds, average='macro')

