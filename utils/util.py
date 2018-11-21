import torch
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def round_choice(x):
    mid = 0.35 #0.15
    x[x > mid] = 1
    x[x == mid] = 0
    x[x < mid] = 0
    return x

def accuracy(y_true, y_pred):
    # Calculates the precision
    y_true = round_choice(np.clip(y_true, 0, 1))
    y_pred = round_choice(np.clip(y_pred, 0, 1))
    count = np.sum(y_true==y_pred)
    accuracy = count / (y_true.shape[0]*y_true.shape[1] + 1e-6)
    return accuracy

def attr_accuracy(y_true, y_pred, index):
    # Calculates the precision
    y_true = round_choice(np.clip(y_true, 0, 1))
    y_pred = round_choice(np.clip(y_pred, 0, 1))
    attr_y_true = y_true[:, index]
    attr_y_pred = y_pred[:, index]

    count = np.sum(attr_y_true == attr_y_pred)
    #accuracy = count / (y_true.shape[0] + 1e-6)
    return count

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = np.sum(round_choice(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(round_choice(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-6)
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = np.sum(round_choice(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(round_choice(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-6)
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if np.sum(np.round(np.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + 1e-6)
    return fbeta_score


def fmeasure(p, r):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return 2 * (p * r) / (p + r + 1e-6)