import pickle
import numpy as np


def loadData(file_path):
    label, Negative, Positive = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [_.strip() for _ in f.readlines()]
    for i, line in enumerate(lines):
        if i:
            items = line.split(',')
            label_item =int(items[1])
            if label_item == 0:
                Negative.append([items[0],label_item])
            else:
                Positive.append([items[0],label_item])
            label.append(label_item)
    return Negative, Positive, label

def loadData_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    Negative = []
    Positive = []
    label = []
    data = np.array(data)

    for item in data:
        label_item = int(item[1])
        if label_item == 0:
            Negative.append([item[0], label_item])
        else:
            Positive.append([item[0], label_item])
        label.append(label_item)

    return Negative, Positive, label
