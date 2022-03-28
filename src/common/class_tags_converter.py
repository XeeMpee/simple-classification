import numpy as np
import pandas as pd


class ClassLabelsConverter:
    def __init__(self, labels):
        self.labels = {}
        index = 0;
        for label in labels[:,0]:
            if(label not in self.labels.keys()):
                self.labels[label] = index
                index += 1

    def label(self, indx: int) -> str:
        return self.labels.keys()[self.label.values().index(indx)]

    def identifier(self, label: str) -> int:
        return self.labels[label]

    def convertToIdentifiers(self, labels):
        identifiers = np.array([])
        for label in labels[:,0]:
            identifiers = np.append(identifiers, self.identifier(label))
        return np.reshape(identifiers, (identifiers.size, 1)) 

    def convertToLabels(self, identifiers):
        labels = []
        for identifier in identifiers:
            labels.append(self.label(identifier))
        return labels
