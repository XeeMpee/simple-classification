import numpy as np
import pandas as pd


class ClassLabelsConverter:
    def __init__(self, df, class_tag):
        self.df = df
        labels = df[class_tag].values
        self.class_tag = class_tag
        self.labels = {}
        index = 0;
        for label in labels:
            if(label not in self.labels.keys()):
                self.labels[label] = index
                index += 1

    def label(self, indx: int) -> str:
        return self.labels.keys()[self.label.values().index(indx)]

    def identifier(self, label: str) -> int:
        return self.labels[label]

    def convertToIdentifiers(self, labels):
        identifiers = np.array([])
        for label in labels:
            identifiers = np.append(identifiers, self.identifier(label))
        return np.reshape(identifiers, (identifiers.size, 1)) 

    def convertToLabels(self, identifiers):
        labels = []
        for identifier in identifiers:
            labels.append(self.label(identifier))
        return labels

    def convert(self):
        y = self.df[self.class_tag].values
        self.df[self.class_tag] = self.convertToIdentifiers(y)
        return self.df

