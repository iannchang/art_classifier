import math
import numpy as np

class NaiveBayes:

    def __init__(self, use_mle):
        self.label_counts = {}
        self.feature_counts = {}
        self.use_mle = use_mle # True for MLE, False for MAP with Laplace add-one smoothing

    def fit(self, train_features, train_labels):
        self.label_counts[0] = 0
        self.label_counts[1] = 0
        self.label_counts[2] = 0
        self.label_counts[3] = 0
        for label in train_labels:
            self.label_counts[label]+=1
        row_counter = 0
        for row in train_features:
            column_counter = 0
            for column in row:
                tuple = (column_counter, train_features[row_counter][column_counter], train_labels[row_counter])
                frequency = self.feature_counts.get(tuple, 0)
                frequency+=1
                self.feature_counts[tuple] = frequency
                column_counter+=1
            row_counter+=1


    def predict(self, test_features):
        preds = np.zeros(test_features.shape[0], dtype=np.uint8)
        laplace_num = 1
        laplace_den = 2
        if self.use_mle:
            laplace_num = 0
            laplace_den = 0
        num_row = 0
        total_label_count = 0
        for count in self.label_counts:
            total_label_count+=self.label_counts[count]
        p = []
        for count in self.label_counts:
            p.append(self.label_counts[count]/total_label_count)
        for row in test_features:
            num_feature = 0
            feature_prob = [[]]
            for feature in row:
                feature_prob.append([])
                for i in range(len(self.label_counts)):
                    num = self.feature_counts.get((num_feature, test_features[num_row][num_feature], i),0) + laplace_num
                    den = self.label_counts[i] + laplace_den
                    feature_prob[num_feature].append(num/den)
                num_feature+=1

            row_prob = [1]*len(self.label_counts)
            for i in range(len(feature_prob)-1):
                for j in range(len(self.label_counts)):
                    row_prob[j] = row_prob[j] * feature_prob[i][j]

            for i in range(len(self.label_counts)):
                row_prob[i] = row_prob[i] * p[i]
            row_prob = np.asarray(row_prob)
            preds[num_row] = np.argmax(row_prob)
            num_row+=1
        return preds
