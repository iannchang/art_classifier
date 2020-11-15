import math
import numpy as np
#Note: I did reference this Github repo since Softmax was not taught in the class. Mainly for understanding one hot encoding, but it also helped my adapt my logistic regression code to softmax regression
#https://github.com/rickwierenga/MLFundamentals

def softmax(vec):
    vec -= np.max(vec)
    return np.exp(vec) / np.sum(np.exp(vec))

def one_hot_encode(train_labels, num_labels):
  one_hot = np.zeros((len(train_labels), num_labels))
  one_hot[np.arange(len(train_labels)), train_labels] = 1
  return one_hot



class SoftmaxRegression:

    def __init__(self, learning_rate, max_steps):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weights = None

    def fit(self, train_features, train_labels, num_labels):
        train_features = np.insert(train_features, 0, 1, axis=1)
        train_features[:, 1:] = (train_features[:, 1:] - np.mean(train_features[:, 1:], axis=0)) / np.std(train_features[:, 1:], axis=0)
        train_features.setflags(write=False)
        m = train_features.shape[1]
        np.random.seed(0)
        theta = np.random.random((train_features.shape[1],num_labels))
        for i in range(self.max_steps):
            thetatx = np.matmul(train_features, theta)
            gradient = 1/m * np.matmul(np.transpose(train_features), (softmax(thetatx) - one_hot_encode(train_labels, num_labels)))
            theta = theta - self.learning_rate*gradient
        self.weights = theta

    def predict(self, test_features):
        test_features = np.insert(test_features, 0, 1, axis=1)
        preds = np.zeros(test_features.shape[0])
        probabilities = softmax(np.matmul(test_features, self.weights))
        counter = 0
        for row in probabilities:
            preds[counter] = np.argmax(row)
            counter+=1
        return preds
