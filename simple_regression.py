import numpy as np


class SimpleRegression(object):
    ''' Interface for a simple regression class '''

    train = NotImplemented

    predict_item = NotImplemented


class WeightedRidgeRegression(SimpleRegression):
    ''' Fits a ridge regression with weighted examples'''

    def __init__(self, xValues, yValues, xWeights, reg_param=0.8, norm=True):
        self.reg_param = reg_param
        self.numSamples, self.numFeatures = np.shape(xValues)
        # weight_0, weight_1,...., weight_numFeatures
        self.weights = [0]*(self.numFeatures+1)
        self.norm = norm
        if norm:
            xValues = np.apply_along_axis(self.normalize, 1, xValues)
        # pad samples with 1's to deal with intercept (weight_0)
        self.xValues = np.concatenate((np.ones((1, self.numSamples)).T,
                                       xValues), 1)
        self.yValues = yValues
        self.xWeights = xWeights  # Example weights
        self.train()

    def normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def train(self):
        ''' Solves for weighted ridge regression (L2-reg linear regression
        weight weighted examples) weights
            solve Aw = b for w, where
            A = (X^T*W*X + lambda*I)
            W = diagonal weight matrix (example weights)
            w = weights (what you're solving for)
            b = X^T*W*y'''
        W = np.diag(self.xWeights)
        A = (np.dot(np.dot(self.xValues.T, W), self.xValues) +
             self.reg_param * np.eye(self.numFeatures + 1))
        b = np.dot(np.dot(self.xValues.T, W), self.yValues.T)
        self.weights = np.linalg.solve(A, b)
        self.weights = self.weights.T

    def predict_item(self, test_vector):
        ''' Returns prediction for single test vector '''
        if self.norm:
            test_vector = self.normalize(test_vector)
        return np.dot(np.append(np.array([1]), test_vector), self.weights.T)


class RidgeRegression(SimpleRegression):
    ''' Simple ridge regression class. Uses np linear algebra functions to
    solve closed form solution for weights'''
    def __init__(self, xValues, yValues, reg_param=0.8, norm=True):
        self.reg_param = reg_param
        self.numSamples, self.numFeatures = np.shape(xValues)
        # weight_0, weight_1,...., weight_numFeatures
        self.weights = [0]*(self.numFeatures+1)
        self.norm = norm
        if norm:
            xValues = np.apply_along_axis(self.normalize, 1, xValues)
        # pad samples with 1's to deal with intercept (weight_0)
        self.xValues = np.concatenate((np.ones((1, self.numSamples)).T,
                                       xValues), 1)
        self.yValues = yValues
        self.train()

    def normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def train(self):
        ''' Solves for ridge regression (L2-reg linear regression) weights
            solve Aw = b for w, where
            A = (X^T*X + lambda*I)
            w = weights (what you're solving for)
            b = X^T*y'''
        A = (np.dot(self.xValues.T, self.xValues) +
             self.reg_param * np.eye(self.numFeatures + 1))
        b = np.dot(self.xValues.T, self.yValues.T)
        self.weights = np.linalg.solve(A, b)
        self.weights = self.weights.T

    def predict_item(self, test_vector):
        ''' Returns prediction for single test vector '''
        if self.norm:
            test_vector = self.normalize(test_vector)
        return np.dot(np.append(np.array([1]), test_vector), self.weights.T)
