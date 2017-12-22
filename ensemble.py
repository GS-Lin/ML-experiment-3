import pickle
from sklearn import tree
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''


    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
            a: The weights of each weak classifier
            classifier: The list of the weak classifier
        '''
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier = weak_classifier
        self.a = np.zeros(n_weakers_limit)
        self.classifier = []
        pass

    def is_good_enough(self,err):
        '''Optional'''
        if err<=1e-2:
            return True
        return False

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        '''
            初始化样本权值
        '''
        w = np.zeros(X.shape[0])
        w[0:] = 1/X.shape[0]
        '''
            主迭代过程
        '''
        for m in range(self.n_weakers_limit):
            self.classifier.append(self.weak_classifier.fit(X,y,sample_weight=w))
            h = self.predict_scores(X,m)
            err = (self.err_rate(h,y,w))
            print ('错误率: ',err)
            if (err>0.5):
                print ("The weak classifier is too weak")
                break
            err = max(1e-5,err)#防止err为0
            if (self.is_good_enough(err)):#若有弱分类器已经足够好，停止迭代
                break
            self.a[m] = 0.5*math.log((1-err)/err)#计算分类器权重
            self.a[m] = round(self.a[m],3)
            z = 0 #正则项
            '''
            计算正则项
            '''
            for j in range(w.shape[0]):
                z = z+w[j]*math.exp(-self.a[m]*y[j]*h[j])
            '''
            更新样本权值
            '''
            for j in range(w.shape[0]):
                w[j] = w[j]/z*math.exp(-self.a[m]*y[j]*h[j])
            '''
            更新正则项
            '''
            for j in range(w.shape[0]):
                z = z+w[j]*math.exp(-self.a[m]*y[j]*h[j])
            print ("分类器权值：",self.a)
        return self


    def predict_scores(self, X, i):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            i: The index of which classifier will be used

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        result = self.classifier[i].predict(X)
        return result

    def err_rate(self,X,y,w):
        '''
            Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_sample,1).
            w: The weights of samples
        '''
        err = 0
        for i in range(X.shape[0]):
            if (X[i]!=y[i]):
                err = err+w[i]
        return err

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        result = np.zeros(X.shape[0])
        for i in range(self.n_weakers_limit):
            predict = self.classifier[i].predict(X)
            result = result+self.a[i]*predict
        result[result<=threshold] = -1
        result[result>threshold] = 1
        return result

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
