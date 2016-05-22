from __future__ import division
from types import *
import sys
import os
import time
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, grid_search, preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from TweetParser import TweetParser, Tweet
from Tokenizer import Tokenizer


class tweet_feature:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.totalCount = 0
        self.count = 0
        self.stats = 0.0
        self.build_fieldinfo()
        pass
    
    def print_stats(self):
        self.stats = self.count/self.totalCount
        print self.name + " : " + str(self.stats  * 100) + " %"
    #features = ["favorite_count", "favorited", "is_quote_status","possibly_sensitive","possibly_sensitive_appealable","retweet_count","truncated", "text", "geo", "place" ]
    def build_fieldinfo(self): 
        if self.name == "possibly_sensitive":
            self.type = bool    
        elif self.name == "favorited":
            self.type = bool
        elif self.name == "is_quote_status":
            self.type = bool        
        elif self.name == "possibly_sensitive_appealable":
            self.type = bool
        elif self.name == "truncated":
            self.type = bool
        elif self.name == "favorite_count":
            self.type = int
        elif self.name == "retweet_count":
            self.type = int
                     
class MetaDataSentimentAnalisis:
    
    def __init__(self, kernel_type):
        self.kernel = kernel_type
        self.tokenizer = Tokenizer()
        self.__init_classifier(kernel_type)
        pass
            
    def __init_classifier(self, kernel_type):
        if kernel_type == 'rbf':
            self.classifier = svm.SVC(C=1, gamma=0.0000001)
        elif kernel_type == 'linear':
            self.classifier = svm.SVC(kernel='linear')
        elif kernel_type == 'liblinear':
            self.classifier = svm.LinearSVC()
        else:
            self.classifier = svm.SVC()
        self.vectorizer = TfidfVectorizer(min_df=5,
                                     max_df = 0.8,
                                     sublinear_tf=True,
                                     use_idf=True)
                    
        
    def train_text(self, train_data_path, metadata_path, 
              train = True,
              parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]},
              store = False,
              storepath = ""):
        parser = TweetParser() 
        tweets = parser.parse(train_data_path, metadata_path)
        train_data = []
        train_labels = []
        clean_train = ""
        polarity = "NONE"
        if store and not train:
            print ("store needs train ... train=True")
            train = True
            
        for tweet in tweets:
            clean_train = self.tokenizer.cleanText(tweet.content)
            train_data.append(clean_train)
            polarity = self.checkPolarity(tweet.polarity)
            train_labels.append(polarity)
            #print clean_train
            #print polarity
        # Create feature vectors
        train_vectors = self.vectorizer.fit_transform(train_data)
        if train:
            Cs = [0.001, 0.01, 0.1, 1, 10]
            gammas = [0.001, 0.01, 0.1, 1]
            param_grid = {'C': Cs, 'gamma' : gammas}
            self.classifier = grid_search.GridSearchCV(self.classifier, param_grid, cv=3, n_jobs=4, verbose=1)
            self.classifier.fit(train_vectors, train_labels)
        if store:
            joblib.dump(self.classifier, storepath) 
        return train_labels
    
    def svc_param_selection(self,X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=4, verbose=1)
        grid_search.fit(X, y)
        grid_search.best_params_
        return grid_search.best_params_

    def train_features(self, train_data_path, metadata_path, features= [],
                        train = True, store = True, storepath = ""):
        (X, train_labels) = self.buildFeaturesFromCorpus(train_data_path, metadata_path, features)
        if train:
            Cs = [0.001, 0.01, 0.1, 1, 10]
            gammas = [0.001, 0.01, 0.1, 1]
            param_grid = {'C': Cs, 'gamma' : gammas}
            self.classifier = grid_search.GridSearchCV(self.classifier, param_grid, cv=3, n_jobs=4, verbose=1)
            self.classifier.fit(X, train_labels)
        if store:
            joblib.dump(self.classifier, storepath)
        return train_labels
             
    def buildFeaturesFromCorpus(self,train_data_path, metadata_path, features= [],
                                 test = False):
        parser = TweetParser() 
        tweets = parser.parse(train_data_path, metadata_path)
        train_data = []
        train_labels = []
        featuresDict ={}
        for tweet in tweets:
            if not tweet.jsonData:
                continue
            for feature in features:
                if feature not in featuresDict:
                    featuresDict[feature] = tweet_feature(feature)#creamos la feature
                if feature == "geo" or feature == "place" or feature == "favorited":
                        if feature in tweet.jsonData:
                            pass
                            #print str(type(tweet.jsonData[feature]))
                            #print tweet.jsonData[feature]
                if feature in tweet.jsonData:
                    dataValue = tweet.jsonData[feature]
                    featuresDict[feature].data.append(dataValue)
                    featuresDict[feature].count +=1
                    featuresDict[feature].totalCount += 1
                else:
                    if featuresDict[feature].type == bool: 
                        featuresDict[feature].data.append(False)
                    elif featuresDict[feature].type == int:
                        featuresDict[feature].data.append(0)
                    else:
                        featuresDict[feature].data.append("null")
                    featuresDict[feature].totalCount += 1
                    
            polarity = self.checkPolarity(tweet.polarity)
            train_labels.append(polarity)   
        for feature in features:
            featuresDict[feature].print_stats()
            if feature == "text":
                Xt = self.buildFeatureDataMatrix(featuresDict[feature], test)
                train_data.append(Xt)
            else:     
                train_data.append(self.buildFeatureDataMatrix(featuresDict[feature], test))
        if train_data:                  
            #Xm = scipy.sparse.csc_matrix(train_data)
            #Xm = Xm.transpose(True)     
            # X : sparse matrix, [n_samples, n_features]
            #    Tf-idf-weighted document-term matrix.    
            #print Xt.shape
            #print Xm.shape
            X = scipy.sparse.hstack(train_data)
        else:
            X = Xt    
        return (X,train_labels)
                
    def test(self, test_data_path = "", metadata_path = "", load = False, model = ""):
        parser = TweetParser() 
        tweets = parser.parse(test_data_path, metadata_path)
        test_data = []
        for tweet in tweets:
            test_data.append(self.tokenizer.cleanText(tweet.content))
            
        test_vectors = self.vectorizer.transform(test_data)
        if load and model:
            self.classifier = joblib.load(model) 
        predictions = self.classifier.predict(test_vectors)
        return predictions
    
    def test_features(self, test_data_path = "", metadata_path = "", features= [], load = False, model = ""):
        (X, train_labels) = self.buildFeaturesFromCorpus(test_data_path, metadata_path, features, True)  
        if load and model:
            self.classifier = joblib.load(model) 
        predictions = self.classifier.predict(X)
        return (predictions, train_labels)
    
    def predict(self, test_data_path = "", metadata_path = "", model = ""):  
        parser = TweetParser() 
        tweets = parser.parse(test_data_path, metadata_path)
        test_data = []
        for tweet in tweets:
            test_data.append(self.tokenizer.cleanText(tweet.content))  
            
        test_vectors = self.vectorizer.transform(test_data)
        self.classifier = joblib.load(model) 
        predictions = self.classifier.predict(test_vectors)   
        for text, prediction in test_data, predictions:
            print text
            print prediction
    
    def checkPolarity(self, polarity_elements):
        polarity = 'NONE'
        if polarity_elements:
            for polarity_element in polarity_elements:
                polarity = polarity_element
                if not polarity == 'NONE':
                    break 
        return polarity
    
    def buildFeatureDataMatrix(self, feature, test = False):
        featureData = []
        for data in feature.data:
            if feature.name == "text":
                featureData.append(self.tokenizer.cleanText(data))
            else:    
                featureData.append(data)
        if feature.name == "text":
            if not test:
                text_features = self.vectorizer.fit_transform(featureData)
            else:
                text_features=self.vectorizer.transform(featureData)       
            return text_features
        else:
            return scipy.sparse.csc_matrix(featureData).transpose(True)
             
def runTestFeature(feature = "text", features =["text"],
                   trainPath = "../data/general-tweets-train-tagged.xml",
                   trainDataPath = "../data/train",
                   testPath = "../data/general-tweets-train-tagged.xml",
                   testDataPath = "../data/train"):
    print "Run test : " + feature 
    class_labels = ['P+', 'P', 'NEU', 'N', 'N+','NONE']
    mclassifier = MetaDataSentimentAnalisis("rbf")
    t0 = time.time()
    model = "../data/"+ feature +"/model_rbf.pkl"
    train = True
    store = True
    groundtrue = mclassifier.train_features(trainPath, trainDataPath, features, train, store, model)
    t1 = time.time()
    print "Time training " + str(t1-t0)
    t2 = time.time()
    (predictions, groundtrue) = mclassifier.test_features(testPath, testDataPath,
                                                           features, True, model)
    t3 = time.time()
    print "Time test " + str(t3-t2)
    matrix = confusion_matrix(groundtrue, predictions)
    print matrix
    report = classification_report(groundtrue, predictions)
    print report
    fresult = open("../data/"+ feature +"/result.txt", 'w+')
    fresult.write(str(matrix))
    fresult.write(report)
    fresult.close()
             
                
def main():
    #features = ["text", "favorite_count", "retweet_count", "favorited", "is_quote_status","possibly_sensitive","possibly_sensitive_appealable","truncated", "geo", "place" ]
    #train-test text feature
    runTestFeature("1-text_train", ["text"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text favorite_count
    runTestFeature("2-favorite_count_train", ["text", "favorite_count"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text retweet_count
    runTestFeature("3-retweet_count_train", ["text", "retweet_count"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text favorited
    runTestFeature("4-favorited_train", ["text", "favorited"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text is_quote_status
    runTestFeature("5-is_quote_status_train", ["text", "is_quote_status"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text possibly_sensitive
    runTestFeature("6-possibly_sensitive_train", ["text", "possibly_sensitive"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text possibly_sensitive_appealable    
    runTestFeature("7-possibly_sensitive_appealable_train", ["text", "possibly_sensitive_appealable"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text truncated
    runTestFeature("8-truncated_train", ["text", "truncated"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    #train-test text all
    runTestFeature("9-all_train", ["text", "favorite_count", "retweet_count", "favorited", "is_quote_status","possibly_sensitive","possibly_sensitive_appealable","truncated"], "../data/general-tweets-train-tagged.xml", "../data/train",
                    "../data/general-tweets-test-tagged.xml","../data/test" )
    
    #train with test and test with train
        #train-test text feature
    runTestFeature("11-text_test", ["text"], "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text favorite_count
    runTestFeature("12-favorite_count_test", ["text", "favorite_count"], "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text retweet_count
    runTestFeature("13-retweet_count_test", ["text", "retweet_count"], "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text favorited
    runTestFeature("14-favorited_test", ["text", "favorited"], "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text is_quote_status
    runTestFeature("15-is_quote_status_test", ["text", "is_quote_status"], "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text possibly_sensitive
    runTestFeature("16-possibly_sensitive_test", ["text", "possibly_sensitive"], "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text possibly_sensitive_appealable    
    runTestFeature("17-possibly_sensitive_appealable_test", ["text", "possibly_sensitive_appealable"], "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text truncated
    runTestFeature("18-truncated_test", ["text", "truncated"], "../data/general-tweets-train-tagged.xml", "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    #train-test text all
    runTestFeature("19-all_test", ["text", "favorite_count", "retweet_count", "favorited", "is_quote_status","possibly_sensitive","possibly_sensitive_appealable","truncated"],  "../data/general-tweets-test-tagged.xml","../data/test",
                     "../data/general-tweets-train-tagged.xml", "../data/train")
    
if __name__ == '__main__':
        main()       