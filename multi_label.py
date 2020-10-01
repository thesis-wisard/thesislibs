import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer

import wisardpkg as wp
import time


class CrossValidation:
    def __init__(self, images, labels, k):
        self.images = images
        self.labels = labels
        self.kf = KFold(n_splits=k)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(list(self.labels))

    def validation(self, method, metrics):
        scores = {}
        k = 1
        ys_pred = []
        ys_true = []

        #print(self.images)
        for train_index, test_index in self.kf.split(self.images):
            print("k: {}".format(k), end="\r")

            X_train, X_test = self.images[train_index], self.images[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]

            y_pred = method.run(X_train, X_test, y_train, self.mlb.classes_)

            for i in range(len(y_pred)):
                ys_pred.append(np.array(y_pred[i]))
                ys_true.append(np.array(y_test[i]))
            # local_scores = metrics.calculate(self.mlb.transform(y_test), self.mlb.transform(y_pred))

            # for key in local_scores.keys():
            #     if key not in scores:
            #         scores[key] = []
                
            #     scores[key].append(local_scores[key])
            
            print("       ", end="\r")
            k += 1

        output = {}

        scores, confusion_matrix = metrics.calculate(self.mlb.transform(ys_true), self.mlb.transform(ys_pred))

        for key in scores.keys():
            output[key] = scores[key]

        output["training_time_mean"] = np.mean(method.training_time)
        output["training_time_std"] = np.std(method.training_time)

        output["classification_time_mean"] = np.mean(method.classification_time)
        output["classification_time_std"] = np.std(method.classification_time)

        return output, confusion_matrix


class Method:
    def __init__(self, addr_size, minZero=0, minOne=0):
        self.addr_size = addr_size
        self.training_time = []
        self.classification_time = []

    def run(self, X_train, X_test, y_train, classes):
        return [None]*len(y_train)

class LabelPowerset(Method):
    def run(self, X_train, X_test, y_train, classes):
        wsd = wp.Wisard(self.addr_size)

        for i, y in enumerate(y_train):
            start_time = time.time()
            y.sort()
            y_ps = "-".join(y)
            ds = wp.DataSet()
            ds.add(wp.BinInput(X_train[i]), y_ps)
            wsd.train(ds)
            self.training_time.append(time.time() - start_time)

        y_pred_ps = []
        for x in X_test:
            start_time = time.time()
            ds_test = wp.DataSet()
            ds_test.add(wp.BinInput(x))
            y_pred_ps.append(wsd.classify(ds_test)[0])
            self.classification_time.append(time.time() - start_time)

        y_pred = []
        for y in y_pred_ps:
            y_pred.append(y.split("-"))

        return y_pred

class BinaryRelevance(Method):
    def run(self, X_train, X_test, y_train, classes):
        wsds = {}
        for label in classes:
            wsds[label] = wp.Wisard(self.addr_size)

            start_time = time.time()
            for i in range(len(X_train)):
                if label in y_train[i]:
                    ds = wp.DataSet()
                    ds.add(wp.BinInput(X_train[i]), "true")
                    wsds[label].train(ds)
                else:
                    ds = wp.DataSet()
                    ds.add(wp.BinInput(X_train[i]), "false")
                    wsds[label].train(ds)
            self.training_time.append(time.time() - start_time)

        y_pred = [[]]*len(X_test)
        for label in classes:
            ds_test = wp.DataSet()
            for i in range(len(X_test)):
                ds_test.add(wp.BinInput(X_test[i]))
            start_time = time.time()
            outputs = wsds[label].classify(ds_test)
            self.classification_time.append((time.time() - start_time)/len(X_test))

            for i in range(len(outputs)):
                if outputs[i] == "true":
                    y_pred[i].append(label)
        
        return y_pred

class Metrics:
    def __init__(self, labels,  operations=[f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix], average="micro"):
    #def __init__(self, operations=[f1_score, recall_score, precision_score, accuracy_score], average="micro"):
        self.operations = operations
        self.average = average
        #labels = [item for sublist in labels for item in sublist]
        #labels = list(set(labels))
        #self.labels = labels
        #self.labels = [ str(x) for x in range(1, 40)]
        #print(labels)
       # print(type(labels))

    def calculate(self, y_true, y_pred):
        scores = {}

        for operation in self.operations:
            if operation.__name__ == "multilabel_confusion_matrix":
                confusion_matrix = operation(y_true, y_pred)
#, labels = self.labels)
                tp = 0
                tn = 0 
                fp = 0
                fn = 0
                for label in confusion_matrix:
                    tp += label[0][0]
                    fp += label[0][1]
                    fn += label[1][0] 
                    fp += label[1][1]
                scores["parcial_accuracy"] = (tp+tn)/(tp+fp+fn+tn)
            else:
                if ((operation.__name__ != "accuracy_score") and (operation.__name__ != "roc_auc_score")):
                    scores[operation.__name__] = operation(y_true, y_pred, average=self.average)
                else:
                    scores[operation.__name__] = operation(y_true, y_pred)

        return scores, confusion_matrix

class ClusLabelPowerset(Method):

    def __init__(self, addr_size, minScore, threshold, discriminatorLimit):
        super().__init__(addr_size)
        self.minScore = minScore
        self.threshold = threshold
        self.discriminatorLimit = discriminatorLimit

    def run(self, X_train, X_test, y_train, classes):
        clus = wp.ClusWisard(self.addr_size, self.minScore, self.threshold, self.discriminatorLimit)

        for i, y in enumerate(y_train):
            start_time = time.time()
            y.sort()
            y_ps = "-".join(y)
            ds = wp.DataSet()
            ds.add(wp.BinInput(X_train[i]), y_ps)
            clus.train(ds)
            self.training_time.append(time.time() - start_time)

        y_pred_ps = []
        for x in X_test:
            start_time = time.time()
            ds_test = wp.DataSet()
            ds_test.add(x)
            y_pred_ps.append(clus.classify(ds_test)[0])
            self.classification_time.append(time.time() - start_time)

        y_pred = []
        for y in y_pred_ps:
            y_pred.append(y.split("-"))

        return y_pred


class ClusBinaryRelevance(Method):

    def __init__(self, addr_size, minScore, threshold, discriminatorLimit):
        super().__init__(addr_size)
        self.minScore = minScore
        self.threshold = threshold
        self.discriminatorLimit = discriminatorLimit

    def run(self, X_train, X_test, y_train, classes):
        wsds = {}
        for label in classes:
            wsds[label] = wp.ClusWisard(
                self.addr_size, self.minScore, self.threshold, self.discriminatorLimit)

            start_time = time.time()
            for i in range(len(X_train)):
                ds = wp.DataSet()
                if label in y_train[i]:
                    ds.add(wp.BinInput(X_train[i]), "true")
                    wsds[label].train(ds)
                else:
                    ds.add(wp.BinInput(X_train[i]), "false")
                    wsds[label].train(ds)
            self.training_time.append(time.time() - start_time)

        y_pred = [[]]*len(X_test)
        ds_test = wp.DataSet()
        for i in range(len(X_test)):
            ds_test.add(X_test[i])
        for label in classes:
            start_time = time.time()
            outputs = wsds[label].classify(ds_test)
            self.classification_time.append(
                (time.time() - start_time)/len(X_test))

            for i in range(len(outputs)):
                if outputs[i] == "true":
                    y_pred[i].append(label)

        return y_pred
